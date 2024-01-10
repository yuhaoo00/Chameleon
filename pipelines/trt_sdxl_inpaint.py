import torch
from cuda import cudart
from typing import Optional, List, Union, Tuple, Any, Dict
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import PipelineImageInput
from diffusers.image_processor import VaeImageProcessor

from .trt_sdxl_base import SD_TRT

def retrieve_latents(encoder_output, generator):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class SDXL_Inpaint_Pipeline:
    def __init__(self, base: SD_TRT):
        self.base = base

        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.base.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        
    def initialize_latents(
            self,
            batch_size,
            height,
            width,
            dtype,
            generator,
            latents=None,
            image=None,
            timestep=None,
            is_strength_max=True,
            return_image_latents=True,):

        shape = (batch_size, 4, height//self.base.vae_scale_factor, width//self.base.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if image.shape[1] == 4:
            image_latents = image.to(device=self.base.device, dtype=dtype)
        elif return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=self.base.device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)
        image_latents = image_latents.repeat(batch_size//image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=self.base.device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.base.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.base.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(self.base.device)
            latents = noise * self.base.scheduler.init_noise_sigma

        outputs = (latents, noise)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs
    
    def prepare_mask(
        self, mask, batch_size, height, width, dtype, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height//self.base.vae_scale_factor, width//self.base.vae_scale_factor)
        )
        mask = mask.to(device=self.base.device, dtype=dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size//mask.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        return mask
    
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        dtype = image.dtype
        if self.base.needs_upcasting:
            image = image.float()
            self.base.vae.to(dtype=torch.float32)

        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.base.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.base.vae.encode(image), generator=generator)

        if self.base.needs_upcasting:
            self.base.vae.to(dtype)

        image_latents = image_latents.to(dtype)
        image_latents = self.base.vae.config.scaling_factor * image_latents

        return image_latents
    
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        timesteps = self.base.scheduler.timesteps[t_start * self.base.scheduler.order :]

        return timesteps, num_inference_steps - t_start
    

    def denoise_latent_inpaint(self,
        latents,
        text_embeddings,
        timesteps,
        generator,
        mask,
        noise,
        image_latents,
        eta=0.0,
        guidance_scale=7.5,
        add_kwargs={}):

        do_cfg = guidance_scale > 1.0

        extra_step_kwargs = self.base.prepare_extra_step_kwargs(generator, eta)

        cudart.cudaEventRecord(self.base.events['denoise-start'], 0)
        for i, timestep in enumerate(timesteps):
            #timestep = torch.tensor([999.]).to(latents.device)

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = self.base.scheduler.scale_model_input(latent_model_input, timestep)

            # Predict the noise residual
            params = {"sample": latent_model_input, "timestep": timestep.reshape(-1).half(), "encoder_hidden_states": text_embeddings}
            if add_kwargs: params.update(add_kwargs)
            noise_pred = self.base.unet_engine.infer(params, self.base.stream, use_cuda_graph=self.base.use_cuda_graph)['out_sample']

            # perform guidance
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.base.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]

            init_latents_proper = image_latents
            if do_cfg:
                init_mask, _ = mask.chunk(2)
            else:
                init_mask = mask

            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                init_latents_proper = self.base.scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([noise_timestep])
                )

            latents = (1 - init_mask) * init_latents_proper + init_mask * latents

        cudart.cudaEventRecord(self.base.events['denoise-stop'], 0)
        return latents

    
    @torch.no_grad()
    def infer(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,):

        if height is None: height = 1024 
        if width is None: width = 1024 
        if num_images_per_prompt is None: num_images_per_prompt = 1

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_cfg = guidance_scale > 0

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.base.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            clip_skip=clip_skip,
        )

        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        
        self.base.scheduler.set_timesteps(num_inference_steps, device=self.base.device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        latent_timestep = timesteps[:1].repeat(batch_size*num_images_per_prompt)
        is_strength_max = strength == 1.0

        init_image = self.base.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=prompt_embeds.dtype)

        mask = self.mask_processor.preprocess(mask_image, height=height, width=width)

        latents, noise, image_latents = self.initialize_latents(
            batch_size*num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_image_latents=True,
        )

        mask = self.prepare_mask(
            mask,
            batch_size*num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            do_cfg,
        )

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.base.loadResources(height, width, batch_size*num_images_per_prompt, do_cfg)

        # Time embeddings
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype).repeat(batch_size*num_images_per_prompt, 1)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.base.device) if do_cfg else add_time_ids.to(self.base.device)
        add_kwargs = {'add_text_embeds': pooled_prompt_embeds, 'add_time_ids': add_time_ids}

        latents = self.denoise_latent_inpaint(
            latents=latents,
            text_embeddings=prompt_embeds,
            timesteps=timesteps,
            generator=generator,
            mask=mask,
            noise=noise,
            image_latents=image_latents,
            eta=eta,
            guidance_scale=guidance_scale,
            add_kwargs=add_kwargs,
        )

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            if self.base.needs_upcasting:
                self.base.upcast_vae()
            
            latents = latents.to(next(iter(self.base.vae.post_quant_conv.parameters())).dtype)

            images = self.base.vae.decode(latents/self.base.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if self.base.needs_upcasting:
                self.base.vae.to(dtype=torch.float16)
            
            images = self.base.image_processor.postprocess(images, output_type=output_type)
        else:
            images = latents

        return images

