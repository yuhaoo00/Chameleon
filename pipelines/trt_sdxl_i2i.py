import torch
from typing import Optional, List, Union, Tuple, Any, Dict
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import PipelineImageInput

from .trt_sdxl_base import SD_TRT

def retrieve_latents(encoder_output, generator):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class SDXL_I2I_Pipeline:
    def __init__(self, base: SD_TRT):
        self.base = base
        
    def initialize_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, generator=None):
        image = image.to(device=self.base.device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            # make sure the VAE is in float32 mode, as it overflows in float16
            if self.base.needs_upcasting:
                image = image.float()
                self.base.upcast_vae()

            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.base.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.base.vae.encode(image), generator=generator)

            if self.base.needs_upcasting:
                self.base.vae.to(dtype=torch.float16)

            init_latents = init_latents.to(dtype)
            init_latents = self.base.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)


        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=self.base.device, dtype=dtype)
        # get latents
        init_latents = self.base.scheduler.add_noise(init_latents, noise, timestep)

        return init_latents
    
    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        timesteps = self.base.scheduler.timesteps[t_start * self.base.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    
    @torch.no_grad()
    def infer(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
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
        clip_skip: Optional[int] = None):

        if self.base.lowvram:
            self.base.vae.cpu()
        self.base.text_encoder.to(self.base.device)
        self.base.text_encoder_2.to(self.base.device)

        if num_images_per_prompt is None: num_images_per_prompt = 1

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        image = self.base.image_processor.preprocess(image)

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

        del negative_prompt_embeds, negative_pooled_prompt_embeds

        if self.base.lowvram:
            self.base.text_encoder.cpu()
            self.base.text_encoder_2.cpu()
        
        self.base.scheduler.set_timesteps(num_inference_steps, device=self.base.device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        latent_timestep = timesteps[:1].repeat(batch_size*num_images_per_prompt)

        latents = self.initialize_latents(
            image,
            latent_timestep,
            batch_size*num_images_per_prompt,
            num_images_per_prompt,
            prompt_embeds.dtype,
            generator,
        )

        height, width = latents.shape[-2:]
        height = height * self.base.vae_scale_factor
        width = width * self.base.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.base.loadResources(height, width, batch_size*num_images_per_prompt, do_cfg)

        # Time embeddings
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype).repeat(batch_size*num_images_per_prompt, 1)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.base.device) if do_cfg else add_time_ids.to(self.base.device)

        latents = self.base.denoise_latent(
            latents=latents,
            text_embeddings=prompt_embeds,
            timesteps=timesteps,
            generator=generator,
            eta=eta,
            guidance_scale=guidance_scale,
            add_text_embeds=pooled_prompt_embeds,
            add_time_ids=add_time_ids,
        )

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            if self.base.lowvram:
                self.base.vae.to(self.base.device)

            if self.base.needs_upcasting:
                self.base.upcast_vae()
            
            latents = latents.to(next(iter(self.base.vae.post_quant_conv.parameters())).dtype)

            images = self.base.vae.decode(latents/self.base.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if self.base.needs_upcasting:
                self.base.vae.to(dtype=torch.float16)
                
            if self.base.lowvram:
                self.base.vae.cpu()
            
            images = self.base.image_processor.postprocess(images, output_type=output_type)
        else:
            images = latents

        return images

