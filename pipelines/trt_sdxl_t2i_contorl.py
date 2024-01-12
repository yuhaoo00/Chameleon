
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from cuda import cudart
from time import time
import PIL.Image
import torch
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from .trt_sdxl_base import SD_TRT
from .engine import EngineWrapper

class SDXL_T2I_CN_Pipeline:
    def __init__(
        self,
        base: SD_TRT,
        controlnet_type: str,
    ):
        self.base = base
        self.controlnet_type = controlnet_type
        self.base.engines[controlnet_type] = EngineWrapper(self.base.engine_dir / f"{controlnet_type}.plan")
        self.base.engines[controlnet_type].load()
        self.base.activateEngines()

        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.base.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

    def unload(self):
        # unload controlnet engine to save MEM
        self.base.engines[self.controlnet_type].__del__()
        self.base.engines.pop(self.controlnet_type, None)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.check_image
    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        dtype,
        do_classifier_free_guidance=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=self.base.device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image


    def initialize_latents(self, batch_size, latent_height, latent_width, generator, latents=None):
        shape = (batch_size, 4, latent_height, latent_width)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=self.base.device, dtype=torch.float16)
        else:
            latents = latents.to(self.base.device)
        latents = latents * self.base.scheduler.init_noise_sigma
        return latents


    def denoise_latent_control(self,
        latents,
        text_embeddings,
        timesteps,
        generator,
        add_text_embeds,
        add_time_ids,
        image,
        eta=0.0,
        guidance_scale=7.5,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        controlnet_conditioning_scale=1.0):

        do_cfg = guidance_scale > 1.0

        extra_step_kwargs = self.base.prepare_extra_step_kwargs(generator, eta)

        controlnet_keep = []
        for i in range(len(timesteps)):
            keep = 1.0 - float(i / len(timesteps) < control_guidance_start or (i + 1) / len(timesteps) > control_guidance_end)
            controlnet_keep.append(keep)

        for i, timestep in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = self.base.scheduler.scale_model_input(latent_model_input, timestep)

            # Predict the noise residual
            # Unet Encoder
            trtTimeStart = time()

            cudart.cudaEventRecord(self.base.event_cn, self.base.stream_cn)
            cudart.cudaStreamWaitEvent(self.base.stream, self.base.event_cn, cudart.cudaEventWaitDefault)
            params = {
                "sample": latent_model_input, 
                "timestep": timestep.reshape(-1).half(), 
                "encoder_hidden_states": text_embeddings,
                "add_text_embeds": add_text_embeds,
                "add_time_ids": add_time_ids
            }
            out = self.base.engines["unet_encoder"].infer(params, self.base.stream, use_cuda_graph=self.base.use_cuda_graph)
            cudart.cudaEventRecord(self.base.event, self.base.stream)

            # ContorlNet
            cudart.cudaStreamWaitEvent(self.base.stream_cn, self.base.event, cudart.cudaEventWaitDefault)
            cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
            print(cond_scale)
            controlnet_params = {
                "controlnet_cond": image,
                "conditioning_scale": cond_scale,
            }
            params.update(controlnet_params)
            input_names = params.keys()

            out_res = self.base.engines[self.controlnet_type].infer(params, self.base.stream_cn, use_cuda_graph=self.base.use_cuda_graph)
            cudart.cudaEventRecord(self.base.event_cn, self.base.stream_cn)

            cudart.cudaEventSynchronize(self.base.event_cn)
            trtTimeEnd = time()
            print("Control+Encoder = %6.3fms" % ((trtTimeEnd - trtTimeStart) * 1000))

            # Unet Decoder
            params = {"encoder_hidden_states": text_embeddings,
                      "emb": out["emb"],}
            for k in out_res.keys(): # inputs + downs + mid
                if k not in input_names: # downs + mid
                    params[k] = out[k] + out_res[k]

            noise_pred = self.base.engines["unet_decoder"].infer(params, self.base.stream, use_cuda_graph=self.base.use_cuda_graph)['out_sample']

            # perform guidance
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.base.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]

        return latents

    @torch.no_grad()
    def infer(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        controlnet_conditioning_scale: float = 1.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        clip_skip: Optional[int] = None,
        lowvram: bool = False,
    ):
        # Define parameters
        self.lowvram = lowvram
        if height is None: height = 1024 
        if width is None: width = 1024 
        if num_images_per_prompt is None: num_images_per_prompt = 1

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_cfg = guidance_scale > 1.0

        # Encode prompt
        if self.lowvram:
            self.base.vae.cpu()
        
        self.base.text_encoder.to(self.base.device)
        self.base.text_encoder_2.to(self.base.device)
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

        if self.lowvram:
            self.base.text_encoder.cpu()
            self.base.text_encoder_2.cpu()

        # Prepare contorl hint images
        image = self.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            dtype=prompt_embeds.dtype,
            do_classifier_free_guidance=do_cfg,
        )
        height, width = image.shape[-2:]

        self.base.loadResources(height, width, batch_size*num_images_per_prompt, do_cfg)

        # Time embeddings
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype).repeat(batch_size*num_images_per_prompt, 1)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.base.device) if do_cfg else add_time_ids.to(self.base.device)

        self.base.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.base.scheduler.timesteps

        latents = self.initialize_latents(
                batch_size=batch_size*num_images_per_prompt,
                latent_height=(height//self.base.vae_scale_factor),
                latent_width=(width//self.base.vae_scale_factor),
                generator=generator,
            )
        
        latents = self.denoise_latent_control(
            latents=latents,
            text_embeddings=prompt_embeds,
            timesteps=timesteps,
            generator=generator,
            add_text_embeds=pooled_prompt_embeds,
            add_time_ids=add_time_ids,
            image=image,
            eta=eta,
            guidance_scale=guidance_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            controlnet_conditioning_scale=controlnet_conditioning_scale
        )

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            if self.lowvram:
                self.base.vae.to(self.base.device)

            if self.base.needs_upcasting:
                self.base.upcast_vae()
            
            latents = latents.to(next(iter(self.base.vae.post_quant_conv.parameters())).dtype)

            images = self.base.vae.decode(latents/self.base.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if self.base.needs_upcasting:
                self.base.vae.to(dtype=torch.float16)
            
            if self.lowvram:
                self.base.vae.cpu()
            
            images = self.base.image_processor.postprocess(images, output_type=output_type)
        else:
            images = latents

        return images