from .trt_sdxl_base import SD_TRT
import torch
from typing import Optional, List, Union, Tuple
from diffusers.utils.torch_utils import randn_tensor


class SDXL_T2I_Pipeline:
    def __init__(self, base: SD_TRT):
        self.base = base

    def initialize_latents(self, batch_size, latent_height, latent_width, generator, latents=None):
        latents_shape = (batch_size, 4, latent_height, latent_width)
        if latents is None:
            latents = randn_tensor(latents_shape, generator=generator, device=self.base.device, dtype=torch.float16)
        else:
            latents = latents.to(self.base.device)
        latents = latents * self.base.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def infer(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
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

        if self.base.lowvram:
            self.base.vae.cpu()
        self.base.text_encoder.to(self.base.device)
        self.base.text_encoder_2.to(self.base.device)

        if height is None: height = 1024 
        if width is None: width = 1024 
        if num_images_per_prompt is None: num_images_per_prompt = 1

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        do_cfg = guidance_scale > 0
        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        self.base.loadResources(height, width, batch_size*num_images_per_prompt, do_cfg)

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

        # Time embeddings
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

