import importlib
import pathlib
import inspect
import torch
from cuda import cudart
from typing import Optional, List, Union, Tuple, Any, Dict

from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)

from .engine import EngineWrapper

class SD_TRT:
    def __init__(
        self,
        hf_dir="./",
        engine_dir="./",
        vae_dir=None,
        engine_config={},
        enable_dynamic_shape=True,
        device='cuda',
        scheduler_class="EulerDiscreteScheduler",
    ):
        hf_dir = pathlib.Path(hf_dir)
        engine_dir = pathlib.Path(engine_dir)
        if not hf_dir.exists():
            RuntimeError(f"The passed \'hf_dir\' is an invalid path!")
        if not engine_dir.exists():
            RuntimeError(f"The passed \'engine_dir\' is an invalid path!")

        self.device = torch.device(device)

        self.shared_device_memory = None
        self.engine_config = engine_config
        self.enable_dynamic_shape = enable_dynamic_shape
        self.use_cuda_graph = not enable_dynamic_shape

        # initialized in loadResources()
        self.stream = None
        self.events = {}

        # load Engines
        self.unet_engine = EngineWrapper(engine_dir/"unet_dy.plan")
        self.unet_engine.load()

        # load Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(engine_dir/"tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(engine_dir/"tokenizer_2")

        # load Text_encoder
        self.text_encoder = CLIPTextModel.from_pretrained(hf_dir/"text_encoder", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(self.device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(hf_dir/"text_encoder_2", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(self.device)

        # load VAE_decoder
        if vae_dir is None:
            self.vae = AutoencoderKL.from_pretrained(hf_dir/"vae_1_0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(self.device)
        else:
            self.vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=torch.float16, use_safetensors=True).to(self.device)
        self.needs_upcasting = (self.vae.dtype == torch.float16 and self.vae.config.force_upcast)
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels)-1)

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # load Scheduler
        self.scheduler_config = hf_dir/"scheduler"
        self.scheduler = getattr(importlib.import_module("diffusers"), scheduler_class).from_pretrained(self.scheduler_config)
    
    def activateEngines(self, shared_device_memory=None):
        if shared_device_memory is None:
            _, shared_device_memory = cudart.cudaMalloc(self.unet_engine.engine.device_memory_size)
        self.shared_device_memory = shared_device_memory
        # Load and activate TensorRT engines
        self.unet_engine.activate(reuse_device_memory=self.shared_device_memory)

    def loadResources(self, image_height, image_width, batch_size, do_cfg):
        # Create CUDA events and stream
        self.events = {}
        self.events['denoise-start'] = cudart.cudaEventCreate()[1]
        self.events['denoise-stop'] = cudart.cudaEventCreate()[1]
        err, self.stream = cudart.cudaStreamCreate()

        # Allocate buffers for TensorRT engine bindings
        input_shape = {}
        for k, v in self.engine_config.items():
            tmp = v['opt']
            if k != 'timestep':
                tmp[0] = batch_size*2 if do_cfg else batch_size
            if k == 'sample':
                tmp[2], tmp[3] = image_height//self.vae_scale_factor, image_width//self.vae_scale_factor
            input_shape[k] = tuple(tmp)
                    
        self.unet_engine.allocate_buffers(input_shape, device=self.device)

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e)

        del self.unet_engine

        if self.shared_device_memory:
            cudart.cudaFree(self.shared_device_memory)

        cudart.cudaStreamDestroy(self.stream)
        del self.stream

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width, generator):
        latents_shape = (batch_size, unet_channels, latent_height, latent_width)
        latents = randn_tensor(latents_shape, generator=generator, device=self.device, dtype=torch.float16)
        #latents = torch.randn(latents_shape, device=self.device, dtype=torch.float16, generator=generator)
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def change_scheduler(self, scheduler_class):
        self.scheduler = getattr(importlib.import_module('diffusers'), scheduler_class).from_pretrained(self.scheduler_config)
        
    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
    ):
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    print(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def denoise_latent(self,
        latents,
        text_embeddings,
        timesteps,
        generator,
        eta=0.0,
        guidance_scale=7.5,
        add_kwargs={}):

        do_cfg = guidance_scale > 1.0

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        cudart.cudaEventRecord(self.events['denoise-start'], 0)
        for step_index, timestep in enumerate(timesteps):
            #timestep = torch.tensor([999.]).to(latents.device)

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

            # Predict the noise residual
            params = {"sample": latent_model_input, "timestep": timestep.reshape(-1).half(), "encoder_hidden_states": text_embeddings}
            if add_kwargs: params.update(add_kwargs)
            noise_pred = self.unet_engine.infer(params, self.stream, use_cuda_graph=self.use_cuda_graph)['out_sample']

            # perform guidance
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False)[0]

        cudart.cudaEventRecord(self.events['denoise-stop'], 0)
        return latents
    
    @torch.no_grad()
    def infer(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
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
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,):

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        do_cfg = guidance_scale > 0
        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        self.loadResources(height, width, batch_size*num_images_per_prompt, do_cfg)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
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

        # Time embeddings
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype).repeat(batch_size, 1)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.device) if do_cfg else add_time_ids.to(self.device)
        add_kwargs = {'add_text_embeds': pooled_prompt_embeds, 'add_time_ids': add_time_ids}

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        latents = self.initialize_latents(
                batch_size=batch_size,
                unet_channels=4,
                latent_height=(height//self.vae_scale_factor),
                latent_width=(width//self.vae_scale_factor),
                generator=generator,
            )
        
        latents = self.denoise_latent(
            latents=latents,
            text_embeddings=prompt_embeds,
            timesteps=timesteps,
            generator=generator,
            eta=eta,
            guidance_scale=guidance_scale,
            add_kwargs=add_kwargs,
        )

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            if self.needs_upcasting:
                self.upcast_vae()
            
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            images = self.vae.decode(latents/self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if self.needs_upcasting:
                self.vae.to(dtype=torch.float16)
            
            images = self.image_processor.postprocess(images, output_type=output_type)
        else:
            images = latents

        return images

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)


