# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cuda import cudart
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random
import torchvision
from PIL import Image
from tqdm.auto import tqdm
from .trt_sdxl_base import SD_TRT

from diffusers.utils.torch_utils import randn_tensor

def gaussian_kernel(kernel_size=3, sigma=1.0, channels=3):
    x_coord = torch.arange(kernel_size)
    gaussian_1d = torch.exp(
        -((x_coord - (kernel_size - 1) / 2) ** 2) / (2 * sigma**2)
    )
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    kernel = gaussian_2d[None, None, :, :].repeat(channels, 1, 1, 1)

    return kernel


def gaussian_filter(latents, kernel_size=3, sigma=1.0):
    channels = latents.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, channels).to(
        latents.device, latents.dtype
    )
    blurred_latents = F.conv2d(
        latents, kernel, padding=kernel_size // 2, groups=channels
    )

    return blurred_latents


class SDXL_DemoFusion:
    def __init__(self, base: SD_TRT):
        self.base = base

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    def initialize_latents(self, batch_size, latent_height, latent_width, generator):
        latents_shape = (batch_size, 4, latent_height, latent_width)
        latents = randn_tensor(latents_shape, generator=generator, device=self.base.device, dtype=torch.float16)
        latents = latents * self.base.scheduler.init_noise_sigma
        return latents

    def get_views(self, height, width, window_size=128, stride=64, random_jitter=False):
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        # if panorama's height/width < window_size, num_blocks of height/width should return 1
        height //= self.base.vae_scale_factor
        width //= self.base.vae_scale_factor
        num_blocks_height = (
            int((height - window_size) / stride - 1e-6) + 2
            if height > window_size
            else 1
        )
        num_blocks_width = (
            int((width - window_size) / stride - 1e-6) + 2 if width > window_size else 1
        )
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size

            if h_end > height:
                h_start = int(h_start + height - h_end)
                h_end = int(height)
            if w_end > width:
                w_start = int(w_start + width - w_end)
                w_end = int(width)
            if h_start < 0:
                h_end = int(h_end - h_start)
                h_start = 0
            if w_start < 0:
                w_end = int(w_end - w_start)
                w_start = 0

            if random_jitter:
                jitter_range = (window_size - stride) // 4
                w_jitter = 0
                h_jitter = 0
                if (w_start != 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range, jitter_range)
                elif (w_start == 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range, 0)
                elif (w_start != 0) and (w_end == width):
                    w_jitter = random.randint(0, jitter_range)
                if (h_start != 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range, jitter_range)
                elif (h_start == 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range, 0)
                elif (h_start != 0) and (h_end == height):
                    h_jitter = random.randint(0, jitter_range)
                h_start += h_jitter + jitter_range
                h_end += h_jitter + jitter_range
                w_start += w_jitter + jitter_range
                w_end += w_jitter + jitter_range

            views.append((h_start, h_end, w_start, w_end))
        return views

    def tiled_decode(self, latents, current_height, current_width):
        sample_size = 128
        core_size = sample_size // 4
        core_stride = core_size
        pad_size = sample_size // 4 * 3
        decoder_view_batch_size = 1

        if self.lowvram:
            core_stride = core_size // 2
            pad_size = core_size

        views = self.get_views(
            current_height, current_width, stride=core_stride, window_size=core_size
        )
        views_batch = [
            views[i : i + decoder_view_batch_size]
            for i in range(0, len(views), decoder_view_batch_size)
        ]
        latents_ = F.pad(
            latents, (pad_size, pad_size, pad_size, pad_size), "constant", 0
        )
        image = torch.zeros(latents.size(0), 3, current_height, current_width).to(
            latents.device
        )
        count = torch.zeros_like(image).to(latents.device)
        # get the latents corresponding to the current view coordinates
        with self.progress_bar(total=len(views_batch)) as progress_bar:
            for j, batch_view in enumerate(views_batch):
                vb_size = len(batch_view)
                latents_for_view = torch.cat(
                    [
                        latents_[
                            :,
                            :,
                            h_start : h_end + pad_size * 2,
                            w_start : w_end + pad_size * 2,
                        ]
                        for h_start, h_end, w_start, w_end in batch_view
                    ]
                ).to(self.base.vae.device)
                image_patch = self.base.vae.decode(
                    latents_for_view / self.base.vae.config.scaling_factor, return_dict=False
                )[0]
                h_start, h_end, w_start, w_end = views[j]
                h_start, h_end, w_start, w_end = (
                    h_start * self.base.vae_scale_factor,
                    h_end * self.base.vae_scale_factor,
                    w_start * self.base.vae_scale_factor,
                    w_end * self.base.vae_scale_factor,
                )
                p_h_start, p_h_end, p_w_start, p_w_end = (
                    pad_size * self.base.vae_scale_factor,
                    image_patch.size(2) - pad_size * self.base.vae_scale_factor,
                    pad_size * self.base.vae_scale_factor,
                    image_patch.size(3) - pad_size * self.base.vae_scale_factor,
                )
                image[:, :, h_start:h_end, w_start:w_end] += image_patch[
                    :, :, p_h_start:p_h_end, p_w_start:p_w_end
                ].to(latents.device)
                count[:, :, h_start:h_end, w_start:w_end] += 1
                progress_bar.update()
        image = image / count

        return image

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
        clip_skip: Optional[int] = None,
        ################### DemoFusion specific parameters ####################
        view_batch_size: int = 16,
        multi_decoder: bool = True,
        stride: Optional[int] = 64,
        cosine_scale_1: Optional[float] = 3.0,
        cosine_scale_2: Optional[float] = 1.0,
        cosine_scale_3: Optional[float] = 1.0,
        sigma: Optional[float] = 1.0,
        show_image: bool = False,
        lowvram: bool = False,
        image_lr: Optional[Image.Image] = None,
    ):
        # Default height and width to unet
        if height is None: height = 1024 
        if width is None: width = 1024 

        x1_size = 128 * self.base.vae_scale_factor

        height_scale = height / x1_size
        width_scale = width / x1_size
        scale_num = int(max(height_scale, width_scale))
        aspect_ratio = min(height_scale, width_scale) / max(height_scale, width_scale)

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        do_cfg = guidance_scale > 0

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        self.lowvram = lowvram
        if self.lowvram:
            self.base.vae.cpu()
            self.base.text_encoder.to(self.base.device)
            self.base.text_encoder_2.to(self.base.device)

        # 3. Encode input prompt
        self.base.loadResources(x1_size, x1_size, batch_size, do_cfg)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.base.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
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

        # Time embeddings
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype).repeat(batch_size, 1)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.base.device) if do_cfg else add_time_ids.to(self.base.device)

        self.base.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.base.scheduler.timesteps

        
        # Prepare latent variables
        latents = self.initialize_latents(
            batch_size,
            height//scale_num,
            width//scale_num,
            generator,
        )

        extra_step_kwargs = self.base.prepare_extra_step_kwargs(generator, eta)
        
        output_images = []

        ############################################################### Phase 1 #################################################################
        cudart.cudaEventRecord(self.base.events['denoise-start'], 0)

        if self.lowvram:
            self.base.text_encoder.cpu()
            self.base.text_encoder_2.cpu()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if image_lr == None:
                print("### Phase 1 Denoising ###")
                for i, t in enumerate(timesteps):
                    if self.lowvram:
                        self.base.vae.cpu()

                    latents_for_view = latents

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        latents.repeat_interleave(2, dim=0)
                        if do_cfg
                        else latents
                    )
                    latent_model_input = self.base.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # predict the noise residual
                    params = {"sample": latent_model_input, "timestep": t.reshape(-1).half(), "encoder_hidden_states": prompt_embeds, "add_text_embeds": pooled_prompt_embeds, "add_time_ids": add_time_ids}
                    noise_pred = self.base.unet_engine.infer(params, self.base.stream, use_cuda_graph=self.base.use_cuda_graph)['out_sample']

                    # perform guidance
                    if do_cfg:
                        noise_pred_uncond, noise_pred_text = (
                                noise_pred[::2],
                                noise_pred[1::2],
                            )
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.base.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) % self.base.scheduler.order == 0):
                        progress_bar.update()
                        
                del (
                    latents_for_view,
                    latent_model_input,
                    noise_pred,
                    noise_pred_text,
                    noise_pred_uncond,
                )
            else:
                print("### Phase Encoding ###")
                image_lr = self.preprocess_imglr(image_lr).to(self.base.device)
                self.base.vae.to(self.base.device)
                latents = self.base.vae.encode(image_lr)
                latents = latents.latent_dist.sample() * self.base.vae.config.scaling_factor

            anchor_mean = latents.mean()
            anchor_std = latents.std()
            if self.lowvram:
                latents = latents.cpu()
                torch.cuda.empty_cache()

            if not output_type == "latent":
                # make sure the VAE is in float32 mode, as it overflows in float16
                if self.lowvram:
                    self.base.vae.to(self.base.device)

                if self.base.needs_upcasting:
                    self.base.upcast_vae()
                    latents = latents.to(
                        next(iter(self.base.vae.post_quant_conv.parameters())).dtype
                    )
                if self.lowvram and multi_decoder:
                    current_width_height = (
                        128 * self.base.vae_scale_factor
                    )
                    image = self.tiled_decode(
                        latents, current_width_height, current_width_height
                    )
                else:
                    image = self.base.vae.decode(
                        latents / self.base.vae.config.scaling_factor, return_dict=False
                    )[0]
                # cast back to fp16 if needed
                if self.base.needs_upcasting:
                    self.base.vae.to(dtype=torch.float16)

            image = self.base.image_processor.postprocess(image, output_type=output_type)
            if show_image:
                plt.figure(figsize=(10, 10))
                plt.imshow(image[0])
                plt.axis("off")  # Turn off axis numbers and ticks
                plt.show()
            output_images.append(image[0])

        ####################################################### Phase 2+ #####################################################
        for current_scale_num in range(1, scale_num + 1):
            if self.lowvram:
                latents = latents.to(self.base.device)
                torch.cuda.empty_cache()
            print("### Phase {} Denoising ###".format(current_scale_num))
            current_height = (
                128 * self.base.vae_scale_factor * current_scale_num
            )
            current_width = (
                128 * self.base.vae_scale_factor * current_scale_num
            )
            if height > width:
                current_width = int(current_width * aspect_ratio)
            else:
                current_height = int(current_height * aspect_ratio)

            latents = F.interpolate(
                latents.to(self.base.device),
                size=(
                    int(current_height / self.base.vae_scale_factor),
                    int(current_width / self.base.vae_scale_factor),
                ),
                mode="bicubic",
            )

            noise_latents = []
            noise = torch.randn_like(latents)
            for timestep in timesteps:
                noise_latent = self.base.scheduler.add_noise(
                    latents, noise, timestep.unsqueeze(0)
                )
                noise_latents.append(noise_latent)
            latents = noise_latents[0]

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    count = torch.zeros_like(latents)
                    value = torch.zeros_like(latents)
                    cosine_factor = (
                        0.5
                        * (
                            1
                            + torch.cos(
                                torch.pi
                                * (self.base.scheduler.config.num_train_timesteps - t)
                                / self.base.scheduler.config.num_train_timesteps
                            )
                        ).cpu()
                    )

                    c1 = cosine_factor**cosine_scale_1
                    latents = latents * (1 - c1) + noise_latents[i] * c1

                    ############################################# MultiDiffusion #############################################

                    views = self.get_views(
                        current_height,
                        current_width,
                        stride=stride,
                        window_size=128,
                        random_jitter=True,
                    )
                    views_batch = [
                        views[i : i + view_batch_size]
                        for i in range(0, len(views), view_batch_size)
                    ]

                    jitter_range = (128 - stride) // 4
                    latents_ = F.pad(
                        latents,
                        (jitter_range, jitter_range, jitter_range, jitter_range),
                        "constant",
                        0,
                    )

                    count_local = torch.zeros_like(latents_)
                    value_local = torch.zeros_like(latents_)

                    for j, batch_view in enumerate(views_batch):
                        vb_size = len(batch_view)

                        self.base.loadResources(x1_size, x1_size, batch_size*vb_size, do_cfg)

                        # get the latents corresponding to the current view coordinates
                        latents_for_view = torch.cat(
                            [
                                latents_[:, :, h_start:h_end, w_start:w_end]
                                for h_start, h_end, w_start, w_end in batch_view
                            ]
                        )

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = latents_for_view
                        latent_model_input = (
                            latent_model_input.repeat_interleave(2, dim=0)
                            if do_cfg
                            else latent_model_input
                        )
                        latent_model_input = self.base.scheduler.scale_model_input(
                            latent_model_input, t
                        )

                        prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                        add_text_embeds_input = torch.cat([pooled_prompt_embeds] * vb_size)
                        add_time_ids_input = []
                        for h_start, h_end, w_start, w_end in batch_view:
                            add_time_ids_ = add_time_ids.clone()
                            add_time_ids_[:, 2] = h_start * self.base.vae_scale_factor
                            add_time_ids_[:, 3] = w_start * self.base.vae_scale_factor
                            add_time_ids_input.append(add_time_ids_)
                        add_time_ids_input = torch.cat(add_time_ids_input)

                        # predict the noise residual
                        params = {"sample": latent_model_input, "timestep": t.reshape(-1).half(), "encoder_hidden_states": prompt_embeds_input, "add_text_embeds": add_text_embeds_input, "add_time_ids": add_time_ids_input}
                        noise_pred = self.base.unet_engine.infer(params, self.base.stream, use_cuda_graph=self.base.use_cuda_graph)['out_sample']
                        

                        if do_cfg:
                            noise_pred_uncond, noise_pred_text = (
                                noise_pred[::2],
                                noise_pred[1::2],
                            )
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )

                        # compute the previous noisy sample x_t -> x_t-1
                        self.base.scheduler._init_step_index(t)
                        latents_denoised_batch = self.base.scheduler.step(
                            noise_pred,
                            t,
                            latents_for_view,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                        # extract value from batch
                        for latents_view_denoised, (
                            h_start,
                            h_end,
                            w_start,
                            w_end,
                        ) in zip(latents_denoised_batch.chunk(vb_size), batch_view):
                            value_local[
                                :, :, h_start:h_end, w_start:w_end
                            ] += latents_view_denoised
                            count_local[:, :, h_start:h_end, w_start:w_end] += 1

                    value_local = value_local[
                        :,
                        :,
                        jitter_range : jitter_range
                        + current_height // self.base.vae_scale_factor,
                        jitter_range : jitter_range
                        + current_width // self.base.vae_scale_factor,
                    ]
                    count_local = count_local[
                        :,
                        :,
                        jitter_range : jitter_range
                        + current_height // self.base.vae_scale_factor,
                        jitter_range : jitter_range
                        + current_width // self.base.vae_scale_factor,
                    ]

                    c2 = cosine_factor**cosine_scale_2

                    value += value_local / count_local * (1 - c2)
                    count += torch.ones_like(value_local) * (1 - c2)

                    ############################################# Dilated Sampling #############################################

                    views = [
                        [h, w]
                        for h in range(current_scale_num)
                        for w in range(current_scale_num)
                    ]
                    views_batch = [
                        views[i : i + view_batch_size]
                        for i in range(0, len(views), view_batch_size)
                    ]

                    h_pad = (
                        current_scale_num - (latents.size(2) % current_scale_num)
                    ) % current_scale_num
                    w_pad = (
                        current_scale_num - (latents.size(3) % current_scale_num)
                    ) % current_scale_num
                    latents_ = F.pad(latents, (w_pad, 0, h_pad, 0), "constant", 0)

                    count_global = torch.zeros_like(latents_)
                    value_global = torch.zeros_like(latents_)

                    c3 = 0.99 * cosine_factor**cosine_scale_3 + 1e-2
                    std_, mean_ = latents_.std(), latents_.mean()
                    latents_gaussian = gaussian_filter(
                        latents_,
                        kernel_size=(2 * current_scale_num - 1),
                        sigma=sigma * c3,
                    )
                    latents_gaussian = (
                        latents_gaussian - latents_gaussian.mean()
                    ) / latents_gaussian.std() * std_ + mean_

                    for j, batch_view in enumerate(views_batch):
                        latents_for_view = torch.cat(
                            [
                                latents_[
                                    :, :, h::current_scale_num, w::current_scale_num
                                ]
                                for h, w in batch_view
                            ]
                        )
                        latents_for_view_gaussian = torch.cat(
                            [
                                latents_gaussian[
                                    :, :, h::current_scale_num, w::current_scale_num
                                ]
                                for h, w in batch_view
                            ]
                        )

                        vb_size = latents_for_view.size(0)

                        self.base.loadResources(x1_size, x1_size, batch_size*vb_size, do_cfg)

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = latents_for_view_gaussian
                        latent_model_input = (
                            latent_model_input.repeat_interleave(2, dim=0)
                            if do_cfg
                            else latent_model_input
                        )
                        latent_model_input = self.base.scheduler.scale_model_input(
                            latent_model_input, t
                        )

                        prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)
                        add_text_embeds_input = torch.cat([pooled_prompt_embeds] * vb_size)
                        add_time_ids_input = torch.cat([add_time_ids] * vb_size)


                        # predict the noise residual
                        params = {"sample": latent_model_input, "timestep": t.reshape(-1).half(), "encoder_hidden_states": prompt_embeds_input, 'add_text_embeds': add_text_embeds_input, 'add_time_ids': add_time_ids_input}
                        noise_pred = self.base.unet_engine.infer(params, self.base.stream, use_cuda_graph=self.base.use_cuda_graph)['out_sample']

                        if do_cfg:
                            noise_pred_uncond, noise_pred_text = (
                                noise_pred[::2],
                                noise_pred[1::2],
                            )
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )

                        # compute the previous noisy sample x_t -> x_t-1
                        self.base.scheduler._init_step_index(t)
                        latents_denoised_batch = self.base.scheduler.step(
                            noise_pred,
                            t,
                            latents_for_view,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                        # extract value from batch
                        for latents_view_denoised, (h, w) in zip(
                            latents_denoised_batch.chunk(vb_size), batch_view
                        ):
                            value_global[
                                :, :, h::current_scale_num, w::current_scale_num
                            ] += latents_view_denoised
                            count_global[
                                :, :, h::current_scale_num, w::current_scale_num
                            ] += 1

                    c2 = cosine_factor**cosine_scale_2

                    value_global = value_global[:, :, h_pad:, w_pad:]

                    value += value_global * c2
                    count += torch.ones_like(value_global) * c2

                    ###########################################################

                    latents = torch.where(count > 0, value / count, value)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) % self.base.scheduler.order == 0):
                        progress_bar.update()

                #########################################################################################################################################

                latents = (
                    latents - latents.mean()
                ) / latents.std() * anchor_std + anchor_mean
                if self.lowvram:
                    latents = latents.cpu()
                    torch.cuda.empty_cache()
                if not output_type == "latent":
                    # make sure the VAE is in float32 mode, as it overflows in float16

                    if self.lowvram:
                        self.base.vae.to(self.base.device)

                    if self.base.needs_upcasting:
                        self.base.upcast_vae()
                        latents = latents.to(
                            next(iter(self.base.vae.post_quant_conv.parameters())).dtype
                        )

                    print("### Phase {} Decoding ###".format(current_scale_num))
                    if multi_decoder:
                        image = self.tiled_decode(
                            latents, current_height, current_width
                        )
                    else:
                        image = self.base.vae.decode(
                            latents / self.base.vae.config.scaling_factor, return_dict=False
                        )[0]

                    # cast back to fp16 if needed
                    if self.base.needs_upcasting:
                        self.base.vae.to(dtype=torch.float16)
                else:
                    image = latents

                if not output_type == "latent":
                    image = self.base.image_processor.postprocess(
                        image, output_type=output_type
                    )
                    if show_image:
                        plt.figure(figsize=(10, 10))
                        plt.imshow(image[0])
                        plt.axis("off")  # Turn off axis numbers and ticks
                        plt.show()
                    output_images.append(image[0])

        cudart.cudaEventRecord(self.base.events['denoise-stop'], 0)
        return output_images
    
    def preprocess_imglr(self, pil_image):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((1024, 1024)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        image = transform(pil_image)
        image = image.unsqueeze(0).half()
        return image
