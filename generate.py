import torch
import numpy as np
from PIL import Image
from pipelines.trt_sdxl_t2i import SDXL_T2I_Pipeline
from pipelines.trt_sdxl_i2i import SDXL_I2I_Pipeline
from pipelines.trt_sdxl_inpaint import SDXL_Inpaint_Pipeline
from utils import torch_gc, str2img, img2str, crop_masked_area, recover_cropped_image

main_dir = "/work/CKPTS/"

upscaler_paths = {
    "ESRGAN-general": main_dir+"Real-ESRGAN/RealESRGAN_x4plus.pth",
    "ESRGAN-anime": main_dir+"Real-ESRGAN/RealESRGAN_x4plus_anime_6B.pth",
}

def get_generator(seed):
    if seed != -1:
        generator=torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator=torch.Generator(device="cuda")
        generator.seed()
    return generator


def txt2img(base, data):
    pipe = SDXL_T2I_Pipeline(base)
    generator = get_generator(data.seed)

    if not data.use_hrfix:
        images = pipe.infer(
                    prompt=data.text,
                    negative_prompt=data.negative_prompt,
                    height=data.h,
                    width=data.w,
                    num_inference_steps=data.num_steps,
                    guidance_scale=data.guidance_scale, 
                    generator=generator,
                    num_images_per_prompt=data.num_samples)
    else:
        lr_latents = pipe.infer(
                    prompt=data.text,
                    negative_prompt=data.negative_prompt,
                    height=data.h,
                    width=data.w,
                    num_inference_steps=data.num_steps,
                    guidance_scale=data.guidance_scale, 
                    generator=generator,
                    num_images_per_prompt=data.num_samples,
                    output_type="latent")
        i2i = SDXL_T2I_Pipeline(base)

        hr_latents = torch.functional.interpolate(lr_latents, scale_factor=data.hrfix_scale, mode="nearest")

        images = i2i.infer(
                    prompt=data.text,
                    negative_prompt=data.negative_prompt,
                    image=hr_latents,
                    num_inference_steps=int(data.num_steps*data.hrfix_steps_ratio),
                    guidance_scale=data.guidance_scale, 
                    strength=data.hrfix_strength,
                    generator=generator,
                    num_images_per_prompt=data.num_samples)
    
    images_str = img2str(images)
    return images_str


def img2img(base, data):
    pipe = SDXL_I2I_Pipeline(base)
    generator = get_generator(data.seed)

    image_prompt = str2img(data.image)[0]

    if not data.use_hrfix:
        images = pipe.infer(
                        prompt=data.text,
                        negative_prompt=data.negative_prompt,
                        image=image_prompt,
                        strength=data.strength,
                        num_inference_steps=data.num_steps,
                        guidance_scale=data.guidance_scale, 
                        generator=generator,
                        num_images_per_prompt=data.num_samples)
    else:
        lr_latents = pipe.infer(
                    prompt=data.text,
                    negative_prompt=data.negative_prompt,
                    image=image_prompt,
                    strength=data.strength,
                    num_inference_steps=data.num_steps,
                    guidance_scale=data.guidance_scale, 
                    generator=generator,
                    num_images_per_prompt=data.num_samples,
                    output_type="latent")

        hr_latents = torch.functional.interpolate(lr_latents, scale_factor=data.hrfix_scale, mode="nearest")

        images = pipe.infer(
                    prompt=data.text,
                    negative_prompt=data.negative_prompt,
                    image=hr_latents,
                    num_inference_steps=int(data.num_steps*data.hrfix_steps_ratio),
                    guidance_scale=data.guidance_scale, 
                    strength=data.hrfix_strength,
                    generator=generator,
                    num_images_per_prompt=data.num_samples)

    images_str = img2str(images)
    return images_str


def inpaint(base, data):
    pipe = SDXL_Inpaint_Pipeline(base)
    generator = get_generator(data.seed)

    img = str2img(data.image)[0]
    mask = str2img(data.mask)[0].convert("L")
    
    if data.focus_mode:
        img_c, mask_c, box = crop_masked_area(img, mask, data.w, data.h)
    else:
        img_c, mask_c = img, mask

    images = pipe.infer(
                prompt=data.text,
                negative_prompt=data.negative_prompt,
                image=img_c,
                mask_image=mask_c,
                height=data.h,
                width=data.w,
                strength=data.strength,
                num_inference_steps=data.num_steps,
                guidance_scale=data.guidance_scale, 
                generator=generator,
                num_images_per_prompt=data.num_samples)
    
    if data.focus_mode:
        images = recover_cropped_image(images, img, box)

    images_str = img2str(images)
    return images_str





def upscale(data):
    from extensions.realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    
    image = str2img(data.image)[0]
    img = np.array(image)

    if data.tag == "ESRGAN-general":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    elif data.tag == "ESRGAN-anime":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)

    model = RealESRGANer(
        scale=4,
        model_path=upscaler_paths[data.tag],
        dni_weight=None,
        model=model,
        half=True)

    output, _ = model.enhance(img, outscale=data.scale)

    image_str = img2str(Image.fromarray(output))
    del model
    return image_str