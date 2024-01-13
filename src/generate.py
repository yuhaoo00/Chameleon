import torch
import numpy as np
import math
from PIL import Image
from .pipelines import SDXL_T2I_Pipeline, SDXL_I2I_Pipeline, SDXL_Inpaint_Pipeline, SDXL_DemoFusion, SDXL_T2I_CN_Pipeline
from .utils import torch_gc, url2img, str2img, img2str, crop_masked_area, recover_cropped_image, mask_to_box, get_angle, rotate_xy

main_dir = "/work/CKPTS/"

upscaler_paths = {
    "ESRGAN-general": main_dir+"Real-ESRGAN/RealESRGAN_x4plus.pth",
    "ESRGAN-anime": main_dir+"Real-ESRGAN/RealESRGAN_x4plus_anime_6B.pth",
}

annotator_paths = {
    "hed": main_dir+"lllyasviel--Annotators/ControlNetHED.pth",
    "zoe": main_dir+"lllyasviel--Annotators/ZoeD_M12_N.pt",
    "depth": main_dir+"lllyasviel--Annotators/dpt_hybrid-midas-501f0c75.pt",
}

sam_paths = {
    "mobile_sam": main_dir+"mobile_sam.pt",
}

def get_generator(seed):
    if seed != -1:
        generator=torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator=torch.Generator(device="cuda")
        generator.seed()
    return generator


def sd_t2i(base, data):
    generator = get_generator(data.seed)

    if not data.control:
        pipe = SDXL_T2I_Pipeline(base)

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
    else:
        if "canny" in data.control[0]:
            pipe = SDXL_T2I_CN_Pipeline(base, "control_canny")
        elif "zoe" in data.control[0]:
            pipe = SDXL_T2I_CN_Pipeline(base, "control_zoe")
        elif "depth" in data.control[0]:
            pipe = SDXL_T2I_CN_Pipeline(base, "control_depth")

        image_hint = url2img(data.control_hint[0])
        print(np.array(image_hint).shape)
        images = pipe.infer(
                    prompt=data.text,
                    negative_prompt=data.negative_prompt,
                    height=data.h,
                    width=data.w,
                    num_inference_steps=data.num_steps,
                    guidance_scale=data.guidance_scale, 
                    generator=generator,
                    num_images_per_prompt=data.num_samples,
                    image=image_hint,
                    controlnet_conditioning_scale=data.control_strength[0],
                    control_guidance_start=data.control_hint_start[0],
                    control_guidance_end=data.control_hint_end[0],
                    )
        pipe.unload()
    
    images_str = img2str(images)
    return images_str


def sd_i2i(base, data):
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


def sd_inpaint(base, data):
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


def sd_demofusion(base, data):
    pipe = SDXL_DemoFusion(base)
    generator = get_generator(data.seed)

    image_prompt = str2img(data.image)[0]

    images = pipe.infer(
                    prompt=data.text,
                    negative_prompt=data.negative_prompt,
                    image_lr=image_prompt,
                    height=data.h,
                    width=data.w,
                    num_inference_steps=data.num_steps,
                    guidance_scale=data.guidance_scale, 
                    generator=generator,
                    view_batch_size=data.view_batch_size,
                    stride=data.stride,
                    multi_decoder=data.multi_decoder,
                    cosine_scale_1=data.cosine_scale_1,
                    cosine_scale_2=data.cosine_scale_2,
                    cosine_scale_3=data.cosine_scale_3,
                    sigma=data.sigma)
    
    images_str = img2str(images)
    return images_str


def nn_upscale(data):
    from .extensions.realesrgan import RealESRGANer
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


def annotating(data):
    from .extensions.annotators.canny import CannyDetector
    from .extensions.annotators.hed import HEDdetector
    from .extensions.annotators.depth import MidasDetector

    if data.type == "canny":
        anno = CannyDetector()
    elif data.type == "hed":
        anno = HEDdetector(annotator_paths[data.type])
    elif data.type == "depth":
        anno = MidasDetector(annotator_paths[data.type])
    else:
        raise ValueError(f"Undefined Annotator: {data.type}")

    image = str2img(data.image)[0]
    image = anno(image, data.low_threshold, data.high_threshold)
    image_str = img2str(image)
    return image_str


def sam_matting(data):
    from .extensions.MobileSAM import sam_model_registry, SamPredictor
    img = str2img(data.image)[0]
    mask = str2img(data.mask)[0].convert("L")
    box = mask_to_box(mask)

    mobile_sam = sam_model_registry["vit_t"](checkpoint=sam_paths["mobile_sam"]).cuda()
    mobile_sam.eval()
    model = SamPredictor(mobile_sam)

    model.set_image(np.array(img))
    mask_fined = model.predict(box=box, multimask_output=False)[0][0,:]
    img_masked = np.concatenate((np.array(img), (mask_fined[:,:,None]*255).astype(np.uint8)), axis=2)
    img_masked = Image.fromarray(img_masked)
    
    image_str = img2str(img_masked)
    return image_str


def easy_fusing(data):
    info0 = data.info0
    info1 = data.info1

    img0 = url2img(info0['img_url']).convert("RGBA")
    img1 = url2img(info1['img_url']).convert("RGBA")

    if info0['z'] > info1['z']:
        info0, info1 = info1, info0
        img0, img1 = img1, img0

    theta0 = get_angle(info0['transform'][0], info0['transform'][2], info0['w'], info0['h'])
    theta1 = get_angle(info1['transform'][0], info1['transform'][2], info1['w'], info1['h'])

    img0 = img0.resize(size=[int(info0['w']), int(info0['h'])]).rotate(np.degrees(theta0), expand=True)
    img1 = img1.resize(size=[int(info1['w']), int(info1['h'])]).rotate(np.degrees(theta1), expand=True)
    
    max_width0 = math.sqrt(info0['w']**2+info0['h']**2)
    max_width1 = math.sqrt(info1['w']**2+info1['h']**2)
    max_width = int(max_width0+max_width1)
    new = Image.new("RGBA", [max_width, max_width], color=(0,0,0,0))

    offset_x0, offset_y0 = rotate_xy(-info0['w']//2,-info0['h']//2,0,0,theta0) 
    offset_x1, offset_y1 = rotate_xy(-info1['w']//2,-info1['h']//2,0,0,theta1) 

    xx = min(info1['x'], info0['x'])
    yy = min(info1['y'], info0['y'])
    info0['x'] -= xx
    info1['x'] -= xx
    info0['y'] -= yy
    info1['y'] -= yy
    offset_x0, offset_y0 = rotate_xy(-info0['w']//2,-info0['h']//2,0,0,theta0) 
    offset_x1, offset_y1 = rotate_xy(-info1['w']//2,-info1['h']//2,0,0,theta1) 
    offset_x0 = int(offset_x0+img0.size[0]/2)
    offset_y0 = int(img0.size[1]/2-offset_y0)
    offset_x1 = int(offset_x1+img1.size[0]/2)
    offset_y1 = int(img1.size[1]/2-offset_y1)

    new.paste(img1, (info1['x']-offset_x1, info1['y']-offset_y1), mask=img1.getchannel("A"))
    new.paste(img0, (info0['x']-offset_x0, info0['y']-offset_y0), mask=img0.getchannel("A"))

    image_str = img2str(new)
    return image_str


