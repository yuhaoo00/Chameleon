from cfdraw import *
import torch
from .load import *
from .prepocess import *
from extensions.annotators.hed import HEDdetector
from extensions.annotators.zoe import ZoeDetector
from extensions.annotators.canny import CannyDetector

def get_generator(seed):
    if seed != -1:
        generator=torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator=torch.Generator(device="cuda")
        generator.seed()
    return generator


def txt2img(pipe, data, hrfix_steps_ratio, step_callback, step_callback2):
    pipe = alter_sampler(pipe, data.extraData["sampler"])
    generator = get_generator(data.extraData["seed"])
    #orig_sampler = pipe.scheduler
    #pipe.scheduler = orig_sampler

    if not data.extraData["use_highres"]:
        images = pipe(
                    prompt=data.extraData["text"],
                    negative_prompt=data.extraData["negative_prompt"],
                    height=data.extraData["h"],
                    width=data.extraData["w"],
                    num_inference_steps=data.extraData["num_steps"],
                    guidance_scale=data.extraData["guidance_scale"], 
                    generator=generator,
                    num_images_per_prompt=data.extraData["num_samples"],
                    callback=step_callback).images
    else:
        lr_latents = pipe(
                    prompt=data.extraData["text"],
                    negative_prompt=data.extraData["negative_prompt"],
                    height=data.extraData["h"],
                    width=data.extraData["w"],
                    num_inference_steps=data.extraData["num_steps"],
                    guidance_scale=data.extraData["guidance_scale"], 
                    generator=generator,
                    num_images_per_prompt=data.extraData["num_samples"],
                    callback=step_callback,
                    output_type="latent").images
        i2i = AutoPipelineForImage2Image.from_pipe(pipe)

        hr_latents = F.interpolate(lr_latents, scale_factor=data.extraData["highres_scale"], mode="nearest")
        images = i2i(
                    prompt=data.extraData["text"], 
                    image=hr_latents, 
                    negative_prompt=data.extraData["negative_prompt"],
                    num_inference_steps=int(data.extraData["num_steps"]*hrfix_steps_ratio),
                    guidance_scale=data.extraData["guidance_scale"], 
                    strength=data.extraData["highres_strength"],
                    generator=generator,
                    num_images_per_prompt=data.extraData["num_samples"],
                    callback=step_callback2).images
        
    torch.cuda.empty_cache()
    return images

def img2img(pipe, img, data, hrfix_steps_ratio, step_callback, step_callback2):
    pipe = alter_sampler(pipe, data.extraData["sampler"])
    generator = get_generator(data.extraData["seed"])

    if not data.extraData["use_highres"]:
        lr_latents = pipe(
                    prompt=data.extraData["text"], 
                    image=img, 
                    negative_prompt=data.extraData["negative_prompt"],
                    num_inference_steps=data.extraData["num_steps"],
                    guidance_scale=data.extraData["guidance_scale"], 
                    strength=data.extraData["strength"],
                    generator=generator,
                    num_images_per_prompt=data.extraData["num_samples"],
                    callback=step_callback).images
    else:
        lr_latents = pipe(
                    prompt=data.extraData["text"], 
                    image=img, 
                    negative_prompt=data.extraData["negative_prompt"],
                    num_inference_steps=data.extraData["num_steps"],
                    guidance_scale=data.extraData["guidance_scale"], 
                    strength=data.extraData["strength"],
                    generator=generator,
                    num_images_per_prompt=data.extraData["num_samples"],
                    callback=step_callback,
                    output_type="latent").images
        
        hr_latents = F.interpolate(lr_latents, scale_factor=data.extraData["highres_scale"], mode="nearest")
        images = pipe(
                    prompt=data.extraData["text"], 
                    image=hr_latents, 
                    negative_prompt=data.extraData["negative_prompt"],
                    num_inference_steps=int(data.extraData["num_steps"]*hrfix_steps_ratio),
                    guidance_scale=data.extraData["guidance_scale"], 
                    strength=data.extraData["highres_strength"],
                    generator=generator,
                    num_images_per_prompt=data.extraData["num_samples"],
                    callback=step_callback2).images
    
    torch.cuda.empty_cache()
    return images

def inpaint(pipe, img, mask, data, step_callback):
    pipe = alter_sampler(pipe, data.extraData["sampler"])
    generator = get_generator(data.extraData["seed"])
    img = img.convert("RGB")
    mask = png_to_mask(mask)

    if data.extraData["focus_mode"]:
        img_c, mask_c, box = crop_masked_area(img, mask, data.extraData["w"], data.extraData["h"])
    else:
        img_c, mask_c = img, mask

    images = pipe(prompt=data.extraData["text"],
                  image=img_c,
                  mask_image=mask_c,
                  height=data.extraData["h"],
                  width=data.extraData["w"],
                  negative_prompt=data.extraData["negative_prompt"],
                  num_inference_steps=data.extraData["num_steps"],
                  guidance_scale=data.extraData["guidance_scale"], 
                  generator=generator,
                  strength=data.extraData["strength"],
                  num_images_per_prompt=data.extraData["num_samples"],
                  callback=step_callback).images
    
    if data.extraData["focus_mode"]:
        images = recover_cropped_image(images, img, box)

    torch.cuda.empty_cache()
    return images

def cn_inpaint(pipe, img, mask, data, step_callback):
    pipe = alter_sampler(pipe, data.extraData["sampler"])
    generator = get_generator(data.extraData["seed"])
    img = img.convert("RGB")
    mask = png_to_mask(mask)

    if data.extraData["focus_mode"]:
        img_c, mask_c, box = crop_masked_area(img, mask, data.extraData["w"], data.extraData["h"])
    else:
        img_c, mask_c = img, mask


    masked_img = make_inpaint_condition(img_c, mask_c)
    images = pipe(prompt=data.extraData["text"],
                  image=img_c,
                  mask_image=mask_c,
                  control_image=masked_img,
                  height=data.extraData["h"],
                  width=data.extraData["w"],
                  negative_prompt=data.extraData["negative_prompt"],
                  num_inference_steps=data.extraData["num_steps"],
                  guidance_scale=data.extraData["guidance_scale"], 
                  generator=generator,
                  strength=data.extraData["strength"],
                  num_images_per_prompt=data.extraData["num_samples"],
                  controlnet_conditioning_scale= data.extraData["controlnet_conditioning_scale"],
                  callback=step_callback).images
    
    if data.extraData["focus_mode"]:
        images = recover_cropped_image(images, img, box)

    torch.cuda.empty_cache()
    return images

def cn_tile(pipe, img, data, step_callback):
    pipe = alter_sampler(pipe, data.extraData["sampler"])
    generator = get_generator(data.extraData["seed"])

    images = pipe(prompt=data.extraData["text"],
                  image=img,
                  height=data.extraData["h"],
                  width=data.extraData["w"],
                  negative_prompt=data.extraData["negative_prompt"],
                  num_inference_steps=data.extraData["num_steps"],
                  guidance_scale=data.extraData["guidance_scale"], 
                  generator=generator,
                  num_images_per_prompt=data.extraData["num_samples"],
                  controlnet_conditioning_scale= data.extraData["controlnet_conditioning_scale"],
                  callback=step_callback).images
    
    torch.cuda.empty_cache()
    return images


def style_transfer(model, img, data, step_callback):
    model.pipe = alter_sampler(model.pipe, data.extraData["sampler"])
    generator = get_generator(data.extraData["seed"])

    images = model.generate(pil_image=img,
                          prompt=data.extraData["text"],
                          negative_prompt=data.extraData["negative_prompt"],
                          height=data.extraData["h"],
                          width=data.extraData["w"],
                          num_inference_steps=data.extraData["num_steps"],
                          generator=generator,
                          guidance_scale=data.extraData["guidance_scale"],
                          scale=data.extraData["scale"],
                          num_samples=data.extraData["num_samples"],
                          callback=step_callback)
    
    torch.cuda.empty_cache()
    return images

def easy_fusing(data0, data1, img0, img1):
    img0 = img0.convert("RGBA")
    img1 = img1.convert("RGBA")
    if data0.z > data1.z:
        data0, data1 = data1, data0
        img0, img1 = img1, img0

    theta0 = get_angle(data0.transform.a, data0.transform.c, data0.w, data0.h)
    theta1 = get_angle(data1.transform.a, data1.transform.c, data1.w, data1.h)

    img0 = img0.resize(size=[int(data0.w), int(data0.h)])
    img1 = img1.resize(size=[int(data1.w), int(data1.h)]).rotate(np.degrees(theta1), expand=True)
    fused_img = img1.copy()
    
    new = Image.new("RGBA", [int(max(data1.w, data0.x-data1.x+data0.w)), int(max(data1.h, data0.y-data1.y+data0.h))], color=(0,0,0,0))
    new.paste(img0, (int(data0.x-data1.x), int(data0.y-data1.y)), mask=img0.getchannel("A"))
    new = new.rotate(np.degrees(theta0), center=[int(data0.x-data1.x), int(data0.y-data1.y)])
    img0 = new.crop((0,0,data1.w,data1.h))

    mask = img0.getchannel("A").convert("L")

    fused_img.paste(img0, (0,0), mask=mask)

    return [fused_img, img1, img0, mask]


def edge_fusing(pipe, data, data0, data1, img0, img1, step_callback):
    pipe = alter_sampler(pipe, data.extraData["sampler"])
    generator = get_generator(data.extraData["seed"])

    img, _, _, mask = easy_fusing(data0, data1, img0, img1)
    mask = ExpandEdge(mask, 10)
    img = img.convert("RGB")

    if data.extraData["focus_mode"]:
        img_c, mask_c, box = crop_masked_area(img, mask, data.extraData["w"], data.extraData["h"], 0.1)
    else:
        img_c, mask_c = img, mask

    images = pipe(prompt=data.extraData["text"],
                  image=img_c,
                  mask_image=mask_c,
                  height=data.extraData["h"],
                  width=data.extraData["w"],
                  negative_prompt=data.extraData["negative_prompt"],
                  num_inference_steps=data.extraData["num_steps"],
                  guidance_scale=data.extraData["guidance_scale"], 
                  generator=generator,
                  strength=data.extraData["strength"],
                  num_images_per_prompt=data.extraData["num_samples"],
                  callback=step_callback).images
    
    if data.extraData["focus_mode"]:
        images = recover_cropped_image(images, img, box)

    torch.cuda.empty_cache()
    return images

def smart_fusing(pipe, data, data0, data1, img0, img1, step_callback):
    pipe = alter_sampler(pipe, data.extraData["sampler"])
    generator = get_generator(data.extraData["seed"])

    orig_fused_img, back_img, fore_img, mask = easy_fusing(data0, data1, img0, img1)
    init_imgs, init_mask, box = crop_masked_area([back_img, fore_img], mask, data.extraData["w"], data.extraData["h"], data.extraData["box_padding"])
    
    init_back_img = init_imgs[0]
    init_fore_img = init_imgs[1]
    init_mask = ExpandMask(init_mask, data.extraData["mask_expand"])

    ann = CannyDetector()
    init_img_prompt = ann(init_fore_img, 100, 200)
    
    fused_imgs = pipe(prompt=data.extraData["text"],
                        image=init_back_img,
                        mask_image=init_mask,
                        control_image=init_img_prompt,
                        height=data.extraData["h"],
                        width=data.extraData["w"],
                        negative_prompt=data.extraData["negative_prompt"],
                        num_inference_steps=data.extraData["num_steps"],
                        guidance_scale=data.extraData["guidance_scale"], 
                        generator=generator,
                        strength=data.extraData["strength"],
                        num_images_per_prompt=data.extraData["num_samples"],
                        callback=step_callback).images

    images = recover_cropped_image(fused_imgs, orig_fused_img, box)
    torch.cuda.empty_cache()
    return images