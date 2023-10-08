from cfdraw import *
import torch
from .load import *
from .prepocess import *

def get_generator(seed):
    if seed != -1:
        generator=torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator=torch.Generator(device="cuda")
        generator.seed()
    return generator


def txt2img(pipe, data, step_callback):
    generator = get_generator(data.extraData["seed"])
    orig_sampler = pipe.scheduler
    pipe = alter_sampler(pipe, data.extraData["sampler"])

    images = pipe(prompt=data.extraData["text"],
                  negative_prompt=data.extraData["negative_prompt"],
                  height=data.extraData["h"],
                  width=data.extraData["w"],
                  num_inference_steps=data.extraData["num_steps"],
                  guidance_scale=data.extraData["guidance_scale"], 
                  generator=generator,
                  num_images_per_prompt=data.extraData["num_samples"],
                  callback=step_callback).images
    
    pipe.scheduler = orig_sampler
    torch.cuda.empty_cache()
    return images

def img2img(pipe, img, data, step_callback):
    generator = get_generator(data.extraData["seed"])
    orig_sampler = pipe.scheduler
    pipe = alter_sampler(pipe, data.extraData["sampler"])

    images = pipe(prompt=data.extraData["text"], 
                  image=img, 
                  negative_prompt=data.extraData["negative_prompt"],
                  num_inference_steps=data.extraData["num_steps"],
                  guidance_scale=data.extraData["guidance_scale"], 
                  strength=data.extraData["strength"],
                  generator=generator,
                  num_images_per_prompt=data.extraData["num_samples"],
                  callback=step_callback).images
    
    pipe.scheduler = orig_sampler
    torch.cuda.empty_cache()
    return images

def inpaint(pipe, img, mask, data, step_callback):
    img = img.convert("RGB")
    mask = png_to_mask(mask)
    generator = get_generator(data.extraData["seed"])

    if data.extraData["focus_mode"]:
        img_c, mask_c, box = crop_masked_area(img, mask)
    else:
        img_c, mask_c = img, mask

    orig_sampler = pipe.scheduler
    pipe = alter_sampler(pipe, data.extraData["sampler"])

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
    
    pipe.scheduler = orig_sampler
    torch.cuda.empty_cache()
    return images

def cn_inpaint(pipe, img, mask, data, step_callback):
    img = img.convert("RGB")
    mask = png_to_mask(mask)
    generator = get_generator(data.extraData["seed"])

    if data.extraData["focus_mode"]:
        img_c, mask_c, box = crop_masked_area(img, mask)
    else:
        img_c, mask_c = img, mask

    orig_sampler = pipe.scheduler
    pipe = alter_sampler(pipe, data.extraData["sampler"])

    masked_img = make_inpaint_condition(img, mask)
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

    pipe.scheduler = orig_sampler
    torch.cuda.empty_cache()
    return images

def cn_tile(pipe, img, data, step_callback):
    generator = get_generator(data.extraData["seed"])

    orig_sampler = pipe.scheduler
    pipe = alter_sampler(pipe, data.extraData["sampler"])

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
    
    pipe.scheduler = orig_sampler
    torch.cuda.empty_cache()
    return images


def style_transfer(model, img, data, step_callback):
    if data.extraData["seed"] != -1:
        generator=torch.Generator(device="cuda").manual_seed(data.extraData["seed"])
    else:
        generator=torch.Generator(device="cuda")
        generator.seed()
    orig_sampler = model.pipe.scheduler
    model.pipe = alter_sampler(model.pipe, data.extraData["sampler"])

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
    
    model.pipe.scheduler = orig_sampler
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
    
    new = Image.new("RGBA", [int(max(data1.w, data0.x-data1.x+data0.w)), int(max(data1.h, data0.y-data1.y+data0.h))], color=(0,0,0,0))
    new.paste(img0, (int(data0.x-data1.x), int(data0.y-data1.y)), mask=img0.getchannel("A"))
    new = new.rotate(np.degrees(theta0), center=[int(data0.x-data1.x), int(data0.y-data1.y)])

    img1.paste(new, (0,0), mask=new.getchannel("A"))

    mask = np.array(new.getchannel("A"))
    mask_edge = ExpandEdge(mask, 10)
    mask_edge = Image.fromarray(mask_edge[:int(data1.h),:int(data1.w)]).convert("L")
    return [img1, mask_edge]



def easy_inpaint(pipe, img, mask, data, step_callback):
    img = img.convert("RGB")
    generator = get_generator(data.extraData["seed"])

    if data.extraData["focus_mode"]:
        img_c, mask_c, box = crop_masked_area(img, mask)
    else:
        img_c, mask_c = img, mask

    orig_sampler = pipe.scheduler
    pipe = alter_sampler(pipe, data.extraData["sampler"])

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
    
    pipe.scheduler = orig_sampler
    torch.cuda.empty_cache()
    return images