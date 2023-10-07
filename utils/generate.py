from cfdraw import *
import torch
from .load import *

def get_generation(seed):
    if seed != -1:
        generator=torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator=torch.Generator(device="cuda")
        generator.seed()
    return generator


def txt2img(pipe, data, step_callback):
    generator = get_generation(data.extraData["seed"])
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
    generator = get_generation(data.extraData["seed"])
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
    generator = get_generation(data.extraData["seed"])
    orig_sampler = pipe.scheduler
    pipe = alter_sampler(pipe, data.extraData["sampler"])

    images = pipe(prompt=data.extraData["text"],
                  image=img,
                  mask_image=mask,
                  height=data.extraData["h"],
                  width=data.extraData["w"],
                  negative_prompt=data.extraData["negative_prompt"],
                  num_inference_steps=data.extraData["num_steps"],
                  guidance_scale=data.extraData["guidance_scale"], 
                  generator=generator,
                  strength=data.extraData["strength"],
                  num_images_per_prompt=data.extraData["num_samples"],
                  callback=step_callback).images
    
    pipe.scheduler = orig_sampler
    torch.cuda.empty_cache()
    return images

def cn_inpaint(pipe, img, mask, img_masked, data, step_callback):
    generator = get_generation(data.extraData["seed"])
    orig_sampler = pipe.scheduler
    pipe = alter_sampler(pipe, data.extraData["sampler"])

    images = pipe(prompt=data.extraData["text"],
                  image=img,
                  mask_image=mask,
                  control_image=img_masked,
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
    
    pipe.scheduler = orig_sampler
    torch.cuda.empty_cache()
    return images

def cn_tile(pipe, img, data, step_callback):
    generator = get_generation(data.extraData["seed"])
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