from cfdraw import *
import torch

def txt2img(pipe, data, step_callback):
    if data.extraData["seed"] != -1:
        generator=torch.Generator(device="cuda").manual_seed(data.extraData["seed"])
    else:
        generator=torch.Generator(device="cuda")
        generator.seed()
    images = pipe(prompt=data.extraData["text"],
                  negative_prompt=data.extraData["negative_prompt"],
                  height=data.extraData["h"],
                  width=data.extraData["w"],
                  num_inference_steps=data.extraData["num_steps"],
                  guidance_scale=data.extraData["guidance_scale"], 
                  generator=generator,
                  num_images_per_prompt=data.extraData["num_samples"],
                  callback=step_callback).images
    torch.cuda.empty_cache()
    return images

def img2img(pipe, img, data, step_callback):
    if data.extraData["seed"] != -1:
        generator=torch.Generator(device="cuda").manual_seed(data.extraData["seed"])
    else:
        generator=torch.Generator(device="cuda")
        generator.seed()
    images = pipe(prompt=data.extraData["text"], 
                  image=img, 
                  negative_prompt=data.extraData["negative_prompt"],
                  num_inference_steps=data.extraData["num_steps"],
                  guidance_scale=data.extraData["guidance_scale"], 
                  generator=generator,
                  num_images_per_prompt=data.extraData["num_samples"],
                  callback=step_callback).images
    torch.cuda.empty_cache()
    return images


def style_transfer(pipe, img, data, step_callback):
    if data.extraData["seed"] != -1:
        generator=torch.Generator(device="cuda").manual_seed(data.extraData["seed"])
    else:
        generator=torch.Generator(device="cuda")
        generator.seed()
    images = pipe.generate(pil_image=img,
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