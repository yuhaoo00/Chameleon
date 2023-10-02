from cfdraw import *
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

sd_repos = {
    "base_sd_1.5": "runwayml/stable-diffusion-v1-5",
    "base_sd_2.1": "/mnt/Data/CodeML/SD/CKPTS/stabilityai--stable-diffusion-2-1",
    "base_sdxl_1.0": "/mnt/Data/CodeML/SD/CKPTS/stabilityai--stable-diffusion-xl-base-1.0",
}

@cache_resource
def get_sd_t2i(tag):
    repo = sd_repos[tag]
    m = AutoPipelineForText2Image.from_pretrained(repo, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    return m

@cache_resource
def get_sd_i2i(tag):
    repo = sd_repos[tag]
    m = AutoPipelineForImage2Image.from_pretrained(repo, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    return m

@cache_resource
def get_mSAM():
    from extensions.MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    model_type = "vit_t"
    sam_checkpoint = "./weights/mobile_sam.pt"
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).cuda()
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)
    return predictor


def txt2img(pipe, data, step_callback):
    if data.extraData["seed"] != -1:
        generator=torch.Generator(device="cuda").manual_seed(data.extraData["seed"])
    else:
        generator=torch.Generator(device="cuda")
        generator.seed()
    image = pipe(prompt=data.extraData["text"],
                  negative_prompt=data.extraData["negative_prompt"],
                  height=data.extraData["h"],
                  width=data.extraData["w"],
                  num_inference_steps=data.extraData["num_steps"],
                  guidance_scale=data.extraData["guidance_scale"], 
                  generator=generator,
                  callback=step_callback).images[0]
    return image

def img2img(pipe, img, data, step_callback):
    if data.extraData["seed"] != -1:
        generator=torch.Generator(device="cuda").manual_seed(data.extraData["seed"])
    else:
        generator=torch.Generator(device="cuda")
        generator.seed()
    image = pipe(prompt=data.extraData["text"], 
                  image=img, 
                  negative_prompt=data.extraData["negative_prompt"],
                  num_inference_steps=data.extraData["num_steps"],
                  guidance_scale=data.extraData["guidance_scale"], 
                  generator=generator,
                  callback=step_callback).images[0]
    return image