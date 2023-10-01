from cfdraw import *
import torch
from diffusers import DiffusionPipeline

sd_repos = {
    "base_sd_1.5": "runwayml/stable-diffusion-v1-5",
    "base_sd_2.1": "stabilityai/stable-diffusion-2-1",
    "base_sdxl_1.0": "stabilityai/stable-diffusion-xl-base-1.0",
}

@cache_resource
def get_model(tag):
    repo = sd_repos[tag]
    m = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16, proxies={'http':'http://127.0.0.1:8889', 'https':'http://127.0.0.1:8889'}).to("cuda")
    return m

def txt2img(pipe, data, step_callback):
    images = pipe(prompt=data.extraData["prompt"]).images
    return images

def img2img(pipe, data, step_callback):
    images = pipe(prompt=data.extraData["prompt"]).images
    return images