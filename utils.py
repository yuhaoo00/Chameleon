from cfdraw import *
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

sd_repos = {
    "base_sd_1.5": "runwayml/stable-diffusion-v1-5",
    "base_sd_2.1": "/mnt/Data/CodeML/SD/CKPTS/stabilityai--stable-diffusion-2-1",
    "base_sdxl_1.0": "/mnt/Data/CodeML/SD/CKPTS/stabilityai--stable-diffusion-xl-base-1.0",
}

def transform_mask(mask, resize=None):
    if resize is not None:
        mask = mask.resize(resize)
    # PIL Image to numpy array
    mask = np.array(mask)
    mask[mask>0] = 1
    #mask = -mask+1
    return mask[None,:,:]#.astype(bool)

def mask_to_box(mask):
    mask = np.array(mask)
    h, w = mask.shape
    mask_x = np.sum(mask, axis=0)
    mask_y = np.sum(mask, axis=1)
    x0, x1 = 0, w-1
    y0, y1 = 0, h-1
    while x0 < w:
        if mask_x[x0] != 0:
            break
        x0 += 1
    while x1 > -1:
        if mask_x[x1] == 0:
            break
        x1 -= 1
    while y0 < h:
        if mask_y[y0] != 0:
            break
        y0 += 1
    while y1 > -1:
        if mask_y[y1] == 0:
            break
        y1 -= 1
    return np.array([x0, y0, x1, y1])

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
    from extensions.MobileSAM import sam_model_registry, SamPredictor
    model_type = "vit_t"
    sam_checkpoint = "/mnt/Data/CodeML/SD/CKPTS/mobile_sam.pt"
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


class ExpandEdge(nn.Module):
    def __init__(self, strength=1):
        super(ExpandEdge, self).__init__()
        self.kernel_size = 3+2*strength
        kernel = torch.ones((1,1,self.kernel_size,self.kernel_size), dtype=torch.float)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, mask):
        # mask [1,1,h,w]
        x = F.conv2d(mask, self.weight, padding=self.kernel_size//2)
        mask_new = torch.zeros_like(x)
        mask_new[x>0 and x<(self.kernel_size**2)] = 1
        return mask_new