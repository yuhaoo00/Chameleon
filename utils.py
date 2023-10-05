from cfdraw import *
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from extensions.IPAdapter.ip_adapter import IPAdapterXL, IPAdapter

sd_repos = {
    "SD 1.5": "runwayml/stable-diffusion-v1-5",
    "SD 2.1": "/mnt/Data/CodeML/SD/CKPTS/stabilityai--stable-diffusion-2-1",
    "SDxl 1.0": "/mnt/Data/CodeML/SD/CKPTS/stabilityai--stable-diffusion-xl-base-1.0",
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
    torch.cuda.empty_cache()
    repo = sd_repos[tag]
    t2i = AutoPipelineForText2Image.from_pretrained(repo, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    return t2i

def get_sd_i2i(tag, t2i=None):
    torch.cuda.empty_cache()
    if t2i is None:
        t2i = get_sd_t2i(tag)
    i2i = AutoPipelineForImage2Image.from_pipe(t2i).to("cuda")
    return i2i

def get_ipadapter(tag, t2i=None):
    torch.cuda.empty_cache()
    if t2i is None:
        t2i = get_sd_t2i(tag)
    if tag == "SDxl 1.0":
        ip_model = IPAdapterXL(t2i, 
                            "/mnt/Data/CodeML/SD/CKPTS/IP-Adapter/sdxl_models/image_encoder", 
                            "/mnt/Data/CodeML/SD/CKPTS/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin",
                            "cuda")
    elif tag == "SD 1.5":
        ip_model = IPAdapter(t2i, 
                            "/mnt/Data/CodeML/SD/CKPTS/IP-Adapter/models/image_encoder", 
                            "/mnt/Data/CodeML/SD/CKPTS/IP-Adapter/models/ip-adapter_sd15.bin",
                            "cuda")
    else:
        ip_model = None
        print("Unsupported SD Version")
    return ip_model



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
                          num_inference_steps=data.extraData["num_steps"],
                          generator=generator,
                          guidance_scale=data.extraData["guidance_scale"],
                          scale=data.extraData["scale"],
                          num_samples=data.extraData["num_samples"],
                          callback=step_callback)
    torch.cuda.empty_cache()
    return images

class ExpandEdge_(nn.Module):
    def __init__(self, strength=1):
        super(ExpandEdge_, self).__init__()
        self.kernel_size = 3+2*strength
        kernel = torch.ones((1,1,self.kernel_size,self.kernel_size), dtype=torch.float)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, mask):
        # mask [1,1,h,w]
        x = F.conv2d(mask, self.weight, padding=self.kernel_size//2)
        mask_new = torch.zeros_like(x)
        mask_new[x>0 and x<(self.kernel_size**2)] = 1
        return mask_new
    
def ExpandEdge(mask, strength=1):
    h, w = mask.shape
    kernel_size = 3+2*strength
    mask = torch.from_numpy(mask).to("cuda").float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1,1,kernel_size,kernel_size), device="cuda")
    x = F.conv2d(mask, kernel, padding=kernel_size//2)
    res = ((x>0) & (x<(kernel_size**2)))
    res = res.cpu().numpy().reshape(h, w)
    return res