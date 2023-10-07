from cfdraw import *
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from extensions.IPAdapter.ip_adapter import IPAdapterXL, IPAdapter

sd_repos = {
    "SD 1.5": "runwayml/stable-diffusion-v1-5",
    "SD 2.1": "/mnt/Data/CodeML/SD/CKPTS/stabilityai--stable-diffusion-2-1",
    "SDxl 1.0": "/mnt/Data/CodeML/SD/CKPTS/stabilityai--stable-diffusion-xl-base-1.0",
}

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
