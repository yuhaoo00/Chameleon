from cfdraw import *
import torch
from extensions.IPAdapter.ip_adapter import IPAdapterXL, IPAdapter
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting
from diffusers import StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers import DPMSolverMultistepScheduler, 	EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler

main_dir = "/mnt/Data/CodeML/SD/CKPTS/"

sd_repos = {
    "SD v1.5": main_dir+"runwayml--stable-diffusion-v1-5",
    "SD v2.1": main_dir+"stabilityai--stable-diffusion-2-1",
    "SDXL v1.0": main_dir+"stabilityai--stable-diffusion-xl-base-1.0",
}

@cache_resource
def get_sd_t2i(tag):
    torch.cuda.empty_cache()
    repo = sd_repos[tag]
    t2i = AutoPipelineForText2Image.from_pretrained(repo, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    return t2i

def get_sd_i2i(tag):
    torch.cuda.empty_cache()
    t2i = get_sd_t2i(tag)
    i2i = AutoPipelineForImage2Image.from_pipe(t2i).to("cuda")
    return i2i

def get_sd_inpaint(tag):
    torch.cuda.empty_cache()
    if tag == "SD v2(ft)":
        pipe = AutoPipelineForInpainting.from_pretrained(main_dir+"stabilityai--stable-diffusion-2-inpainting",
                                                         torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    elif tag == "SDXL (ft)":
        pipe = AutoPipelineForInpainting.from_pretrained(main_dir+"diffusers--stable-diffusion-xl-1.0-inpainting-0.1",
                                                         torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    else:
        t2i = get_sd_t2i(tag)
        pipe = AutoPipelineForInpainting.from_pipe(t2i).to("cuda")
    return pipe

def get_controlnet(tag):
    torch.cuda.empty_cache()
    if tag == "v11_sd15_inapint":
        t2i = get_sd_t2i("SD v1.5")
        cn = ControlNetModel.from_pretrained(main_dir+"lllyasviel--control_v11p_sd15_inpaint",
                                             torch_dtype=torch.float16,
                                             use_safetensors=True,
                                             variant="fp16").to("cuda")
        pipe = StableDiffusionControlNetInpaintPipeline(controlnet=cn, **t2i.components)
    return pipe

def get_ipadapter(tag):
    torch.cuda.empty_cache()
    t2i = get_sd_t2i(tag)
    if tag == "SDXL v1.0":
        ip_model = IPAdapterXL(t2i, 
                            main_dir+"IP-Adapter/sdxl_models/image_encoder", 
                            main_dir+"IP-Adapter/sdxl_models/ip-adapter_sdxl.bin",
                            "cuda")
    elif tag == "SD v1.5":
        ip_model = IPAdapter(t2i, 
                            main_dir+"IP-Adapter/models/image_encoder", 
                            main_dir+"IP-Adapter/models/ip-adapter_sd15.bin",
                            "cuda")
    else:
        ip_model = None
        print("Unsupported SD Version")
    return ip_model


@cache_resource
def get_mSAM():
    from extensions.MobileSAM import sam_model_registry, SamPredictor
    model_type = "vit_t"
    sam_checkpoint = main_dir+"mobile_sam.pt"
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).cuda()
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)
    return predictor

def alter_sampler(pipe, sampler_name):
    if sampler_name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "DPM++ 2M":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True).from_config(pipe.scheduler.config)
    elif sampler_name == "DPM++ 2M SDE":
        pipe.scheduler = DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++").from_config(pipe.scheduler.config)
    elif sampler_name == "DPM++ 2M SDE Karras":
        pipe.scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++").from_config(pipe.scheduler.config)
    return pipe
