from cfdraw import *
import torch
from extensions.IPAdapter.ip_adapter import IPAdapterXL, IPAdapter
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers import DPMSolverMultistepScheduler, 	EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler
from extensions.realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

main_dir = "/mnt/Data/CodeML/SD/CKPTS/"

sd_repos = {
    "SDv1": main_dir+"runwayml--stable-diffusion-v1-5",
    "SDv2": main_dir+"stabilityai--stable-diffusion-2-1",
    "SDXL": main_dir+"stabilityai--stable-diffusion-xl-base-1.0",
}

sam_paths = {
    "vit_t": main_dir+"mobile_sam.pt",
    "vit_h": main_dir+"sam_vit_h_4b8939.pth",
}

upscaler_paths = {
    "ESRGAN-general": main_dir+"Real-ESRGAN/RealESRGAN_x4plus.pth",
    "ESRGAN-anime": main_dir+"Real-ESRGAN/RealESRGAN_x4plus_anime_6B.pth",
}

#@cache_resource
def get_sd_t2i(tag, cpu_off=True):
    torch.cuda.empty_cache()
    repo = sd_repos[tag]
    t2i = AutoPipelineForText2Image.from_pretrained(repo, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    if cpu_off:
        t2i.enable_model_cpu_offload()
    return t2i

def get_sd_i2i(tag):
    torch.cuda.empty_cache()
    t2i = get_sd_t2i(tag)
    i2i = AutoPipelineForImage2Image.from_pipe(t2i)
    return i2i

def get_sd_inpaint(tag, cpu_off=True):
    torch.cuda.empty_cache()
    if tag == "SDv2 (ft)":
        pipe = AutoPipelineForInpainting.from_pretrained(main_dir+"stabilityai--stable-diffusion-2-inpainting",
                                                         torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    elif tag == "SDv1 (ft)":
        pipe = AutoPipelineForInpainting.from_pretrained(main_dir+"runwayml--stable-diffusion-inpainting",
                                                         torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    elif tag == "SDXL (ft)":
        pipe = AutoPipelineForInpainting.from_pretrained(main_dir+"diffusers--stable-diffusion-xl-1.0-inpainting-0.1",
                                                         torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    else:
        t2i = get_sd_t2i(tag, False)
        pipe = AutoPipelineForInpainting.from_pipe(t2i)
    if cpu_off:
        pipe.enable_model_cpu_offload()
    return pipe


def get_style_inpaint(tag, cn_tag):
    from pipelines.sd15_inpaint_control import StyleControlInpaint_sd15
    from pipelines.sdxl_inpaint_control import StyleControlInpaint_sdxl
    torch.cuda.empty_cache()
    inpaint_pipe = get_sd_inpaint(tag, False)
    if 'v1' in tag:
        if cn_tag == 'canny':
            cn = ControlNetModel.from_pretrained(main_dir+"lllyasviel--control_v11p_sd15_canny",
                                                 torch_dtype=torch.float16,
                                                 use_safetensors=True,
                                                 variant="fp16")
        elif cn_tag == 'soft edge':
            cn = ControlNetModel.from_pretrained(main_dir+"lllyasviel--control_v11p_sd15_softedge",
                                                 torch_dtype=torch.float16,
                                                 use_safetensors=True,
                                                 variant="fp16")
        elif cn_tag == 'zoe depth':
            cn = ControlNetModel.from_pretrained(main_dir+"lllyasviel--control_v11f1p_sd15_depth",
                                                 torch_dtype=torch.float16,
                                                 use_safetensors=True,
                                                 variant="fp16")
        pipe = StyleControlInpaint_sd15(controlnet=cn, **inpaint_pipe.components)
    elif 'XL' in tag:
        if cn_tag == 'canny':
            cn = ControlNetModel.from_pretrained(main_dir+"diffusers--controlnet-canny-sdxl-1.0",
                                                 torch_dtype=torch.float16,
                                                 use_safetensors=True,
                                                 variant="fp16")
        elif cn_tag == 'zoe depth':
            cn = ControlNetModel.from_pretrained(main_dir+"diffusers--controlnet-zoe-depth-sdxl-1.0",
                                                 torch_dtype=torch.float16,
                                                 use_safetensors=True)
        pipe = StyleControlInpaint_sdxl(controlnet=cn, **inpaint_pipe.components)
    pipe.enable_model_cpu_offload()
    return pipe

def get_controlnet(tag):
    torch.cuda.empty_cache()
    if tag == "v11_sd15_inapint":
        t2i = get_sd_t2i("SDv1", False)
        cn = ControlNetModel.from_pretrained(main_dir+"lllyasviel--control_v11p_sd15_inpaint",
                                             torch_dtype=torch.float16,
                                             use_safetensors=True,
                                             variant="fp16")
        pipe = StableDiffusionControlNetInpaintPipeline(controlnet=cn, **t2i.components)
    elif tag == "v11_sd15_tile":
        t2i = get_sd_t2i("SDv1", False)
        cn = ControlNetModel.from_pretrained(main_dir+"lllyasviel--control_v11f1e_sd15_tile",
                                             torch_dtype=torch.float16
                                             )
        pipe = StableDiffusionControlNetPipeline(controlnet=cn, **t2i.components)
    pipe.enable_model_cpu_offload()
    return pipe

def get_ipadapter(tag):
    torch.cuda.empty_cache()
    t2i = get_sd_t2i(tag)
    if tag == "SDXL":
        ip_model = IPAdapterXL(t2i, 
                            main_dir+"IP-Adapter/sdxl_models/image_encoder", 
                            main_dir+"IP-Adapter/sdxl_models/ip-adapter_sdxl.bin",
                            "cuda")
    elif tag == "SDv1":
        ip_model = IPAdapter(t2i, 
                            main_dir+"IP-Adapter/models/image_encoder", 
                            main_dir+"IP-Adapter/models/ip-adapter_sd15.bin",
                            "cuda")
    else:
        ip_model = None
        print("Unsupported SD Version")
    return ip_model


def get_mSAM(tag):
    from extensions.MobileSAM import sam_model_registry, SamPredictor
    sam_checkpoint = sam_paths[tag]
    mobile_sam = sam_model_registry[tag](checkpoint=sam_checkpoint).cuda()
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

def get_upscaler(tag):
    if tag == "ESRGAN-general":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    elif tag == "ESRGAN-anime":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=4,
        model_path=upscaler_paths[tag],
        dni_weight=None,
        model=model)
    
    return upsampler