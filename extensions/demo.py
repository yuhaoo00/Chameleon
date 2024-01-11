
'''
Installation on Windows for GPU with 8 Gb of VRAM and xformers:

git clone "https://github.com/PRIS-CV/DemoFusion"
cd DemoFusion
python -m venv venv
venv\Scripts\activate
pip install -U "xformers==0.0.22.post7+cu118" --index-url https://download.pytorch.org/whl/cu118
pip install "diffusers==0.21.4" "matplotlib==3.8.2" "transformers==4.35.2" "accelerate==0.25.0"
'''

from demofusion import DemoFusionSDXLPipeline
from PIL import Image

import torch, torchvision
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained("/mnt/Data/CodeML/SD/CKPTS/madebyollin--sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)

model_ckpt = "/mnt/Data/CodeML/SD/CKPTS/stabilityai--stable-diffusion-xl-base-1.0"
pipe = DemoFusionSDXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16, vae=vae, use_safetensors=True, variant="fp16")
pipe = pipe.to("cuda")

prompt = "tree"

img = Image.open("/mnt/Data/CodeML/SD/Chameleon/.images/11327dcb1227473dabffd2610ebf1f63.png")


images = pipe(prompt, negative_prompt=None, image_lr=img,
              height=1024, width=1024, view_batch_size=4, stride=64,
              num_inference_steps=40, guidance_scale=7.5,
              cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
              multi_decoder=True, show_image=False, lowvram=True,
              generator=torch.Generator(device="cuda").manual_seed(123),
             )

for i, image in enumerate(images):
    image.save('image_'+str(i)+'.png')