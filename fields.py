from cfdraw import *
from pathlib import Path
from collections import OrderedDict


# common styles
common_styles = dict(
    w=0.75,
    h=0.4,
    maxW=800,
    minH=520,
    useModal=True,
)
common_group_styles = dict(w=230, h=110)
# common diffusion fields
w_field = INumberField(
    default=512,
    min=512,
    max=1280,
    step=64,
    isInt=True,
    label=I18N(
        zh="宽",
        en="Width",
    ),
    tooltip=I18N(
        zh="生成图片的宽度",
        en="The width of the generated image",
    ),
)
h_field = w_field.copy()
h_field.label = I18N(zh="高", en="Height")
h_field.tooltip = I18N(zh="生成图片的高度", en="The height of the generated image")
max_wh_field = INumberField(
    default=1024,
    min=256,
    max=2048,
    step=64,
    isInt=True,
    label=I18N(
        zh="最大宽高",
        en="Max WH",
    ),
    tooltip=I18N(
        zh="把图片传给模型处理前，会先把图片限制在这个尺寸内",
        en="Before passing the image to the model, the image will be ensured to be within this size",
    ),
)
text = ITextField(
    label=I18N(
        zh="提示词",
        en="Prompt",
    ),
    numRows=2,
    tooltip=I18N(
        zh="想要生成的图片的描述",
        en="The description of the image",
    ),
)
version_field = ISelectField(
    default="SDXL v1.0",
    options=["SDXL v1.0", "SD v2.1", "SD v1.5"],
    label=I18N(
        zh="模型", 
        en="Model"
    ),
)
sampler = ISelectField(
    default="Default",
    options=["Default",
             "Euler",
             "Euler a",
             "DPM++ 2M",
             "DPM++ 2M Karras",
             "DPM++ 2M SDE",
             "DPM++ 2M SDE Karras",],
    label=I18N(
        zh="采样器",
        en="Sampler",
    ),
)
num_steps = INumberField(
    default=20,
    min=5,
    max=100,
    step=1,
    isInt=True,
    label=I18N(
        zh="采样步数",
        en="Steps",
    ),
)
negative_prompt = ITextField(
    label=I18N(
        zh="负面词",
        en="Negative Prompt",
    ),
    numRows=2,
    tooltip=I18N(
        zh="不想图片中出现的东西的描述",
        en="The negative description of the image",
    ),
)
num_samples = INumberField(
    default=1,
    min=1,
    max=4,
    step=1,
    isInt=True,
    label=I18N(
        zh="生成数量",
        en="Number of Samples",
    ),
)
guidance_scale = INumberField(
    default=7.5,
    min=-20.0,
    max=25.0,
    step=0.5,
    precision=1,
    label=I18N(
        zh="扣题程度",
        en="Cfg Scale",
    ),
)
seed = INumberField(
    default=-1,
    min=-1,
    max=2**32,
    step=1,
    scale=NumberScale.LOGARITHMIC,
    isInt=True,
    label=I18N(
        zh="随机种子",
        en="Seed",
    ),
    tooltip=I18N(
        zh="'-1' 表示种子将会被随机生成",
        en="'-1' means the seed will be randomly generated",
    ),
)

use_highres = IBooleanField(
    default=False,
    label=I18N(
        zh="高清生成",
        en="Highres",
    ),
    tooltip=I18N(
        zh="生成 2 倍宽高的图片",
        en="Generate images with 2x width & height",
    ),
)
highres_fidelity = INumberField(
    default=0.3,
    min=0.0,
    max=1.0,
    step=0.05,
    label=I18N(
        zh="相似度",
        en="Fidelity",
    ),
    tooltip=I18N(
        zh="高清生成的图片与直出图片的相似度",
        en="How similar the (2x) generated image should be to the (original) generated image",
    ),
    condition="use_highres",
)
strength = INumberField(
    default=0.8,
    min=0.0,
    max=1.0,
    step=0.02,
    label=I18N(
        zh="初始加噪强度",
        en="Initial Noise Strength",
    ),
)

# txt2img
txt2img_fields = OrderedDict(
    w=w_field,
    h=h_field,
    text=text,
    version=version_field,
    sampler=sampler,
    negative_prompt=negative_prompt,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    #use_circular=use_circular,
    seed=seed,
    use_highres=use_highres,
    #lora=lora_field,
    highres_fidelity=highres_fidelity,
    num_samples=num_samples,
)

# img2img fields
img2img_prompt = text.copy()
img2img_prompt.numRows = 3
img2img_fields = OrderedDict(
    text=img2img_prompt,
    strength=strength,
    version=version_field,
    sampler=sampler,
    negative_prompt=negative_prompt,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    #use_circular=use_circular,
    seed=seed,
    use_highres=use_highres,
    #lora=lora_field,
    highres_fidelity=highres_fidelity,
    num_samples=num_samples,
)

# styletransfer fields
scale = INumberField(
    default=1.0,
    min=0.0,
    max=1.0,
    step=0.1,
    label=I18N(
        zh="缩放新增的注意力",
        en="Adapter Scale",
    ),
)
st_prompt = text.copy()
st_prompt.numRows = 3
st_fields = OrderedDict(
    w=w_field,
    h=h_field,
    text=st_prompt,
    negative_prompt=negative_prompt,
    version=ISelectField(
        default="SDXL v1.0",
        options=["SDXL v1.0", "SD v1.5"],
        label=I18N(
            zh="模型", 
            en="Model"
        ),
    ),
    sampler=sampler,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    seed=seed,
    scale=scale,
    num_samples=num_samples,
)

# sd_inpainting / sd_outpainting fields
inpainting_prompt = text.copy()
inpainting_prompt.numRows = 3
inpainting_fields = OrderedDict(
    w=w_field,
    h=h_field,
    version=ISelectField(
        default="SDXL v1.0",
        options=["SDXL v1.0", "SDXL (ft)", "SD v2.1", "SD v2(ft)"],
        label=I18N(
            zh="模型", 
            en="Model"
        ),
    ),
    text=inpainting_prompt,
    strength=strength,
    sampler=sampler,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    negative_prompt=negative_prompt,
    seed=seed,
    num_samples=num_samples,
    focus_mode=IBooleanField(
        default=False,
        label=I18N(zh="聚焦模式", en="Focus Mode"),
        tooltip=I18N(
            zh="启用聚焦模式时，模型会仅关注蒙版区域及周边的一些像素",
            en="When enabled, the model will only focus on the masked region and some surrounding pixels.",
        ),
    ),
)
# controlnet
controlnet_scale = INumberField(
    default=0.5,
    min=0.0,
    max=1.0,
    step=0.02,
    label=I18N(
        zh="条件控制强度",
        en="Conditioning Scale",
    ),
)
tile_prompt = text.copy()
tile_prompt.numRows = 3
tile_fields = OrderedDict(
    w=w_field,
    h=h_field,
    text=tile_prompt,
    sampler=sampler,
    negative_prompt=negative_prompt,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    seed=seed,
    num_samples=num_samples,
    controlnet_conditioning_scale=controlnet_scale,
)

cn_inpainting_fields = OrderedDict(
    w=w_field,
    h=h_field,
    text=inpainting_prompt,
    strength=strength,
    sampler=sampler,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    negative_prompt=negative_prompt,
    seed=seed,
    num_samples=num_samples,
    controlnet_conditioning_scale=controlnet_scale,
    focus_mode=IBooleanField(
        default=False,
        label=I18N(zh="聚焦模式", en="Focus Mode"),
        tooltip=I18N(
            zh="启用聚焦模式时，模型会仅关注蒙版区域及周边的一些像素",
            en="When enabled, the model will only focus on the masked region and some surrounding pixels.",
        ),
    ),
)

# super resolution fields
sr_w_field = w_field.copy()
sr_w_field.default = 2048
sr_w_field.min = 1024
sr_w_field.max = 3072
sr_h_field = h_field.copy()
sr_h_field.default = 2048
sr_h_field.min = 1024
sr_h_field.max = 3072
sr_fields = OrderedDict(
    model=ISelectField(
        default="sd2-x4-upscaler",
        options=["sd2-x4-upscaler", "ESRGAN", "SwinIR"],
        label=I18N(
            zh="模型", 
            en="Model")
    ),
    target_w=sr_w_field,
    target_h=sr_h_field,
)


__all__ = [
    "common_styles",
    "common_group_styles",
    "version_field",
    "txt2img_fields",
    "img2img_fields",
    "st_fields",
    "sr_fields",
    "inpainting_fields",
    "cn_inpainting_fields",
    "tile_fields",
]
