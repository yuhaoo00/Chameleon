from cfdraw import *
from pathlib import Path
from collections import OrderedDict


# common styles
common_styles = dict(
    w=0.4,
    h=0.65,
    maxW=520,
    minH=720,
    useModal=True,
)
common_group_styles = dict(w=180, h=110)
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
    default="SDXL",
    options=["SDXL", "SDv2", "SDv1"],
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
        zh="高清生成",
        en="Highres",
    ),
)
highres_scale = INumberField(
    default=2.0,
    min=1.1,
    max=4.0,
    step=0.1,
    label=I18N(
        zh="上采样倍数",
        en="Upscale by",
    ),
    tooltip=I18N(
        zh="上采样倍数",
        en="Upscale by",
    ),
    condition="use_highres",
)
highres_strength = INumberField(
    default=0.5,
    min=0.0,
    max=1.0,
    step=0.05,
    label=I18N(
        zh="相似度",
        en="strength",
    ),
    tooltip=I18N(
        zh="相似度",
        en="strength",
    ),
    condition="use_highres",
)
strength = INumberField(
    default=0.8,
    min=0.0,
    max=1.0,
    step=0.02,
    label=I18N(
        zh="图生图强度",
        en="Img2Img Strength",
    ),
)
# Canny Annotator
low_threshold = INumberField(
    default=100,
    min=0,
    max=255,
    step=5,
    isInt=True,
    label=I18N(
        zh="最小阈值",
        en="low threshold",
    ),
)

high_threshold = INumberField(
    default=200,
    min=0,
    max=255,
    step=5,
    isInt=True,
    label=I18N(
        zh="最大阈值",
        en="high threshold",
    ),
)

canny_fields = OrderedDict(
    low_threshold=low_threshold,
    high_threshold=high_threshold,
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
    seed=seed,
    num_samples=num_samples,
    use_highres=use_highres,
    highres_scale=highres_scale,
    highres_strength=highres_strength,
)

# img2img fields
img2img_prompt = text.copy()
img2img_prompt.numRows = 3
img2img_fields = OrderedDict(
    version=version_field,
    text=img2img_prompt,
    negative_prompt=negative_prompt,
    sampler=sampler,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    strength=strength,
    seed=seed,
    num_samples=num_samples,
    use_highres=use_highres,
    highres_scale=highres_scale,
    highres_strength=highres_strength,
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
    version=ISelectField(
        default="SDXL",
        options=["SDXL", "SDv1"],
        label=I18N(
            zh="模型", 
            en="Model"
        ),
    ),
    w=w_field,
    h=h_field,
    text=st_prompt,
    negative_prompt=negative_prompt,
    sampler=sampler,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    scale=scale,
    seed=seed,
    num_samples=num_samples,
)

# sd_inpainting / sd_outpainting fields
inpainting_prompt = text.copy()
inpainting_prompt.numRows = 3
inpainting_fields = OrderedDict(
    version=ISelectField(
        default="SDXL",
        options=["SDXL", "SDXL (ft)", "SDv2", "SDv2 (ft)", "SDv1", "SDv1 (ft)"],
        label=I18N(
            zh="模型", 
            en="Model"
        ),
    ),
    w=w_field,
    h=h_field,
    text=inpainting_prompt,
    negative_prompt=negative_prompt,
    sampler=sampler,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    strength=strength,
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
    negative_prompt=negative_prompt,
    sampler=sampler,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    controlnet_conditioning_scale=controlnet_scale,
    seed=seed,
    num_samples=num_samples,
)

cn_inpainting_fields = OrderedDict(
    w=w_field,
    h=h_field,
    text=inpainting_prompt,
    negative_prompt=negative_prompt,
    strength=strength,
    sampler=sampler,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    controlnet_conditioning_scale=controlnet_scale,
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

# Smart Fusing
box_padding = INumberField(
    default=0.1,
    min=0.0,
    max=0.5,
    step=0.02,
    label=I18N(
        zh="聚焦范围",
        en="Box Padding Scale",
    ),
)
mask_expand = INumberField(
    default=10,
    min=0,
    max=50,
    step=1,
    isInt=True,
    label=I18N(
        zh="遮罩外扩",
        en="Mask Expanding Scale",
    ),
)
smart_fusing_fields = OrderedDict(
    version=ISelectField(
        default="SDXL",
        options=["SDXL", "SDXL (ft)", "SDv1", "SDv1 (ft)"],
        label=I18N(
            zh="Inpaint模型", 
            en="Inpaint Model"
        ),
    ),
    cn_type=ISelectField(
        default="canny",
        options=["canny", "soft edge", "zoe depth"],
        label=I18N(
            zh="ControlNet模型", 
            en="ControlNet Model"
        ),
    ),
    w=w_field,
    h=h_field,
    text=inpainting_prompt,
    negative_prompt=negative_prompt,
    sampler=sampler,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    strength=strength,
    controlnet_conditioning_scale=controlnet_scale,
    seed=seed,
    num_samples=num_samples,
    guess_mode=IBooleanField(
        default=False,
        label=I18N(zh="推测模式", en="Guess Mode"),
        tooltip=I18N(
            zh="即使删除所有提示，ControlNet编码器也会尝试识别输入图像的内容",
            en="The ControlNet encoder tries to recognize the content of the input image even if you remove all prompts.",
        ),
    ),
    box_padding=box_padding,
    mask_expand=mask_expand,
)

# matting
matting_fields = OrderedDict(
    version=ISelectField(
        default="vit_t",
        options=["vit_t", "vit_h"],
        label=I18N(
            zh="模型", 
            en="Model"
        ),
    ),
)


# super resolution fields
sr_fields = OrderedDict(
    version=ISelectField(
        default="ESRGAN-general",
        options=["ESRGAN-general", "ESRGAN-anime"],
        label=I18N(
            zh="模型", 
            en="Model")
    ),
    scale=INumberField(
        default=2.0,
        min=1.1,
        max=4.0,
        step=0.1,
        label=I18N(
            zh="上采样倍数",
            en="Upscale by",
        ),
        tooltip=I18N(
            zh="上采样倍数",
            en="Upscale by",
        ),
    )
)


__all__ = [
    "common_styles",
    "common_group_styles",
    "version_field",
    "canny_fields",
    "txt2img_fields",
    "img2img_fields",
    "st_fields",
    "sr_fields",
    "inpainting_fields",
    "cn_inpainting_fields",
    "smart_fusing_fields",
    "tile_fields",
    "matting_fields",
]
