from cfdraw import *
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
    max=1024,
    step=64,
    isInt=True,
    label=I18N(
        zh="宽",
        en="Width",
    ),
)
h_field = w_field.copy()
h_field.label = I18N(zh="高", en="Height")
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
)
text = ITextField(
    label=I18N(
        zh="提示词",
        en="Prompt",
    ),
    numRows=2,
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
)

use_highres = IBooleanField(
    default=False,
    label=I18N(
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

controlnet_fields = OrderedDict(
    type=ISelectField(
            options=["canny", "depth"],
            default="canny",
            label=I18N(zh="参考图类型", en="Hint Type"),
        ),
    hint_url=IImageField(default="", label=I18N(zh="参考图", en="Hint Image")),
    hint_start=INumberField(
        default=0.0,
        min=0.0,
        max=1.0,
        step=0.01,
        label=I18N(zh="参考图生效时机", en="Hint Start"),
    ),
    hint_end=INumberField(
        default=1.0,
        min=0.0,
        max=1.0,
        step=0.01,
        label=I18N(zh="参考图失效时机", en="Hint End"),
    ),
    control_strength=INumberField(
        default=1.0,
        min=0.0,
        max=2.0,
        step=0.01,
        label=I18N(zh="参考强度", en="Control Strength"),
    ),
)

###############################################################################

# txt2img
txt2img_fields = OrderedDict(
    w=w_field,
    h=h_field,
    text=text,
    sampler=sampler,
    negative_prompt=negative_prompt,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    seed=seed,
    num_samples=num_samples,
    use_highres=use_highres,
    highres_scale=highres_scale,
    highres_strength=highres_strength,
    control=IListField(
        label="ControlNet",
        item=controlnet_fields,
    ),
)

# img2img fields
img2img_prompt = text.copy()
img2img_prompt.numRows = 3
img2img_fields = OrderedDict(
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
    control=IListField(
        label="ControlNet",
        item=controlnet_fields,
    ),
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

# demofusion
demofusion_prompt = text.copy()
demofusion_prompt.numRows = 3
demofusion_fields = OrderedDict(
    w=INumberField(
            default=2048,
            min=1024,
            max=4096,
            step=256,
            isInt=True,
            label=I18N(
                zh="宽",
                en="Width",
            ),
        ),
    h=INumberField(
            default=2048,
            min=1024,
            max=4096,
            step=256,
            isInt=True,
            label=I18N(
                zh="高",
                en="Height",
            ),
        ),
    text=demofusion_prompt,
    negative_prompt=negative_prompt,
    sampler=sampler,
    num_steps=num_steps,
    guidance_scale=guidance_scale,
    seed=seed,
    view_batch_size=INumberField(
            default=4,
            min=1,
            max=8,
            step=1,
            isInt=True,
            label=I18N(
                zh="View Batch Size",
                en="View Batch Size",
            ),
        ),
    stride=INumberField(
            default=64,
            min=0,
            max=128,
            step=16,
            isInt=True,
            label=I18N(
                zh="View Stride",
                en="View Stride",
            ),
        ),
    multi_decoder=IBooleanField(
            default=True,
            label=I18N(
                zh="Multi decoder",
                en="Multi decoder",
            ),
        ),
    cosine_scale_1=INumberField(
            default=3,
            min=0,
            max=10,
            isInt=False,
            label=I18N(
                zh="Cosine Scale 1",
                en="Cosine Scale 1",
            ),
        ),
    cosine_scale_2=INumberField(
            default=1,
            min=0,
            max=10,
            isInt=False,
            label=I18N(
                zh="Cosine Scale 2",
                en="Cosine Scale 2",
            ),
        ),
    cosine_scale_3=INumberField(
            default=1,
            min=0,
            max=10,
            isInt=False,
            label=I18N(
                zh="Cosine Scale 3",
                en="Cosine Scale 3",
            ),
        ),
    sigma=INumberField(
            default=0.8,
            min=0,
            max=1.0,
            isInt=False,
            label=I18N(
                zh="Gaussian Sigma",
                en="Gaussian Sigma",
            ),
        ),
    lowvram=IBooleanField(
            default=True,
            label=I18N(
                zh="Lowvram",
                en="Lowvram",
            ),
        ),
)


# Style Fusion
sfusion_fields = OrderedDict(
    strength=INumberField(
        default=0.5,
        min=0.0,
        max=1.0,
        step=0.02,
        label=I18N(
            zh="Creativity",
            en="Creativity",
        ),
    ),
    pad_strength=INumberField(
        default=20,
        min=0,
        max=40,
        step=1,
        isInt=True,
        label=I18N(
            zh="Semanticity",
            en="Semanticity",
        ),
    ),
    blur_strength=INumberField(
        default=5,
        min=0,
        max=10,
        step=1,
        isInt=True,
        label=I18N(
            zh="Soft Edge",
            en="Soft Edge",
        ),
    )
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


# super resolution
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
    "demofusion_fields",
    "matting_fields",
    "sfusion_fields",
]
