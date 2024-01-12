from typing import Optional, List, Union
from pydantic import BaseModel

class Inputdata(BaseModel):
    text: str
    num_steps: int = 40
    guidance_scale: float = 7.5
    seed: Optional[int] = -1
    negative_prompt: Optional[str] = None
    image: Optional[str] = None
    strength: Optional[float] = 0.3
    h: Optional[int] = 1024
    w: Optional[int] = 1024
    num_samples: Optional[int] = 1
    use_hrfix: Optional[bool] = False
    hrfix_scale: Optional[float] = 2.0
    hrfix_strength: Optional[float] = 0.3
    hrfix_steps_ratio: Optional[float] = 0.5
    control: Optional[Union[str, List[str]]] = None
    control_hint: Optional[Union[str, List[str]]] = None
    control_hint_skip: Optional[Union[bool, List[bool]]] = None
    control_hint_start: Optional[Union[float, List[float]]] = None
    control_hint_end: Optional[Union[float, List[float]]] = None
    control_strength: Optional[Union[float, List[float]]] = None

class Outputdata(BaseModel):
    imgs: List[str]
    time: float

class Inputdata_upscale(BaseModel):
    image: str
    tag: str = "ESRGAN-general"
    scale: float = 2.0


class Inputdata_inpaint(BaseModel):
    text: str
    image: str
    mask: str
    num_steps: int = 40
    guidance_scale: float = 7.5
    seed: Optional[int] = -1
    negative_prompt: Optional[str] = None
    strength: Optional[float] = 0.3
    h: Optional[int] = 1024
    w: Optional[int] = 1024
    num_samples: Optional[int] = 1
    focus_mode: bool = False

class Inputdata_demofusion(BaseModel):
    text: str
    image: str
    num_steps: int = 40
    guidance_scale: float = 7.5

    seed: Optional[int] = -1
    negative_prompt: Optional[str] = None
    h: Optional[int] = 1024
    w: Optional[int] = 1024
    view_batch_size: Optional[int] = 4
    stride: Optional[int] = 64
    multi_decoder: Optional[bool] = True
    cosine_scale_1: Optional[float] = 3.
    cosine_scale_2: Optional[float] = 1.
    cosine_scale_3: Optional[float] = 1.
    sigma: Optional[float] = 0.8
    lowvram: Optional[bool] = True

