from typing import Optional, List
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

