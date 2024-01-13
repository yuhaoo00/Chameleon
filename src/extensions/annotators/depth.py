
import numpy as np
import torch
from einops import rearrange
import torch
from PIL import Image
from .midas.dpt_depth import DPTDepthModel

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class MidasDetector:
    def __init__(self, model_path):
        self.model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        ).cuda().eval()
        self.model.train = disabled_train

    @torch.no_grad()
    def __call__(self, input_image, *args, **kargs):
        input_image = np.asarray(input_image.convert("RGB"))
        assert input_image.ndim == 3
        image_depth = input_image
        
        image_depth = torch.from_numpy(image_depth).float().cuda()
        image_depth = image_depth / 127.5 - 1.0
        image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
        depth = self.model(image_depth)[0]

        depth -= torch.min(depth)
        depth /= torch.max(depth)
        depth = depth.cpu().numpy()
        depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

        depth_image = Image.fromarray(depth_image)

        return depth_image
