import os
import cv2
import math
import json
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import TypeVar
from typing import Optional
from pathlib import Path
from pydantic import BaseModel
from cftool.misc import shallow_copy_dict
from cfcreator.common import InpaintingMode
from cflearn.misc.toolkit import new_seed
from cfcreator.sdks.apis import ALL_LATENCIES_KEY
import torchvision.transforms.functional as TF

from cfdraw import *

from utils import *
from fields import *


# plugins

class Txt2Img(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=constants.TEXT_TO_IMAGE_ICON,
            tooltip=I18N(
                zh="生成符合文本描述的图片",
                en="Text to Image",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="文本生成图片",
                    en="Text to Image",
                ),
                numColumns=2,
                definitions=txt2img_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        def callback(step: int, num_steps: int, latents: torch.FloatTensor) -> bool:
            return self.send_progress((step+1) / data.extraData["num_steps"])
        
        pipe = get_sd_t2i(data.extraData["version"])
        return txt2img(pipe, data, callback)


class Img2Img(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=constants.IMAGE_TO_IMAGE_ICON,
            tooltip=I18N(
                zh="以当前图片为参考图，生成符合文本描述的图片",
                en="Image to Image",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="垫图生成",
                    en="Image to Image",
                ),
                numColumns=2,
                definitions=img2img_fields,
                #exportFullImages=True,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        def callback(step: int, num_steps: int, latents: torch.FloatTensor) -> bool:
            return self.send_progress((step+1) / data.extraData["num_steps"])
        
        img = await self.load_image(data.nodeData.src)
        pipe = get_sd_i2i(data.extraData["version"])
        return img2img(pipe, img, data, callback)


class StyleTransfer(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=constants.IMAGE_TO_IMAGE_ICON,
            tooltip=I18N(
                zh="以当前图片为风格参考图，生成符合文本描述的图片",
                en="Style Transfer"
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="风格迁移",
                    en="Style Transfer",
                ),
                numColumns=2,
                definitions=st_fields,
                #exportFullImages=True,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        def callback(step: int, num_steps: int, latents: torch.FloatTensor) -> bool:
            return self.send_progress((step+1) / data.extraData["num_steps"])
        
        img = await self.load_image(data.nodeData.src)
        pipe = get_ipadapter(data.extraData["version"])
        return style_transfer(pipe, img, data, callback)


class SR(IFieldsPlugin):
    image_should_audit = False

    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=320,
            h=300,
            src=constants.SR_ICON,
            tooltip=I18N(
                zh="图片变高清",
                en="Super Resolution",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="图片变高清",
                    en="Super Resolution",
                ),
                definitions=sr_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        return


class Matting(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=240,
            h=110,
            src=constants.SOD_ICON,
            tooltip=I18N(
                zh="抠图",
                en="Image Matting",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="抠图",
                    en="Image Matting",
                ),
                definitions={},
                #exportFullImages=True,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:

        url_node = self.filter(data.nodeDataList, SingleNodeType.IMAGE)[0]
        mask_node = self.filter(data.nodeDataList, SingleNodeType.PATH)[0]
        img = await self.load_image(url_node.src)
        img = img.convert("RGB")
        mask = await self.load_image(mask_node.src)
        mask = mask.convert("L")
        box = mask_to_box(mask)
        
        model = get_mSAM()
        model.set_image(np.array(img))
        mask_fined = model.predict(box=box,
                                    #mask_input=mask[None,:,:],
                                    #point_coords=np.array([[463, 455]]),
                                    #point_labels=np.array([1]),
                                    multimask_output=False)[0][0,:]
        
        img_masked = np.concatenate((np.array(img), (mask_fined[:,:,None]*255).astype(np.uint8)), axis=2)
        img_masked = Image.fromarray(img_masked)

        return [img_masked]
    
class Fusing(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=240,
            h=110,
            src=constants.SOD_ICON,
            tooltip=I18N(
                zh="融合",
                en="Images Fusing",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="融合",
                    en="Images Fusing",
                ),
                definitions={},
                #exportFullImages=True,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:

        url_nodes = self.filter(data.nodeDataList, SingleNodeType.IMAGE)

        data0 = url_nodes[0]
        data1 = url_nodes[1]
        img0 = await self.load_image(data0.meta['data']['response']['value']['url'])
        img1 = await self.load_image(data1.meta['data']['response']['value']['url'])
        img0 = img0.convert("RGBA")
        img1 = img1.convert("RGBA")
        if data0.z > data1.z:
            data0, data1 = data1, data0
            img0, img1 = img1, img0
        # img0 在 img1 上面

        def get_angle(a,c,w,h):
            theta = np.arccos(a/w)
            if np.sin(theta)*c <= 0:
                theta = -theta
            return theta
        
        theta0 = get_angle(data0.transform.a, data0.transform.c, data0.w, data0.h)
        theta1 = get_angle(data1.transform.a, data1.transform.c, data1.w, data1.h)


        img0 = img0.resize(size=[int(data0.w), int(data0.h)])
        img1 = img1.resize(size=[int(data1.w), int(data1.h)]).rotate(np.degrees(theta1), expand=True)
        
        new = Image.new("RGBA", [int(max(data1.w, data0.x-data1.x+data0.w)), int(max(data1.h, data0.y-data1.y+data0.h))], color=(0,0,0,0))
        new.paste(img0, (int(data0.x-data1.x), int(data0.y-data1.y)), mask=img0.getchannel("A"))
        new = new.rotate(np.degrees(theta0), center=[int(data0.x-data1.x), int(data0.y-data1.y)])

        img1.paste(new, (0,0), mask=new.getchannel("A"))

        mask = np.array(new.getchannel("A"))
        mask_edge = ExpandEdge(mask, 10)
        mask_edge = Image.fromarray(mask_edge[:int(data1.h),:int(data1.w)])


        return [img1, mask_edge]


# groups
class StaticPlugins(IPluginGroup):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_group_styles,
            tooltip=I18N(
                zh="一组用于生成图片的插件",
                en="A set of plugins for generating images",
            ),
            pivot=PivotType.RIGHT,
            follow=False,
            pluginInfo=IPluginGroupInfo(
                name=I18N(
                    zh="创意工具箱",
                    en="Creator Toolbox",
                ),
                header=I18N(
                    zh="创意工具箱",
                    en="Creator Toolbox",
                ),
                plugins={
                    "txt2img": Txt2Img,
                },
            ),
        )


class ImageFollowers(IPluginGroup):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=common_group_styles["w"],
            h=164,
            tooltip=I18N(
                zh="一组将 AI 技术应用于当前图片的插件",
                en="A set of plugins that apply AI techniques to the given image",
            ),
            nodeConstraint=NodeConstraints.IMAGE,
            pivot=PivotType.RT,
            follow=True,
            pluginInfo=IPluginGroupInfo(
                name=I18N(
                    zh="图片工具箱",
                    en="Image Toolbox",
                ),
                header=I18N(
                    zh="图片工具箱",
                    en="Image Toolbox",
                ),
                plugins={
                    "img2img": Img2Img,
                    "styletransfer": StyleTransfer,
                },
            ),
        )

class ImageAndMaskFollowers(IPluginGroup):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_group_styles,
            offsetX=-48,
            expandOffsetX=64,
            tooltip=I18N(
                zh="一组利用当前图片+蒙版来进行生成的插件",
                en="A set of plugins which uses an image and a mask to generate images",
            ),
            nodeConstraintRules=NodeConstraintRules(
                exactly=[NodeConstraints.IMAGE, NodeConstraints.PATH]
            ),
            pivot=PivotType.RT,
            follow=True,
            pluginInfo=IPluginGroupInfo(
                name=I18N(
                    zh="蒙版工具箱",
                    en="Image & Mask Toolbox",
                ),
                header=I18N(
                    zh="蒙版工具箱",
                    en="Image & Mask Toolbox",
                ),
                plugins={
                    "matting": Matting,
                },
            ),
        )
    
class ImagesFollowers(IPluginGroup):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_group_styles,
            offsetX=-48,
            expandOffsetX=64,
            tooltip=I18N(
                zh="一组利用两张图片来进行融合的插件",
                en="A set of plugins that fuse two images",
            ),
            nodeConstraintRules=NodeConstraintRules(
                exactly=[NodeConstraints.IMAGE, NodeConstraints.IMAGE]
            ),
            pivot=PivotType.RT,
            follow=True,
            pluginInfo=IPluginGroupInfo(
                name=I18N(
                    zh="融合工具箱",
                    en="Fusing Toolbox",
                ),
                header=I18N(
                    zh="融合工具箱",
                    en="Fusing Toolbox",
                ),
                plugins={
                    "fusing": Fusing,
                },
            ),
        )

# uncomment this line to pre-load the models
# get_apis()
register_plugin("static")(StaticPlugins)
register_plugin("image_followers")(ImageFollowers)
register_plugin("images_followers")(ImagesFollowers)
register_plugin("image_and_mask_followers")(ImageAndMaskFollowers)
app = App()