import os
import json

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
        def callback(step: int, num_steps: int) -> bool:
            return self.send_progress(step / num_steps)
        
        print(data.extraData)
        pipe = get_model(data.extraData["version"])
        return await txt2img(pipe, data, callback)


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
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        def callback(step: int, num_steps: int) -> bool:
            return self.send_progress(step / num_steps)

        print(data.extraData)
        pipe = get_model(data.extraData["version"])
        return await img2img(pipe, data, callback)

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
                },
            ),
        )

# uncomment this line to pre-load the models
# get_apis()
register_plugin("static")(StaticPlugins)
register_plugin("image_followers")(ImageFollowers)
app = App()