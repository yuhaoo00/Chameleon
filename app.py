import sys
sys.path.append("/mnt/Data/CodeML/SD/Chameleon")
import requests
import numpy as np
from PIL import Image
from typing import List
from cfdraw import *
from src.utils import img_transform, str2img, img2str, png_to_mask, parser_controlnet, parser_INodeData, get_angle, url2img
from src.fields import *
import src.icons as paths
# plugins

class Upscale(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=480,
            h=250,
            src=paths.SR_ICON,
            tooltip=I18N(
                zh="超分辨率",
                en="Super Resolution",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="超分辨率",
                    en="Super Resolution",
                ),
                definitions=sr_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        img = await self.load_image(data.nodeData.src)
        img = img_transform(img, data.nodeData)

        data_to_send = {
            "image": img2str(img)[0],
            "tag": data.extraData["version"],
            "scale": data.extraData["scale"],
        }

        response = requests.post('http://0.0.0.0:8000/upscale', json=data_to_send).json()
        imgs = str2img(response["imgs"])
        return imgs


class Txt2Img(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=0.75,
            h=0.4,
            maxW=800,
            minH=520,
            useModal=True,
            src=paths.TEXT_TO_IMAGE_ICON,
            tooltip=I18N(
                zh="文生图",
                en="Text to Image",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="文生图",
                    en="Text to Image",
                ),
                numColumns=2,
                definitions=txt2img_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        control, control_hint, control_hint_start, control_hint_end, control_strength = parser_controlnet(data.extraData["control"])

        data_to_send = {
            "text": data.extraData["text"],
            "num_steps": data.extraData["num_steps"],
            "guidance_scale": data.extraData["guidance_scale"],
            "seed": data.extraData["seed"],
            "negative_prompt": data.extraData["negative_prompt"],
            "h": data.extraData["h"],
            "w": data.extraData["w"],
            "num_samples": data.extraData["num_samples"],
            "use_hrfix": data.extraData["use_highres"],
            "hrfix_scale": data.extraData["highres_scale"] if data.extraData["use_highres"] else None,
            "hrfix_strength": data.extraData["highres_strength"] if data.extraData["use_highres"] else None,
            "control": control,
            "control_hint": control_hint,
            "control_hint_start": control_hint_start,
            "control_hint_end": control_hint_end,
            "control_strength": control_strength,
        }

        response = requests.post('http://0.0.0.0:8000/t2i', json=data_to_send).json()
        imgs = str2img(response["imgs"])
        return imgs


class Img2Img(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=0.75,
            h=0.4,
            maxW=800,
            minH=520,
            useModal=True,
            src=paths.IMAGE_TO_IMAGE_ICON,
            tooltip=I18N(
                zh="图生图",
                en="Image to Image",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="图生图",
                    en="Image to Image",
                ),
                numColumns=2,
                definitions=img2img_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        img = await self.load_image(data.nodeData.src)
        img = img_transform(img, data.nodeData)
        control, control_hint, control_hint_skip, control_hint_start, control_strength = parser_controlnet(data.extraData["control"])

        data_to_send = {
            "text": data.extraData["text"],
            "image": img2str(img)[0],
            "strength": data.extraData["strength"],
            "num_steps": data.extraData["num_steps"],
            "guidance_scale": data.extraData["guidance_scale"],
            "seed": data.extraData["seed"],
            "negative_prompt": data.extraData["negative_prompt"],
            "num_samples": data.extraData["num_samples"],
            "use_hrfix": data.extraData["use_highres"],
            "hrfix_scale": data.extraData["highres_scale"] if data.extraData["use_highres"] else None,
            "hrfix_strength": data.extraData["highres_strength"] if data.extraData["use_highres"] else None,
            "control": control,
            "control_hint": control_hint,
            "control_hint_skip": control_hint_skip,
            "control_hint_start": control_hint_start,
            "control_strength": control_strength,
        }

        response = requests.post('http://0.0.0.0:8000/i2i', json=data_to_send).json()
        imgs = str2img(response["imgs"])
        return imgs
    
class DemoFusion(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=paths.DETAIL_ICON,
            tooltip=I18N(
                zh="重绘细节",
                en="DemoFusion",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="重绘细节",
                    en="DemoFusion",
                ),
                definitions=demofusion_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        img = await self.load_image(data.nodeData.src)
        img = img_transform(img, data.nodeData)

        data_to_send = {
            "text": data.extraData["text"],
            "image": img2str(img)[0],
            "num_steps": data.extraData["num_steps"],
            "guidance_scale": data.extraData["guidance_scale"],
            "seed": data.extraData["seed"],
            "negative_prompt": data.extraData["negative_prompt"],
            "h": data.extraData["h"],
            "w": data.extraData["w"],
            "view_batch_size": data.extraData["view_batch_size"],
            "stride": data.extraData["stride"],
            "multi_decoder": data.extraData["multi_decoder"],
            "cosine_scale_1": data.extraData["cosine_scale_1"],
            "cosine_scale_2": data.extraData["cosine_scale_2"],
            "cosine_scale_3": data.extraData["cosine_scale_3"],
            "sigma": data.extraData["sigma"],
            "lowvram": data.extraData["lowvram"],
        }

        response = requests.post('http://0.0.0.0:8000/demofusion', json=data_to_send).json()
        imgs = str2img(response["imgs"])
        return imgs
    
class Inpainting(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=paths.INPAINT_ICON,
            tooltip=I18N(
                zh="局部重绘",
                en="Inpainting",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="局部重绘",
                    en="Inpainting",
                ),
                definitions=inpainting_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        url_node = self.filter(data.nodeDataList, SingleNodeType.IMAGE)[0]
        mask_node = self.filter(data.nodeDataList, SingleNodeType.PATH)[0]
        img = await self.load_image(url_node.src)
        img = img_transform(img, url_node)
        mask = await self.load_image(mask_node.src)
        mask = png_to_mask(mask)

        data_to_send = {
            "text": data.extraData["text"],
            "image": img2str(img)[0],
            "mask": img2str(mask)[0],
            "strength": data.extraData["strength"],
            "num_steps": data.extraData["num_steps"],
            "guidance_scale": data.extraData["guidance_scale"],
            "seed": data.extraData["seed"],
            "negative_prompt": data.extraData["negative_prompt"],
            "num_samples": data.extraData["num_samples"],
            "h": data.extraData["h"],
            "w": data.extraData["w"],
            "focus_mode": data.extraData["focus_mode"]
        }

        response = requests.post('http://0.0.0.0:8000/inpaint', json=data_to_send).json()
        imgs = str2img(response["imgs"])
        return imgs
    
class StyleTransfer(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=paths.STYLE_ICON,
            tooltip=I18N(
                zh="风格迁移",
                en="Style Transfer"
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="风格迁移",
                    en="Style Transfer",
                ),
                definitions=st_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        def callback(step: int, *args) -> bool:
            return self.send_progress((step+1) / data.extraData["num_steps"])
        return 
        
        img = await self.load_image(data.nodeData.src)
        img = img_transform(img, data.nodeData)

        pipe = get_ipadapter(data.extraData["version"])
        return style_transfer(pipe, img, data, callback)


class Matting(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=240,
            h=180,
            src=paths.SOD_ICON,
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
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        url_node = self.filter(data.nodeDataList, SingleNodeType.IMAGE)[0]
        mask_node = self.filter(data.nodeDataList, SingleNodeType.PATH)[0]
        img = await self.load_image(url_node.src)
        img = img_transform(img, url_node).convert("RGB")
        mask = await self.load_image(mask_node.src)
        mask = mask.convert("L")
        
        data_to_send = {
            "image": img2str(img)[0],
            "mask": img2str(mask)[0],
        }

        response = requests.post('http://0.0.0.0:8000/matting', json=data_to_send).json()
        imgs = str2img(response["imgs"])
        return imgs

    
class EasyFusion(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=240,
            h=110,
            src=paths.FUSION_ICON,
            tooltip=I18N(
                zh="直接融合",
                en="Easy Fusion",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="直接融合",
                    en="Easy Fusion",
                ),
                definitions={},
            ),
        )

    async def process(self, data: ISocketRequest) -> Image.Image:
        url_nodes = self.filter(data.nodeDataList, SingleNodeType.IMAGE)

        info0 = parser_INodeData(url_nodes[0])
        info1 = parser_INodeData(url_nodes[1])

        data_to_send = {
            "info0": info0,
            "info1": info1,
        }
        
        response = requests.post('http://0.0.0.0:8000/fusion', json=data_to_send).json()
        imgs = str2img(response["imgs"])
        return imgs
    
class StyleFusion(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=paths.FUSION_PLUS_ICON,
            tooltip=I18N(
                zh="风格融合",
                en="Style Fusion",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="风格融合",
                    en="Style Fusion",
                ),
                definitions=sfusion_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        url_nodes = self.filter(data.nodeDataList, SingleNodeType.IMAGE)

        info0 = parser_INodeData(url_nodes[0])
        info1 = parser_INodeData(url_nodes[1])

        data_to_send = {
            "info0": info0,
            "info1": info1,
            "text": data.extraData["text"],
            "strength": data.extraData["strength"],
            "num_steps": data.extraData["num_steps"],
            "guidance_scale": data.extraData["guidance_scale"],
            "seed": data.extraData["seed"],
            "negative_prompt": data.extraData["negative_prompt"],
            "num_samples": data.extraData["num_samples"],
            "h": data.extraData["h"],
            "w": data.extraData["w"],
            "pad_strength": data.extraData["pad_strength"],
            "blur_strength": data.extraData["blur_strength"],
        }
        
        response = requests.post('http://0.0.0.0:8000/fusion_plus', json=data_to_send).json()
        imgs = str2img(response["imgs"])
        return imgs


class Canny(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=480,
            h=250,
            src=paths.EDGE_ICON,
            tooltip=I18N(
                zh="获取Canny图",
                en="Get Canny",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="获取Canny图",
                    en="Get Canny",
                ),
                definitions=canny_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        img = await self.load_image(data.nodeData.src)
        img = img_transform(img, data.nodeData)

        data_to_send = {
            "type": "canny",
            "image": img2str(img)[0],
            "low_threshold": data.extraData["low_threshold"],
            "high_threshold": data.extraData["high_threshold"],
        }

        response = requests.post('http://0.0.0.0:8000/get_hint', json=data_to_send).json()
        imgs = str2img(response["imgs"])
        return imgs


class Depth(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=240,
            h=110,
            src=paths.DEPTH_ICON,
            tooltip=I18N(
                zh="获取深度图",
                en="Get Depth",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="获取深度图",
                    en="Get Depth",
                ),
                definitions={},
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        img = await self.load_image(data.nodeData.src)
        img = img_transform(img, data.nodeData)

        data_to_send = {
            "type": "depth",
            "image": img2str(img)[0],
        }

        response = requests.post('http://0.0.0.0:8000/get_hint', json=data_to_send).json()
        imgs = str2img(response["imgs"])
        return imgs


class Hed(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=240,
            h=110,
            src=paths.SOFTEDGE_ICON,
            tooltip=I18N(
                zh="获取HED图",
                en="Get HED",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="获取HED图",
                    en="Get HED",
                ),
                definitions={},
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        img = await self.load_image(data.nodeData.src)
        img = img_transform(img, data.nodeData)

        data_to_send = {
            "type": "hed",
            "image": img2str(img)[0],
        }

        response = requests.post('http://0.0.0.0:8000/get_hint', json=data_to_send).json()
        imgs = str2img(response["imgs"])
        return imgs

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
            h=220,
            tooltip=I18N(
                zh="AIGC工具箱",
                en="AIGC Toolbox",
            ),
            nodeConstraint=NodeConstraints.IMAGE,
            pivot=PivotType.RT,
            follow=True,
            pluginInfo=IPluginGroupInfo(
                name=I18N(
                    zh="AIGC工具箱",
                    en="AIGC Toolbox",
                ),
                header=I18N(
                    zh="AIGC工具箱",
                    en="AIGC Toolbox",
                ),
                plugins={
                    "img2img": Img2Img,
                    "styletransfer": StyleTransfer,
                    "demofusion": DemoFusion,
                    "canny": Canny,
                    "hed": Hed,
                    "depth": Depth,
                    "upscale": Upscale,
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
                    en="Mask Toolbox",
                ),
                header=I18N(
                    zh="蒙版工具箱",
                    en="Mask Toolbox",
                ),
                plugins={
                    "matting": Matting,
                    "inpainting": Inpainting,
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
                    en="Fusion Toolbox",
                ),
                header=I18N(
                    zh="融合工具箱",
                    en="Fusion Toolbox",
                ),
                plugins={
                    "easyfusion": EasyFusion,
                    "stylefusion": StyleFusion,
                },
            ),
        )

# uncomment this line to pre-load the models
# get_apis()
register_plugin("static")(Txt2Img)
register_plugin("image_followers")(ImageFollowers)
register_plugin("images_followers")(ImagesFollowers)
register_plugin("image_and_mask_followers")(ImageAndMaskFollowers)
app = App()