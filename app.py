import sys
sys.path.append("/mnt/Data/CodeML/SD/Chameleon")

import numpy as np
import cv2
from PIL import Image
from typing import List
from cfdraw import *
from utils.generate import *
from utils.load import *
from utils.prepocess import *
from fields import *
from extensions.annotators.hed import HEDdetector
from extensions.annotators.zoe import ZoeDetector
from extensions.annotators.canny import CannyDetector
# plugins

class Canny(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=480,
            h=250,
            src=constants.IMAGE_TO_IMAGE_ICON,
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
        img = img_transform(img, data.nodeData).convert("RGB")

        model = CannyDetector()
        res = model(img, data.extraData["low_threshold"], data.extraData["high_threshold"])
        return [res]

class Zoe(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=240,
            h=110,
            src=constants.IMAGE_TO_IMAGE_ICON,
            tooltip=I18N(
                zh="获取Zoe图",
                en="Get Zoe",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="获取Zoe图",
                    en="Get Zoe",
                ),
                definitions={},
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        img = await self.load_image(data.nodeData.src)
        img = img_transform(img, data.nodeData).convert("RGB")

        model = ZoeDetector()
        res = model(img)
        return [res]

class Hed(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=240,
            h=110,
            src=constants.IMAGE_TO_IMAGE_ICON,
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
        img = img_transform(img, data.nodeData).convert("RGB")

        model = HEDdetector()
        res = model(img)
        return [res]

class Txt2Img(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=0.75,
            h=0.4,
            maxW=800,
            minH=520,
            useModal=True,
            src=constants.TEXT_TO_IMAGE_ICON,
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
        def callback(step: int, *args) -> bool:
            return self.send_progress(step / data.extraData["num_steps"])
        
        pipe = get_sd_t2i(data.extraData["version"])
        return txt2img(pipe, data, callback)


class Img2Img(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=constants.IMAGE_TO_IMAGE_ICON,
            tooltip=I18N(
                zh="图生图",
                en="Image to Image",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="图生图",
                    en="Image to Image",
                ),
                definitions=img2img_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        def callback(step: int, *args) -> bool:
            return self.send_progress(step / data.extraData["num_steps"])
        
        img = await self.load_image(data.nodeData.src)
        img = img_transform(img, data.nodeData)

        pipe = get_sd_i2i(data.extraData["version"])
        return img2img(pipe, img, data, callback)
    
class Tile(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=constants.IMAGE_TO_IMAGE_ICON,
            tooltip=I18N(
                zh="重绘细节",
                en="Repaint Details (Tile)",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="重绘细节",
                    en="Repaint Details (Tile)",
                ),
                definitions=tile_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        def callback(step: int, *args) -> bool:
            return self.send_progress(step / data.extraData["num_steps"])
        
        img = await self.load_image(data.nodeData.src)
        img = img_transform(img, data.nodeData)

        pipe = get_controlnet("v11_sd15_tile")
        return cn_tile(pipe, img, data, callback)
    
class Inpainting(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=constants.SD_INPAINTING_ICON,
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
        def callback(step: int, *args) -> bool:
            return self.send_progress(step / data.extraData["num_steps"])

        url_node = self.filter(data.nodeDataList, SingleNodeType.IMAGE)[0]
        mask_node = self.filter(data.nodeDataList, SingleNodeType.PATH)[0]
        img = await self.load_image(url_node.src)
        img = img_transform(img, url_node)
        mask = await self.load_image(mask_node.src)

        pipe = get_sd_inpaint(data.extraData["version"])
        return inpaint(pipe, img, mask, data, callback)
    
class CNInpainting(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=constants.SD_INPAINTING_ICON,
            tooltip=I18N(
                zh="局部替换 (ControlNet)",
                en="Inpainting (ControlNet)",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="局部替换 (ControlNet)",
                    en="Inpainting (ControlNet)",
                ),
                definitions=cn_inpainting_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        def callback(step: int, *args) -> bool:
            return self.send_progress(step / data.extraData["num_steps"])

        url_node = self.filter(data.nodeDataList, SingleNodeType.IMAGE)[0]
        mask_node = self.filter(data.nodeDataList, SingleNodeType.PATH)[0]
        img = await self.load_image(url_node.src)
        img = img_transform(img, url_node)
        mask = await self.load_image(mask_node.src)

        pipe = get_controlnet("v11_sd15_inapint")
        return cn_inpaint(pipe, img, mask, data, callback)


class StyleTransfer(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=constants.IMAGE_TO_IMAGE_ICON,
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
                definitions=matting_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:

        url_node = self.filter(data.nodeDataList, SingleNodeType.IMAGE)[0]
        mask_node = self.filter(data.nodeDataList, SingleNodeType.PATH)[0]
        img = await self.load_image(url_node.src)
        img = img_transform(img, url_node).convert("RGB")
        mask = await self.load_image(mask_node.src)
        mask = mask.convert("L")
        box = mask_to_box(mask)
        
        model = get_mSAM(data.extraData["version"])
        model.set_image(np.array(img))
        mask_fined = model.predict(box=box,
                                    #mask_input=mask[None,:,:],
                                    #point_coords=np.array([[463, 455]]),
                                    #point_labels=np.array([1]),
                                    multimask_output=False)[0][0,:]
        
        img_masked = np.concatenate((np.array(img), (mask_fined[:,:,None]*255).astype(np.uint8)), axis=2)
        img_masked = Image.fromarray(img_masked)

        return [img_masked]
    
class EasyFusing(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            w=240,
            h=110,
            src=constants.SOD_ICON,
            tooltip=I18N(
                zh="直接融合",
                en="Easy Fusing",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="直接融合",
                    en="Easy Fusing",
                ),
                definitions={},
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        url_nodes = self.filter(data.nodeDataList, SingleNodeType.IMAGE)
        data0 = url_nodes[0]
        data1 = url_nodes[1]
        if 'response' in data0.meta['data']: 
            img0 = await self.load_image(data0.meta['data']['response']['value']['url'])
        else:
            img0 = await self.load_image(data0.meta['data']['url'])
        if 'response' in data1.meta['data']: 
            img1 = await self.load_image(data1.meta['data']['response']['value']['url'])
        else:
            img1 = await self.load_image(data1.meta['data']['url'])

        return easy_fusing(data0, data1, img0, img1)[0]
    
class EdgeFusing(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=constants.SOD_ICON,
            tooltip=I18N(
                zh="边缘融合",
                en="Edge Fusing",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="边缘融合",
                    en="Edge Fusing",
                ),
                definitions=inpainting_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        def callback(step: int, *args) -> bool:
            return self.send_progress(step / data.extraData["num_steps"])
        url_nodes = self.filter(data.nodeDataList, SingleNodeType.IMAGE)
        data0 = url_nodes[0]
        data1 = url_nodes[1]
        if 'response' in data0.meta['data']: 
            img0 = await self.load_image(data0.meta['data']['response']['value']['url'])
        else:
            img0 = await self.load_image(data0.meta['data']['url'])
        if 'response' in data1.meta['data']: 
            img1 = await self.load_image(data1.meta['data']['response']['value']['url'])
        else:
            img1 = await self.load_image(data1.meta['data']['url'])

        pipe = get_sd_inpaint(data.extraData["version"])
        return edge_fusing(pipe, data, data0, data1, img0, img1, callback)
    
class SmartFusing(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=constants.SOD_ICON,
            tooltip=I18N(
                zh="智能融合",
                en="Smart Fusing",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="智能融合",
                    en="Smart Fusing",
                ),
                definitions=smart_fusing_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        def callback(step: int, *args) -> bool:
            return self.send_progress(step / data.extraData["num_steps"])
        url_nodes = self.filter(data.nodeDataList, SingleNodeType.IMAGE)
        data0 = url_nodes[0]
        data1 = url_nodes[1]
        if 'response' in data0.meta['data']: 
            img0 = await self.load_image(data0.meta['data']['response']['value']['url'])
        else:
            img0 = await self.load_image(data0.meta['data']['url'])
        if 'response' in data1.meta['data']: 
            img1 = await self.load_image(data1.meta['data']['response']['value']['url'])
        else:
            img1 = await self.load_image(data1.meta['data']['url'])

        pipe = get_style_inpaint(data.extraData["version"], data.extraData["cn_type"])
        return smart_fusing(pipe, data, data0, data1, img0, img1, callback)


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
                    "tile": Tile,
                    "canny": Canny,
                    "hed": Hed,
                    "zoe": Zoe,
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
                    "cninpainting": CNInpainting,
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
                    "easyfusing": EasyFusing,
                    "edgefusing": EdgeFusing,
                    "smartfusing": SmartFusing,
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