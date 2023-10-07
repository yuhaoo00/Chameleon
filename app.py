import numpy as np
from PIL import Image
from typing import List
from cfdraw import *
from utils.generate import *
from utils.load import *
from utils.prepocess import *
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
        def callback(step: int, *args) -> bool:
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
        def callback(step: int, *args) -> bool:
            return self.send_progress((step+1) / data.extraData["num_steps"])
        
        img = await self.load_image(data.nodeData.src)
        pipe = get_sd_i2i(data.extraData["version"])
        return img2img(pipe, img, data, callback)
    
class Inpainting(IFieldsPlugin):
    @property
    def settings(self) -> IPluginSettings:
        return IPluginSettings(
            **common_styles,
            src=constants.SD_INPAINTING_ICON,
            tooltip=I18N(
                zh="在蒙版区域内填充符合描述的内容",
                en="Replace the masked area and fill it with the description",
            ),
            pluginInfo=IFieldsPluginInfo(
                header=I18N(
                    zh="局部替换",
                    en="Erase & Replace",
                ),
                numColumns=2,
                definitions=inpainting_fields,
            ),
        )

    async def process(self, data: ISocketRequest) -> List[Image.Image]:
        def callback(step: int, *args) -> bool:
            return self.send_progress(step / data.extraData["num_steps"])

        url_node = self.filter(data.nodeDataList, SingleNodeType.IMAGE)[0]
        mask_node = self.filter(data.nodeDataList, SingleNodeType.PATH)[0]
        img = await self.load_image(url_node.src)
        img = img.convert("RGB")
        mask = await self.load_image(mask_node.src)
        mask = transform_mask(mask)

        pipe = get_sd_inpaint(data.extraData["version"])
        return inpaint(pipe, img, mask, data, callback)


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
        def callback(step: int, *args) -> bool:
            return self.send_progress((step+1) / data.extraData["num_steps"])
        
        img = await self.load_image(data.nodeData.src)
        pipe = get_ipadapter(data.extraData["version"])
        return style_transfer(pipe, img, data, callback)


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
                #exportFullImages=True,
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
                    en="Fusing Toolbox",
                ),
                header=I18N(
                    zh="融合工具箱",
                    en="Fusing Toolbox",
                ),
                plugins={
                    "easyfusing": EasyFusing,
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