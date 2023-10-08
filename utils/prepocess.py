from cfdraw import *
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from cftool.cv import ImageBox

def png_to_mask(mask):
    mask = mask.getchannel("A")
    mask = np.array(mask)
    mask[mask >0] = 255
    mask = Image.fromarray(mask)
    return mask

def mask_to_box(mask):
    mask = np.array(mask)
    h, w = mask.shape
    mask_x = np.sum(mask, axis=0)
    mask_y = np.sum(mask, axis=1)
    x0, x1 = 0, w-1
    y0, y1 = 0, h-1
    while x0 < w:
        if mask_x[x0] != 0:
            break
        x0 += 1
    while x1 > -1:
        if mask_x[x1] == 0:
            break
        x1 -= 1
    while y0 < h:
        if mask_y[y0] != 0:
            break
        y0 += 1
    while y1 > -1:
        if mask_y[y1] == 0:
            break
        y1 -= 1
    return np.array([x0, y0, x1, y1])

def adjust_lt_rb(lt_rb: ImageBox, w: int, h: int, paddingx: int, paddingy: int) -> ImageBox:
    l, t, r, b = lt_rb.tuple
    l = max(0, l - paddingx)
    t = max(0, t - paddingy)
    r = min(w, r + paddingx)
    b = min(h, b + paddingy)
    cropped_h, cropped_w = b - t, r - l
    # adjust lt_rb to make the cropped aspect ratio equals to the original one
    if cropped_h / cropped_w > h / w:
        dw = (int(cropped_h * w / h) - cropped_w) // 2
        dh = 0
    else:
        dw = 0
        dh = (int(cropped_w * h / w) - cropped_h) // 2
    if dw > 0:
        if l < dw:
            l = 0
            r = min(w, cropped_w + dw * 2)
        elif r + dw > w:
            r = w
            l = max(0, w - cropped_w - dw * 2)
        else:
            l -= dw
            r += dw
    if dh > 0:
        if t < dh:
            t = 0
            b = min(h, cropped_h + dh * 2)
        elif b + dh > h:
            b = h
            t = max(0, h - cropped_h - dh * 2)
        else:
            t -= dh
            b += dh
    return ImageBox(l, t, r, b)

def crop_masked_area(image, mask, padding_scale=0.1):
    """
    image: PIL.Image "RGB", uint8
    mask: PIL.Image "L", uint8
    """
    w, h = image.size
    lt_rb = ImageBox.from_mask(np.array(mask), 0)
    lt_rb = adjust_lt_rb(lt_rb, w, h, int(padding_scale*w), int(padding_scale*h))

    cropped_image = image.crop(lt_rb.tuple).resize((w,h))
    cropped_mask = mask.crop(lt_rb.tuple).resize((w,h))
    return cropped_image, cropped_mask, lt_rb


def recover_cropped_image(gen_images, orig_image, lt_rb):
    new = []
    l, t, r, b = lt_rb.tuple
    bw, bh = r-l, b-t
    for img in gen_images:
        img = img.resize((bw, bh))
        canvas = orig_image.copy()
        canvas.paste(img, (l,t))
        new.append(canvas)
    return new


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def get_angle(a,c,w,h):
    theta = np.arccos(a/w)
    if np.sin(theta)*c <= 0:
        theta = -theta
    return theta

def ExpandEdge(mask, strength=1):
    mask = mask/255.
    h, w = mask.shape
    kernel_size = 3+2*strength
    mask = torch.from_numpy(mask).to("cuda").float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1,1,kernel_size,kernel_size), device="cuda")
    x = F.conv2d(mask, kernel, padding=kernel_size//2)
    res = ((x>0) & (x<(kernel_size**2)))
    res = res.cpu().numpy().reshape(h, w)
    return res

def img_transform(img, data):
    theta = get_angle(data.transform.a, data.transform.c, data.w, data.h)
    img = img.resize(size=[int(data.w), int(data.h)]).rotate(np.degrees(theta), expand=True)
    return img
