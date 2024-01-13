import io
import base64
import requests
import numpy as np
from PIL import Image
from cftool.cv import ImageBox

def url2img(url):
    response = requests.get(url)
    if response.status_code == 200:
        img_data = io.BytesIO(response.content)
        img = Image.open(img_data)   
        return img
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
        return None

def img2str(imgs):
    if isinstance(imgs, Image.Image):
        imgs = [imgs]
    assert isinstance(imgs[0], Image.Image)

    imgs_str = []
    for img in imgs:
        image_buffer = io.BytesIO()
        img.save(image_buffer, format='png')
        base64_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
        imgs_str.append(base64_image)

    return imgs_str

def str2img(strs):
    if isinstance(strs, str):
        strs = [strs]
    assert isinstance(strs[0], str)

    imgs = []
    for string in strs:
        decoded_image = base64.b64decode(string)
        image = Image.open(io.BytesIO(decoded_image))
        imgs.append(image)

    return imgs

def adjust_lt_rb(lt_rb: ImageBox, w: int, h: int, paddingx: int, paddingy: int, tw: int, th: int) -> ImageBox:
    l, t, r, b = lt_rb.tuple
    l = max(0, l - paddingx)
    t = max(0, t - paddingy)
    r = min(w, r + paddingx)
    b = min(h, b + paddingy)
    cropped_h, cropped_w = b - t, r - l
    # adjust lt_rb to make the cropped aspect ratio equals to the th/tw
    if cropped_h / cropped_w > th / tw:
        dw = (int(cropped_h * tw / th) - cropped_w) // 2
        dh = 0
    else:
        dw = 0
        dh = (int(cropped_w * th / tw) - cropped_h) // 2
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

def crop_masked_area(image, mask, tw, th, padding_scale=0.1):
    """
    image: PIL.Image "RGB", uint8
    mask: PIL.Image "L", uint8
    """
    w, h = mask.size
    lt_rb = ImageBox.from_mask(np.array(mask), 0)
    lt_rb = adjust_lt_rb(lt_rb, w, h, int(padding_scale*w), int(padding_scale*h), tw, th)

    
    cropped_mask = mask.crop(lt_rb.tuple).resize((tw,th))
    if isinstance(image, list): 
        cropped_image = []
        for img in image:
            cropped_image.append(img.crop(lt_rb.tuple).resize((tw,th)))
    else:
        cropped_image = image.crop(lt_rb.tuple).resize((tw,th))
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

def get_angle(a,c,w,h):
    theta = np.arccos(a/w)
    if np.sin(theta)*c <= 0:
        theta = -theta
    return theta

def rotate_xy(x,y,xo,yo,theta):
    x_ = (x-xo)*np.cos(theta) - (y-yo)*np.sin(theta) + xo
    y_ = (x-xo)*np.sin(theta) + (y+yo)*np.cos(theta) + yo
    return x_, y_

def img_transform(img, data):
    if img.mode == "RGBA":
        alpha = img.getchannel("A")
        alpha = np.array(alpha, dtype=np.float32)/255.
        img_data = np.array(img, dtype=np.float32) * alpha[:,:,None]
        img = Image.fromarray(img_data.astype(np.uint8))
    theta = get_angle(data.transform.a, data.transform.c, data.w, data.h)
    img = img.resize(size=[int(data.w), int(data.h)]).rotate(np.degrees(theta), expand=True)
    return img

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