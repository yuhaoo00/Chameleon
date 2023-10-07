from cfdraw import *
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

def transform_mask(mask):
    mask = mask.getchannel("A").convert("RGB")
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

def easy_fusing(data0, data1, img0, img1):
    img0 = img0.convert("RGBA")
    img1 = img1.convert("RGBA")
    if data0.z > data1.z:
        data0, data1 = data1, data0
        img0, img1 = img1, img0

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
