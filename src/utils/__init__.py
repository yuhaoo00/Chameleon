import torch
from .process_images import *
from .dataformat import *
from .fields import *

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda:0"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def parser_controlnet(controls_list):
    types = []
    hints = []
    hints_start = []
    hints_end = []
    strengths = []

    if controls_list:
        for ci in controls_list:
            types.append(ci["type"])
            hints.append(ci["hint_url"])
            hints_start.append(ci["hint_start"])
            hints_end.append(ci["hint_end"])
            strengths.append(ci["control_strength"])
        
    return types, hints, hints_start, hints_end, strengths

def parser_INodeData(nodedata):
    if 'response' in nodedata.meta['data']: 
        img_url = nodedata.meta['data']['response']['value']['url']
    else:
        img_url = nodedata.meta['data']['url']
    
    info = {
        "img_url": img_url,
        "z": nodedata.z,
        "x": nodedata.x,
        "y": nodedata.y,
        "w": nodedata.w,
        "h": nodedata.h,
        "transform": [nodedata.transform.a,nodedata.transform.b,nodedata.transform.c,
                      nodedata.transform.d,nodedata.transform.e,nodedata.transform.f,],
        "text": nodedata.text,
    }
    return info