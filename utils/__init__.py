import torch
from .process_images import *
from .dataformat import *

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda:0"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def parser_controlnet(controls_list):
    types = []
    hints = []
    hints_skip = []
    hints_start = []
    strengths = []

    if controls_list:
        for ci in controls_list:
            types.append(ci["type"])
            hints.append(ci["hint_url"])
            hints_skip.append(ci["skip_annotator"])
            hints_start.append(ci["hint_start"])
            strengths.append(ci["control_strength"])
        
    return types, hints, hints_skip, hints_start, strengths

