import torch
from .process_images import *
from .dataformat import *

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda:0"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

