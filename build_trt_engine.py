
import argparse
import os
import shutil
import tensorrt as trt
import ctypes
import torch


from src.utils_trt import export_onnx, optimize_onnx, export_engine
import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", default=False)
    cmd_args = parser.parse_args()
    
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    if os.path.exists(config.static_plugin_sofile):
        ctypes.cdll.LoadLibrary(config.static_plugin_sofile)

    if not torch.cuda.is_available():
        raise ValueError("Only supported on GPUs with CUDA")

    if os.path.exists(config.save_dir):
        if cmd_args.overwrite:
            shutil.rmtree(config.save_dir)
        else:
            raise RuntimeError(f"output directory existed:{config['save_dir']}. Add --overwrite to empty the directory.")

    export_onnx(
        config.pipe_dir, 
        config.save_dir, 
        config.lora_dir, 
        config.control_dir, 
        config.opset, 
        torch.float16 if config.fp16 else torch.float32,
    )

    optimize_onnx(
        config.save_dir,
    )

    export_engine(
        config.save_dir,
        config.save_dir,
        config.fp16,
        config.dynamic_input_shapes,
    )



        
