import ctypes
import argparse
import yaml
import os
import shutil
import numpy as np
import tensorrt as trt
import torch
from pathlib import Path
from polygraphy.backend.trt import (
    CreateConfig,
    ModifyNetworkOutputs,
    Profile,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)


# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
}


def _export_engine(
    engine_path,
    onnx_path,
    fp16,
    input_shapes,
    enable_dynamic_shape=True,
    enable_refit=False,
    enable_all_tactics=False,
    timing_cache=None,
    update_output_names=None,
):

    print(f"Building TensorRT engine for {onnx_path}: {engine_path}")
    p = Profile()

    if os.path.exists(engine_path):
        os.remove(engine_path)

    for name, dims in input_shapes.items():
        if enable_dynamic_shape:
            p.add(name, min=dims[0], opt=dims[1], max=dims[2])
        else:
            p.add(name, min=dims[1], opt=dims[1], max=dims[1])

    config_kwargs = {}
    if not enable_all_tactics:
        config_kwargs["tactic_sources"] = []

    network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])

    if update_output_names:
        print(f"Updating network outputs to {update_output_names}")
        network = ModifyNetworkOutputs(network, update_output_names)
    engine = engine_from_network(
        network,
        config=CreateConfig(
            fp16=fp16, refittable=enable_refit, profiles=[p], load_timing_cache=timing_cache, **config_kwargs
        ),
        save_timing_cache=timing_cache,
    )
    save_engine(engine, path=engine_path)


def export_engine(
    save_dir,
    onnx_dir,
    fp16,
    dynamic_input_shapes,
):
    save_dir = Path(save_dir)
    onnx_dir = Path(onnx_dir)
    if not onnx_dir.exists():
        raise RuntimeError("{onnx_dir} directory is not existed!")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    for model_name, shapes in dynamic_input_shapes.items():
        onnx_path = onnx_dir/model_name/"model.opt.onnx"
        engine_path = save_dir/f"{model_name}.plan"
        
        _export_engine(
                    engine_path.as_posix(),
                    onnx_path.as_posix(),
                    fp16,
                    input_shapes=shapes,
                    enable_dynamic_shape=True,)
        
        print("TRT engine saved to", engine_path)

    