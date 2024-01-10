import time, datetime
import yaml
import torch
import os
import uvicorn
import ctypes
import argparse
import tensorrt as trt
from fastapi import FastAPI

from pipelines.engine import TRT_LOGGER
from pipelines.trt_sdxl_base import SD_TRT
from generate import txt2img, img2img, upscale, inpaint
from utils import torch_gc, Inputdata, Outputdata, Inputdata_upscale, Inputdata_inpaint

app = FastAPI()


@app.post("/t2i")
async def sd_t2i(request: Inputdata) -> Outputdata:
    time1 = time.time()

    imgs_str = txt2img(sdbase, request)

    time2 = time.time()
    answer = Outputdata(
        imgs=imgs_str,
        time=round(time2-time1,8)
    )

    log = "#SD_t2i " + "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]"
    print(log)
    torch_gc()
    return answer

@app.post("/i2i")
async def sd_i2i(request: Inputdata) -> Outputdata:
    time1 = time.time()

    imgs_str = img2img(sdbase, request)

    time2 = time.time()
    answer = Outputdata(
        imgs=imgs_str,
        time=round(time2-time1,8)
    )

    log = "#SD_i2i " + "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]"
    print(log)
    torch_gc()
    return answer

@app.post("/upscale")
async def nn_upscale(request: Inputdata_upscale) -> Outputdata:
    time1 = time.time()

    imgs_str = upscale(request)

    time2 = time.time()
    answer = Outputdata(
        imgs=imgs_str,
        time=round(time2-time1,8)
    )

    log = "#NN_Upscale " + "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]"
    print(log)
    torch_gc()
    return answer

@app.post("/inpaint")
async def sd_inpaint(request: Inputdata_inpaint) -> Outputdata:
    time1 = time.time()

    imgs_str = inpaint(sdbase, request)

    time2 = time.time()
    answer = Outputdata(
        imgs=imgs_str,
        time=round(time2-time1,8)
    )

    log = "#SD_inpaint " + "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]"
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="API for Stable Diffusion (TensorRT)")
    parser.add_argument('--config', type=str, default="/work/Stable_Diffusion_GPU_Deploy/configs/sdxl.yaml")
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    ctypes.cdll.LoadLibrary(config['TRT_build']['static_plugin_sofile'])

    sdbase = SD_TRT(
            hf_dir=config['pipe_dir']['hf_dir'],
            engine_dir=config['pipe_dir']['onnx_opt_dir'],
            engine_config=config['TRT_build']['input_shapes']['unet'],
            enable_dynamic_shape=config['TRT_build']['enable_dynamic_shape'])
    
    sdbase.activateEngines()

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)

    sdbase.teardown()