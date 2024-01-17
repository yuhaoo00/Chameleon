import time, datetime
import yaml
import torch
import os
import uvicorn
import ctypes
import argparse
import tensorrt as trt
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.pipelines import SD_TRT
from src.pipelines.engine import TRT_LOGGER
from src.generate import *
from src.utils import *

app = FastAPI()


@app.post("/t2i")
async def t2i(request: Inputdata) -> Outputdata:
    time1 = time.time()

    imgs_str = sd_t2i(sdbase, request)

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
async def i2i(request: Inputdata) -> Outputdata:
    time1 = time.time()

    imgs_str = sd_i2i(sdbase, request)

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
async def upscale(request: Inputdata_upscale) -> Outputdata:
    time1 = time.time()

    imgs_str = nn_upscale(request)

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
async def inpaint(request: Inputdata_inpaint) -> Outputdata:
    time1 = time.time()

    imgs_str = sd_inpaint(sdbase, request)

    time2 = time.time()
    answer = Outputdata(
        imgs=imgs_str,
        time=round(time2-time1,8)
    )

    log = "#SD_inpaint " + "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]"
    print(log)
    torch_gc()
    return answer

@app.post("/demofusion")
async def demofusion(request: Inputdata_demofusion) -> Outputdata:
    time1 = time.time()

    imgs_str = sd_demofusion(sdbase, request)

    time2 = time.time()
    answer = Outputdata(
        imgs=imgs_str,
        time=round(time2-time1,8)
    )

    log = "#SD_DemoFusion " + "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]"
    print(log)
    torch_gc()
    return answer

@app.post("/get_hint")
async def get_hint(request: Inputdata_anno) -> Outputdata:
    time1 = time.time()

    imgs_str = annotating(request)

    time2 = time.time()
    answer = Outputdata(
        imgs=imgs_str,
        time=round(time2-time1,8)
    )

    log = f"#Get_{request.type} " + "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]"
    print(log)
    torch_gc()
    return answer

@app.post("/matting")
async def matting(request: Inputdata_matting) -> Outputdata:
    time1 = time.time()

    imgs_str = sam_matting(request)

    time2 = time.time()
    answer = Outputdata(
        imgs=imgs_str,
        time=round(time2-time1,8)
    )

    log = f"#SAM_Matting " + "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]"
    print(log)
    torch_gc()
    return answer


@app.post("/fusion")
async def fusion(request: Inputdata_fusing) -> Outputdata:
    time1 = time.time()

    imgs_str = easy_fusion(request)
    
    time2 = time.time()
    answer = Outputdata(
        imgs=imgs_str,
        time=round(time2-time1,8)
    )

    log = f"#Easy Fusion " + "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]"
    print(log)
    torch_gc()
    return answer


@app.post("/fusion_plus")
async def fusion_plus(request: Inputdata_fusing_plus) -> Outputdata:
    time1 = time.time()

    imgs_str, caption = style_fusion(sdbase, tokenizer, vlmodel, request)
    
    time2 = time.time()
    answer = Outputdata(
        imgs=imgs_str,
        time=round(time2-time1,8)
    )
    print(f"#Style Fusion (caption: \"{caption}\")")

    log = f"#Style Fusion " + "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]"
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="API for Stable Diffusion (TensorRT)")
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    import config

    tokenizer = AutoTokenizer.from_pretrained(config.vlm_dir, trust_remote_code=True)
    vlm_config = AutoConfig.from_pretrained(config.vlm_dir, trust_remote_code=True)
    vlm_config.quantization_config["use_exllama"] = False
    vlmodel = AutoModelForCausalLM.from_pretrained(config.vlm_dir, config=vlm_config, device_map="cpu", trust_remote_code=True).eval()
    torch_gc()

    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    ctypes.cdll.LoadLibrary(config.static_plugin_sofile)
    sdbase = SD_TRT(
        pipe_dir=config.pipe_dir,
        engine_dir=config.save_dir,
        vae_dir=config.vae_dir,
        engine_config=config.dynamic_input_shapes,
        enable_dynamic_shape=True,
        use_cuda_graph=config.use_cuda_graph,
        lowvram=config.lowvram,
    )

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)

    sdbase.teardown()