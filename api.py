import apiconfig_local
from src.pipelines import SD_TRT
from src.pipelines.engine import TRT_LOGGER
from src.utils_trt import export_onnx, optimize_onnx, export_engine
from src.generate import *
from src.utils import *

import os
import time, datetime
import uvicorn
import ctypes
import argparse
import tensorrt as trt
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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

    # build SD trt_engines
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    ctypes.cdll.LoadLibrary(apiconfig_local.static_plugin_sofile)
    if not os.path.exists(apiconfig_local.save_dir):
        export_onnx(
            apiconfig_local.pipe_dir, 
            apiconfig_local.save_dir, 
            apiconfig_local.lora_dir, 
            apiconfig_local.control_dir, 
            apiconfig_local.opset, 
            torch.float16 if apiconfig_local.fp16 else torch.float32,
        )
        optimize_onnx(
            apiconfig_local.save_dir,
        )
        export_engine(
            apiconfig_local.save_dir,
            apiconfig_local.save_dir,
            apiconfig_local.fp16,
            apiconfig_local.dynamic_input_shapes,
        )

    # load VLM pipeline
    tokenizer = AutoTokenizer.from_pretrained(apiconfig_local.vlm_dir, trust_remote_code=True)
    vlm_config = AutoConfig.from_pretrained(apiconfig_local.vlm_dir, trust_remote_code=True)
    vlm_config.quantization_config["use_exllama"] = False
    vlmodel = AutoModelForCausalLM.from_pretrained(apiconfig_local.vlm_dir, config=vlm_config, device_map="cpu", trust_remote_code=True).eval()
    torch_gc()

    # load SD trt_pipeline
    sdbase = SD_TRT(
        pipe_dir=apiconfig_local.pipe_dir,
        engine_dir=apiconfig_local.save_dir,
        vae_dir=apiconfig_local.vae_dir,
        engine_config=apiconfig_local.dynamic_input_shapes,
        enable_dynamic_shape=True,
        use_cuda_graph=apiconfig_local.use_cuda_graph,
        lowvram=apiconfig_local.lowvram,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
    sdbase.teardown()