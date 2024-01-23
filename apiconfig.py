pipe_dir = "stabilityai/stable-diffusion-xl-base-1.0"
save_dir = "/work/CKPTS/Exports/sdxl"
lora_dir = None
vae_dir = "madebyollin/sdxl-vae-fp16-fix"
control_dir = {
  "control_canny": "diffusers/controlnet-canny-sdxl-1.0",
  "control_depth": "diffusers/controlnet-depth-sdxl-1.0"
}
vlm_dir = "Qwen/Qwen-VL-Chat-Int4"

lowvram = True
opset = 17
fp16 = True
static_plugin_sofile = "src/utils_trt/plugins/build/libplugin.so"

batch_size = [2, 2, 8]
height = [512, 1024, 1024]
width = [512, 1024, 1024]
use_cuda_graph = False

dynamic_input_shapes = {
  "control_canny": {
    "sample": [(bs, 4, h//8, w//8) for (bs, h, w) in zip(batch_size, height, width)],
    "timestep": [[1],[1],[1]],
    "encoder_hidden_states": [(bs, 77, 2048) for bs in batch_size],
    "add_text_embeds": [(bs, 1280) for bs in batch_size],
    "add_time_ids": [(bs, 6) for bs in batch_size],
    "controlnet_cond": [(bs, 3, h, w) for (bs, h, w) in zip(batch_size, height, width)],
    "conditioning_scale": [[1],[1],[1]],
  },

  "control_depth": {
    "sample": [(bs, 4, h//8, w//8) for (bs, h, w) in zip(batch_size, height, width)],
    "timestep": [[1],[1],[1]],
    "encoder_hidden_states": [(bs, 77, 2048) for bs in batch_size],
    "add_text_embeds": [(bs, 1280) for bs in batch_size],
    "add_time_ids": [(bs, 6) for bs in batch_size],
    "controlnet_cond": [(bs, 3, h, w) for (bs, h, w) in zip(batch_size, height, width)],
    "conditioning_scale": [[1],[1],[1]],
  },


  "unet_encoder": {
    "sample": [(bs, 4, h//8, w//8) for (bs, h, w) in zip(batch_size, height, width)],
    "timestep": [[1],[1],[1]],
    "encoder_hidden_states": [(bs, 77, 2048) for bs in batch_size],
    "add_text_embeds": [(bs, 1280) for bs in batch_size],
    "add_time_ids": [(bs, 6) for bs in batch_size],
  },

  "unet_decoder": {
    "encoder_hidden_states": [(bs, 77, 2048) for bs in batch_size],
    "down00": [(bs, 320, h//8, w//8) for (bs, h, w) in zip(batch_size, height, width)],
    "down01": [(bs, 320, h//8, w//8) for (bs, h, w) in zip(batch_size, height, width)],
    "down02": [(bs, 320, h//8, w//8) for (bs, h, w) in zip(batch_size, height, width)],
    "down10": [(bs, 320, h//16, w//16) for (bs, h, w) in zip(batch_size, height, width)],
    "down11": [(bs, 640, h//16, w//16) for (bs, h, w) in zip(batch_size, height, width)],
    "down12": [(bs, 640, h//16, w//16) for (bs, h, w) in zip(batch_size, height, width)],
    "down20": [(bs, 640, h//32, w//32) for (bs, h, w) in zip(batch_size, height, width)],
    "down21": [(bs, 1280, h//32, w//32) for (bs, h, w) in zip(batch_size, height, width)],
    "down22": [(bs, 1280, h//32, w//32) for (bs, h, w) in zip(batch_size, height, width)],
    "mid": [(bs, 1280, h//32, w//32) for (bs, h, w) in zip(batch_size, height, width)],
    "emb": [(bs, 1280) for bs in batch_size],
  },

}