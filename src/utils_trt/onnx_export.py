
import shutil
from pathlib import Path
import onnx
import torch
from typing import Dict, Optional
from packaging import version
from diffusers import AutoPipelineForText2Image, ControlNetModel
is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")

class FrozenUnetEncoder(torch.nn.Module):
    def __init__(self, UNet2DConditionModel):
        super().__init__()
        assert UNet2DConditionModel.__class__.__name__ == 'UNet2DConditionModel'
        self.net = UNet2DConditionModel
    
    def forward(self, sample, timestep, encoder_hidden_states, add_text_embeds, add_time_ids,):
        # 0. center input if necessary
        if self.net.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timestep = timestep.expand(sample.shape[0])
        t_emb = self.net.time_proj(timestep)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.net.time_embedding(t_emb, None)

        # SDXL-specify
        time_embeds = self.net.add_time_proj(add_time_ids.flatten())
        time_embeds = time_embeds.reshape((add_text_embeds.shape[0], -1))
        add_embeds = torch.concat([add_text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.net.add_embedding(add_embeds)
        emb = emb + aug_emb 

        if self.net.time_embed_act is not None:
            emb = self.net.time_embed_act(emb)

        # 2. pre-process
        sample = self.net.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.net.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.net.mid_block is not None:
            if hasattr(self.net.mid_block, "has_cross_attention") and self.net.mid_block.has_cross_attention:
                sample = self.net.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = self.net.mid_block(sample, emb)


        return down_block_res_samples + (sample,emb,)
    
class FrozenUnetDecoder(torch.nn.Module):
    def __init__(self, UNet2DConditionModel):
        super().__init__()
        assert UNet2DConditionModel.__class__.__name__ == 'UNet2DConditionModel'
        self.net = UNet2DConditionModel
    
    def forward(self, encoder_hidden_states, 
                down00, down01, down02, down10, down11, down12, down20, down21, down22, mid, emb):
        
        down_block_res_samples = (down00, down01, down02, down10, down11, down12, down20, down21, down22)
        sample = mid

        # 5. up
        for i, upsample_block in enumerate(self.net.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=None,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=None,
                )
        # 6. post-process
        if self.net.conv_norm_out:
            sample = self.net.conv_norm_out(sample)
            sample = self.net.conv_act(sample)
        sample = self.net.conv_out(sample)
        return sample


class FrozenContorlnet(torch.nn.Module):
    def __init__(self, ControlNetModel):
        super().__init__()
        assert ControlNetModel.__class__.__name__ == 'ControlNetModel'
        self.net = ControlNetModel
    
    def forward(self, sample, timestep, encoder_hidden_states, add_text_embeds, add_time_ids, controlnet_cond, conditioning_scale):
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        downs, mid = self.net(
                        sample=sample, 
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                        controlnet_cond=controlnet_cond,
                        conditioning_scale=conditioning_scale,
                        return_dict=False)
        downs.append(mid)
        return tuple(downs)


def _export_onnx(
    model: torch.nn.Module,
    model_args: tuple,
    output_path: Path,
    input_names: list,
    output_names: list,
    dynamic_axes: dict,
    opset: int=17,
    use_external_data_format: bool=False,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
    # so we check the torch version for backwards compatibility
    if is_torch_less_than_1_11:
        torch.onnx.export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            use_external_data_format=use_external_data_format,
            enable_onnx_checker=True,
            opset_version=opset,
        )
    else:
        torch.onnx.export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset,
        )
    
    if use_external_data_format:
        onnxnet = onnx.load(output_path.as_posix())
        shutil.rmtree(output_path.parent.as_posix())
        output_path.parent.mkdir(parents=True, exist_ok=True)

        onnx.save_model(
            onnxnet,
            output_path.as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
            convert_attribute=False,
        )
    return


@torch.no_grad()
def export_onnx(
    pipe_dir: str, 
    save_dir: str, 
    lora_dir: Optional[str], 
    control_dir: Dict[str, str],
    opset: int=17, 
    dtype: torch.dtype=torch.float16,
):
    # Check Path
    pipe_dir = Path(pipe_dir)
    save_dir = Path(save_dir)
    
    device = torch.device("cuda")

    # Params
    height = 1024
    width = 1024

    # Load ControlNets
    controlnets = {}
    if control_dir:
        for hint_type, dir in control_dir.items():
            controlnets[hint_type] = ControlNetModel.from_pretrained(dir, torch_dtype=dtype, use_safetensors=True, variant="fp16").to(device)

    # Load Diffusers Pipeline
    pipeline = AutoPipelineForText2Image.from_pretrained(pipe_dir, torch_dtype=dtype, use_safetensors=True, variant="fp16")
    if lora_dir is not None:
        pipeline.load_lora_weights(lora_dir)
        pipeline.fuse_lora()

    ########################### ControlNet
    for hint_type, net in controlnets.items():
        save_path = save_dir/f"{hint_type}"/"model.onnx"
        _export_onnx(
            FrozenContorlnet(net),
            model_args=(
                torch.randn(2, 4, height//8, width//8).to(device=device, dtype=dtype),
                torch.randn(1).to(device=device, dtype=dtype),
                torch.randn(2, 77, 2048).to(device=device, dtype=dtype),
                torch.randn(2, 1280).to(device=device, dtype=dtype),
                torch.randn(2, 6).to(device=device, dtype=dtype),
                torch.randn(2, 3, height, width).to(device=device, dtype=dtype),
                torch.randn(1).to(device=device, dtype=dtype),
            ),
            output_path=save_path,
            input_names=["sample", "timestep", "encoder_hidden_states", "add_text_embeds", "add_time_ids", "controlnet_cond", "conditioning_scale"],
            output_names=["down00","down01","down02","down10","down11","down12","down20","down21","down22","mid"],
            dynamic_axes={
                "sample": {0: "batch", 2: "latent_height", 3: "latent_width"},
                "encoder_hidden_states": {0: "batch"},
                "add_text_embeds": {0: "batch"},
                "add_time_ids": {0: "batch"},
                "controlnet_cond": {0: "batch", 2: "height", 3: "width"},
            },
            opset=opset,
            use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
        )



    ########################### UNET
    unet = FrozenUnetEncoder(pipeline.unet).to(device)
    _export_onnx(
        unet,
        model_args=(
            torch.randn(2, 4, height//8, width//8).to(device=device, dtype=dtype),
            torch.randn(1).to(device=device, dtype=dtype),
            torch.randn(2, 77, 2048).to(device=device, dtype=dtype),
            torch.randn(2, 1280).to(device=device, dtype=dtype),
            torch.randn(2, 6).to(device=device, dtype=dtype),
        ),
        output_path=save_dir/"unet_encoder"/"model.onnx",
        input_names=["sample", "timestep", "encoder_hidden_states", "add_text_embeds", "add_time_ids"],
        output_names=["down00","down01","down02","down10","down11","down12","down20","down21","down22","mid","emb"],
        dynamic_axes={
            "sample": {0: "batch", 2: "latent_height", 3: "latent_width"},
            "encoder_hidden_states": {0: "batch"},
            "add_text_embeds": {0: "batch"},
            "add_time_ids": {0: "batch"},
        },
        opset=opset,
        use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
    )
    del unet

    unet = FrozenUnetDecoder(pipeline.unet).to(device)
    _export_onnx(
        unet,
        model_args=(
            torch.randn(2, 77, 2048).to(device=device, dtype=dtype),
            torch.randn(2, 320, height//8, width//8).to(device=device, dtype=dtype),
            torch.randn(2, 320, height//8, width//8).to(device=device, dtype=dtype),
            torch.randn(2, 320, height//8, width//8).to(device=device, dtype=dtype),
            torch.randn(2, 320, height//16, width//16).to(device=device, dtype=dtype),
            torch.randn(2, 640, height//16, width//16).to(device=device, dtype=dtype),
            torch.randn(2, 640, height//16, width//16).to(device=device, dtype=dtype),
            torch.randn(2, 640, height//32, width//32).to(device=device, dtype=dtype),
            torch.randn(2, 1280, height//32, width//32).to(device=device, dtype=dtype),
            torch.randn(2, 1280, height//32, width//32).to(device=device, dtype=dtype),
            torch.randn(2, 1280, height//32, width//32).to(device=device, dtype=dtype),
            torch.randn(2, 1280).to(device=device, dtype=dtype),
        ),
        output_path=save_dir/"unet_decoder"/"model.onnx",
        input_names=["encoder_hidden_states", "down00","down01","down02","down10","down11","down12","down20","down21","down22","mid","emb"],
        output_names=["out_sample"],
        dynamic_axes={
            "encoder_hidden_states": {0: "batch"},
            "down00": {0: "batch", 2: "latent_height", 3: "latent_width"},
            "down01": {0: "batch", 2: "latent_height", 3: "latent_width"},
            "down02": {0: "batch", 2: "latent_height", 3: "latent_width"},
            "down10": {0: "batch", 2: "latent_height/2", 3: "latent_width/2"},
            "down11": {0: "batch", 2: "latent_height/2", 3: "latent_width/2"},
            "down12": {0: "batch", 2: "latent_height/2", 3: "latent_width/2"},
            "down20": {0: "batch", 2: "latent_height/4", 3: "latent_width/4"},
            "down21": {0: "batch", 2: "latent_height/4", 3: "latent_width/4"},
            "down22": {0: "batch", 2: "latent_height/4", 3: "latent_width/4"},
            "mid": {0: "batch", 2: "latent_height/4", 3: "latent_width/4"},
            "emb": {0: "batch"},
        },
        opset=opset,
        use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
    )

    print("ONNX pipeline saved to", save_dir)
    del pipeline
    del controlnets

    torch.cuda.empty_cache()

