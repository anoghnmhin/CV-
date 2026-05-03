"""
Load Stable Diffusion inpainting pipelines efficiently.
Only keeps ONE pipeline in memory at a time to prevent OOM.
"""
from __future__ import annotations

import gc
import lpips
import streamlit as st
import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
)
from peft import PeftModel

def _device_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return device, dtype

# max_entries=1 đảm bảo khi đổi config, model cũ sẽ bị xóa khỏi bộ nhớ!
@st.cache_resource(max_entries=1, show_spinner="Loading pipeline (can take a minute)...")
def get_pipeline(use_lora: bool, use_cn: bool, lora_path: str):
    device, dtype = _device_dtype()

    # Dọn dẹp VRAM trước khi load model mới
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if use_cn:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=dtype
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        # Load luôn IP-Adapter nếu dùng ControlNet
        pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin"
        )
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    # Tối ưu hóa bộ nhớ và tốc độ
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing("auto")
    pipe.vae.enable_slicing()
    pipe = pipe.to(device)

    # Gắn weights LoRA nếu cần
    if use_lora:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        pipe.unet = pipe.unet.merge_and_unload()

    return pipe

@st.cache_resource(max_entries=1, show_spinner="Loading LPIPS...")
def get_lpips():
    device, _ = _device_dtype()
    lp_model = lpips.LPIPS(net="alex", spatial=True).to(device)
    lp_model.eval()
    return lp_model