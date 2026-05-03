"""
Load four Stable Diffusion inpainting pipelines (cached once per Streamlit session).
"""

from __future__ import annotations

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


def _build_pipe_base(dtype, device):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing("auto")
    pipe.vae.enable_slicing()
    return pipe.to(device)


def _build_lora_pipe_base(lora_path: str, dtype, device):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing("auto")
    pipe.vae.enable_slicing()
    pipe = pipe.to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.unet = pipe.unet.merge_and_unload()
    return pipe


def _build_pipe_cn(dtype, device):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=dtype,
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing("auto")
    pipe.vae.enable_slicing()
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter-plus_sd15.bin",
    )
    return pipe.to(device)


def _build_lora_pipe_cn(lora_path: str, dtype, device):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=dtype,
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing("auto")
    pipe.vae.enable_slicing()
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter-plus_sd15.bin",
    )
    pipe = pipe.to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.unet = pipe.unet.merge_and_unload()
    return pipe


@st.cache_resource(show_spinner="Loading models…")
def load_pipelines(lora_path: str) -> dict:
    """Return four diffuse pipelines plus a shared spatial LPIPS model."""
    device, dtype = _device_dtype()

    pipe_base = _build_pipe_base(dtype, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    lora_pipe_base = _build_lora_pipe_base(lora_path, dtype, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pipe_cn = _build_pipe_cn(dtype, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Fresh build avoids `deepcopy` of a GPU pipeline (very heavy).
    lora_pipe_cn = _build_lora_pipe_cn(lora_path, dtype, device)

    lp_model = lpips.LPIPS(net="alex", spatial=True).to(device)
    lp_model.eval()

    return {
        "device": device,
        "dtype": dtype,
        "pipe_base": pipe_base,
        "lora_pipe_base": lora_pipe_base,
        "pipe_cn": pipe_cn,
        "lora_pipe_cn": lora_pipe_cn,
        "lpips_model": lp_model,
    }
