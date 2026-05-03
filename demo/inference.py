"""Inpainting entrypoint and preprocessing helpers."""
from __future__ import annotations
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from metrics import masked_lpips, masked_ssim

CANNY_LOW = 80
CANNY_HIGH = 150
MASK_DILATE_PX = 5
PROMPT = "a car, realistic, high quality, detailed"
NEG_PROMPT = "blurry, distorted, artifacts, deformed"

CONFIGS = {
    "A — Baseline": dict(lora=False, cn=False, ip=False),
    "B — ControlNet": dict(lora=False, cn=True, ip=False),
    "C — CN + IP-Adapter": dict(lora=False, cn=True, ip=True),
    "E — LoRA only": dict(lora=True, cn=False, ip=False),
    "F — LoRA + ControlNet": dict(lora=True, cn=True, ip=False),
    "G — Full stack (best)": dict(lora=True, cn=True, ip=True),
}

def extract_canny_masked(image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    msk = np.array(mask_pil.convert("L"))
    if MASK_DILATE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE_PX * 2 + 1, MASK_DILATE_PX * 2 + 1))
        msk = cv2.dilate(msk, k, iterations=1)
    edges[msk > 127] = 0
    return Image.fromarray(np.stack([edges] * 3, axis=2))

def extract_visible_patch(image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    img = np.array(image_pil.convert("RGB"))
    msk = np.array(mask_pil.convert("L"))
    visible = msk < 128
    if visible.sum() > 0:
        mean_c = img[visible].mean(axis=0).astype(np.uint8)
    else:
        mean_c = np.array([128, 128, 128], dtype=np.uint8)
    patch = img.copy()
    patch[~visible] = mean_c
    return Image.fromarray(patch)

def _rgba_display_to_mask_u8(rgba: np.ndarray | None) -> tuple[Image.Image, bool]:
    if rgba is None:
        return Image.new("L", (1, 1), 0), False
    alpha = rgba[:, :, 3].astype(np.int32)
    mask_arr = (alpha > 10).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_arr, mode="L")
    return mask_pil, bool(mask_arr.max() > 0)

def _resize_mask(mask_pil: Image.Image, size: tuple[int, int]) -> Image.Image:
    return mask_pil.resize(size, Image.Resampling.NEAREST)

def inpaint(
    image_and_mask: dict[str, Any],
    config_name: str,
    pipe: Any,          # Đổi từ pipes dict sang pipe object
    lpips_model: Any,   # Truyền rời model đo đạc
    steps: int = 50,
    guidance: float = 7.5,
    cn_scale: float = 0.7,
    ip_scale: float = 0.5,
    seed: int = 42,
) -> tuple[Image.Image, str]:
    base_512 = image_and_mask["base_512"].convert("RGB")
    rgba = image_and_mask.get("rgba")

    # BUG 2 FIX: Sửa thứ tự unpack
    disp_w, disp_h = image_and_mask["display_wh"] 
    mask_display, has_stroke = _rgba_display_to_mask_u8(rgba)
    
    if mask_display.size != (disp_w, disp_h):
        mask_display = mask_display.resize((disp_w, disp_h), Image.Resampling.NEAREST)
    mask_512 = _resize_mask(mask_display, (512, 512))

    if config_name not in CONFIGS:
        return base_512, f"⚠️ Unknown config: {config_name}"
    
    # Check đã được dời ra frontend app.py cho user-friendly hơn, nhưng cứ giữ back-up
    if not has_stroke:
        return base_512, "⚠️ Please draw a mask first."

    cfg = CONFIGS[config_name]
    device = pipe.device

    kwargs: dict[str, Any] = dict(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        image=base_512,
        mask_image=mask_512,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    if cfg["cn"]:
        kwargs["control_image"] = extract_canny_masked(base_512, mask_512)
        kwargs["controlnet_conditioning_scale"] = cn_scale
        
        # BUG 5 FIX: Check an toàn trước khi set
        if hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(ip_scale if cfg["ip"] else 0.0)
            
        if cfg["ip"]:
            kwargs["ip_adapter_image"] = extract_visible_patch(base_512, mask_512)

    elif cfg["ip"]:
        return base_512, "⚠️ IP-Adapter is only used with ControlNet in this demo."

    try:
        result = pipe(**kwargs).images[0]
    except torch.cuda.OutOfMemoryError:
        return base_512, "❌ CUDA OOM — reduce Steps or use CPU"
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return base_512, "❌ CUDA OOM — reduce Steps or use CPU"
        raise

    ssim_v = masked_ssim(base_512, result, mask_512)
    lp = masked_lpips(base_512, result, mask_512, lpips_model)

    metrics_str = f"Masked SSIM: {ssim_v:.4f}\nMasked LPIPS: {lp:.4f}"

    return result, metrics_str