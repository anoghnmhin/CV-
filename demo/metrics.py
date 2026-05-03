"""Masked SSIM and spatial LPIPS on the inpainting region."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def _pil_to_float_rgb(pil: Image.Image) -> np.ndarray:
    return np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0


def _pil_mask_bool(mask: Image.Image, thresh: float = 0.5) -> np.ndarray:
    m = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return m > thresh


def masked_ssim(gt: Image.Image, pred: Image.Image, mask: Image.Image) -> float:
    gt_f = _pil_to_float_rgb(gt)
    pred_f = _pil_to_float_rgb(pred)
    mb = _pil_mask_bool(mask)
    if not np.any(mb):
        return float("nan")
    smap = ssim(
        gt_f,
        pred_f,
        data_range=1.0,
        channel_axis=2,
        full=True,
    )[1]
    return float(np.mean(smap[mb]))


def masked_lpips(gt: Image.Image, pred: Image.Image, mask: Image.Image, model) -> float:
    device = next(model.parameters()).device
    mb = _pil_mask_bool(mask)
    if not np.any(mb):
        return float("nan")

    def pil_to_n11(pil: Image.Image) -> torch.Tensor:
        arr = _pil_to_float_rgb(pil)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        return t * 2.0 - 1.0

    gt_t = pil_to_n11(gt)
    pred_t = pil_to_n11(pred)

    mask_t = torch.from_numpy(mb.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device=device)

    with torch.no_grad():
        spatial = model(pred_t, gt_t, normalize=False)

    spatial = spatial.mean(dim=1, keepdim=True)
    mh, mw = spatial.shape[-2], spatial.shape[-1]
    mask_r = F.interpolate(mask_t, size=(mh, mw), mode="nearest")

    num = (spatial * mask_r).sum()
    denom = mask_r.sum().clamp(min=1e-8)
    return float((num / denom).item())
