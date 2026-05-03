"""Streamlit demo — Car inpainting with LoRA, ControlNet Canny, and IP-Adapter."""
from __future__ import annotations

import gc
import io
import json
import os
from pathlib import Path

# Quản lý VRAM thông minh hơn (tránh lỗi 1455)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from inference import CONFIGS, inpaint
from model_loader import get_pipeline, get_lpips

REPO_ROOT    = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

def _resolve_lora_path() -> Path:
    raw = ""
    try:
        raw = st.secrets["LORA_PATH"]
    except (KeyError, FileNotFoundError):
        raw = os.environ.get("LORA_WEIGHTS_PATH", "")
    if not raw.strip():
        raw = "outputs/lora_weights/r8/best"
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()

def _fit_to_canvas(pil_rgb: Image.Image, max_side: int = 512):
    w, h   = pil_rgb.size
    scale  = min(max_side / w, max_side / h, 1.0)
    nw     = max(1, int(round(w * scale)))
    nh     = max(1, int(round(h * scale)))
    resized = pil_rgb.resize((nw, nh), Image.Resampling.LANCZOS)
    return resized, nw, nh

def _mask_from_canvas(rgba: np.ndarray | None, canvas_wh: tuple[int, int]) -> Image.Image:
    cw, ch = canvas_wh
    if rgba is None or not isinstance(rgba, np.ndarray):
        return Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    rgba = np.asarray(rgba, dtype=np.uint8)
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        alpha = rgba[:, :, 3]
    else:
        alpha = rgba[:, :, 0] if rgba.ndim == 3 else rgba
    mask_bin = (alpha > 10).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_bin, mode="L").resize((512, 512), Image.Resampling.NEAREST)
    return mask_pil

def _load_preset_manifest():
    path = EXAMPLES_DIR / "manifest.json"
    if not path.is_file(): return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def main():
    st.set_page_config(layout="wide", page_title="Car Inpainting — CS331")
    st.title("🚗 Car Inpainting Demo")
    st.caption("CS331 · Diffusion Models with LoRA + ControlNet + IP-Adapter")

    for key, default in [("result_img", None), ("result_metrics", ""), ("preset_label", "")]:
        if key not in st.session_state:
            st.session_state[key] = default

    lora_path = _resolve_lora_path()
    if not lora_path.is_dir():
        st.error(f"Không thấy thư mục trọng số LoRA: `{lora_path}`")
        st.stop()

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        uploaded = st.file_uploader("Upload car image", type=["jpg", "jpeg", "png"], key="uploader")
        pil_orig   = None
        canvas_wh  = (512, 512)
        canvas_result = None

        if uploaded is not None:
            raw_bytes = uploaded.read()
            pil_orig  = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            pil_canvas, cw, ch = _fit_to_canvas(pil_orig, max_side=512)
            canvas_wh = (cw, ch)

            # CHÌA KHÓA CHÍNH NẰM Ở ĐÂY: Lưu ảnh vào session_state để Streamlit không xóa bộ nhớ
            if "bg_image" not in st.session_state:
                st.session_state["bg_image"] = None
            st.session_state["bg_image"] = pil_canvas.convert("RGBA")

            st.markdown("**Tô vào vùng bị che để tạo mask:**")
            brush_size = st.slider("Brush size", 5, 80, 25, key="brush")

            # ── Canvas ──────────────────────────────────────────────────────
            canvas_result = st_canvas(
                fill_color      = "rgba(255, 50, 50, 0.45)",
                stroke_width    = brush_size,
                stroke_color    = "#FF3232",
                background_image= st.session_state["bg_image"],  # Gọi ảnh trực tiếp từ session_state
                update_streamlit= True,
                height          = ch,
                width           = cw,
                drawing_mode    = "freedraw",
                key             = f"canvas_{uploaded.name}_{cw}_{ch}",
            )

        config = st.selectbox("Config", list(CONFIGS.keys()), index=len(CONFIGS) - 1)

        with st.expander("Advanced parameters", expanded=False):
            ac1, ac2 = st.columns(2)
            steps    = ac1.slider("Steps",    20, 100,  50, 5)
            guidance = ac2.slider("Guidance",  1.0, 15.0, 7.5, 0.5)
            cn_scale = ac1.slider("CN scale",  0.0,  1.0, 0.7, 0.05)
            ip_scale = ac2.slider("IP scale",  0.0,  1.0, 0.5, 0.05)
            seed     = st.number_input("Seed", value=42, step=1)

        run = st.button("🔁 Restore masked region", type="primary", use_container_width=True, disabled=(uploaded is None))

    with col_right:
        preset_rows = _load_preset_manifest()
        if preset_rows:
            st.markdown("**Preset examples** (kết quả đã tính sẵn):")
            ex_cols = st.columns(min(len(preset_rows), 3))
            for i, preset in enumerate(preset_rows[:3]):
                with ex_cols[i]:
                    inp_p = EXAMPLES_DIR / preset["input"]
                    ok    = inp_p.is_file()
                    if st.button(preset["label"], key=f"preset_{preset['id']}", use_container_width=True, disabled=not ok):
                        res_p = EXAMPLES_DIR / preset["result"]
                        st.session_state.result_img = Image.open(res_p if res_p.is_file() else inp_p).convert("RGB")
                        st.session_state.result_metrics = preset.get("metrics", "")
                        st.session_state.preset_label   = preset["label"]

        st.markdown("### Kết quả")
        result_placeholder  = st.empty()
        metrics_placeholder = st.empty()
        dl_slot             = st.empty()

        if run and pil_orig is not None:
            rgba = getattr(canvas_result, "image_data", None) if canvas_result else None
            
            # BUG 3 FIX: Cảnh báo rõ ràng nếu user chưa vẽ gì
            if rgba is None or np.max(rgba[:, :, 3]) == 0:
                st.warning("⚠️ Bạn chưa vẽ mask! Hãy dùng cọ đỏ tô vào vùng bị che trên ảnh bên trái trước khi bấm Restore.")
            else:
                # Dynamic model loading (Chỉ load đúng config được yêu cầu)
                cfg_dict = CONFIGS[config]
                pipe = get_pipeline(use_lora=cfg_dict["lora"], use_cn=cfg_dict["cn"], lora_path=str(lora_path))
                lp_model = get_lpips()

                mask_512 = _mask_from_canvas(rgba, canvas_wh)
                base_512 = pil_orig.resize((512, 512), Image.Resampling.LANCZOS)
                payload = {
                    "base_512": base_512,
                    "mask_512": mask_512,
                    "rgba": rgba,
                    "display_wh": canvas_wh,
                }

                with st.spinner(f"Running {config} …"):
                    result_img, metrics_str = inpaint(
                        payload,
                        config_name = config,
                        pipe        = pipe,
                        lpips_model = lp_model,
                        steps       = int(steps),
                        guidance    = float(guidance),
                        cn_scale    = float(cn_scale),
                        ip_scale    = float(ip_scale),
                        seed        = int(seed),
                    )

                st.session_state.result_img     = result_img
                st.session_state.result_metrics = metrics_str
                st.session_state.preset_label   = ""

                # Giải phóng RAM/VRAM ngay sau khi render xong
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if st.session_state.result_img is not None:
            caption = f"{st.session_state.preset_label} · preset" if st.session_state.preset_label else "Restored"
            result_placeholder.image(
                _pil_to_bytes(st.session_state.result_img),
                caption=caption,
                use_column_width=True,
            )
            if st.session_state.result_metrics:
                metrics_placeholder.info(st.session_state.result_metrics)

            dl_slot.download_button(
                "⬇️ Download result",
                data=_pil_to_bytes(st.session_state.result_img),
                file_name="inpaint_result.png",
                mime="image/png",
                use_container_width=True,
                key="dl_btn",
            )

if __name__ == "__main__":
    main()