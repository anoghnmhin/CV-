"""Streamlit demo — Car inpainting with LoRA, ControlNet Canny, and IP-Adapter."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from inference import CONFIGS, inpaint
from model_loader import load_pipelines

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def _resolve_lora_path() -> Path:
    raw = ""
    try:
        raw = st.secrets["LORA_PATH"]
    except KeyError:
        raw = os.environ.get("LORA_WEIGHTS_PATH", "")
    if not raw.strip():
        raw = "outputs/lora_weights/r8/best"
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def _img_for_display(rgb: Image.Image, max_w: int = 512):
    rgb = rgb.convert("RGB")
    w, h = rgb.size
    if w <= max_w:
        return rgb, w, h
    nw = max_w
    nh = int(round(h * (max_w / w)))
    resized = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
    return resized, nw, nh


def _load_preset_manifest():
    path = EXAMPLES_DIR / "manifest.json"
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def main():
    st.set_page_config(layout="wide", page_title="Car Inpainting — CS331")
    st.title("🚗 Car Inpainting Demo")
    st.caption("CS331 · Diffusion Models with LoRA + ControlNet + IP-Adapter")

    if "demo_result_img" not in st.session_state:
        st.session_state.demo_result_img = None
        st.session_state.demo_result_metrics = ""
    if "preset_label" not in st.session_state:
        st.session_state.preset_label = ""

    lora_path = _resolve_lora_path()
    if not lora_path.is_dir():
        st.error(
            f"Không thấy thư mục trọng số LoRA: `{lora_path}`\n\n"
            "Đặt `LORA_PATH` trong `.streamlit/secrets.toml`, hoặc biến môi trường `LORA_WEIGHTS_PATH`, "
            "hoặc đặt PEFT vào đúng đường dẫn mặc định trong repo."
        )
        st.stop()

    pipes = load_pipelines(str(lora_path))

    col_left, col_right = st.columns([1, 1], gap="large")

    pil_orig = None
    disp_hw = (512, 512)
    canvas_result = None
    uploaded = None
    config = list(CONFIGS.keys())[-1]
    steps = 50
    guidance = 7.5
    cn_scale = 0.7
    ip_scale = 0.5
    seed = 42
    run = False

    with col_left:
        uploaded = st.file_uploader("Upload car image", type=["jpg", "jpeg", "png"])

        if uploaded is not None:
            pil_orig = Image.open(uploaded).convert("RGB")
            pil_display, disp_w, disp_h = _img_for_display(pil_orig, max_w=512)
            disp_hw = (disp_h, disp_w)

            brush_size = st.slider("Brush size", 10, 80, 30)

            canvas_id = getattr(uploaded, "file_id", None) or f"{uploaded.name}_{getattr(uploaded, 'size', 0)}"
            canvas_result = st_canvas(
                fill_color="rgba(255, 68, 68, 0.4)",
                stroke_width=brush_size,
                stroke_color="rgba(255, 68, 68, 0.9)",
                background_image=pil_display,
                update_streamlit=True,
                height=disp_h,
                width=disp_w,
                drawing_mode="freedraw",
                key=f"canvas_{canvas_id}",
            )

        config = st.selectbox("Config", list(CONFIGS.keys()), index=len(CONFIGS) - 1)

        with st.expander("Advanced parameters", expanded=False):
            ac1, ac2 = st.columns(2)
            steps = ac1.slider("Steps", 20, 100, 50, 5)
            guidance = ac2.slider("Guidance", 1.0, 15.0, 7.5, 0.5)
            cn_scale = ac1.slider("CN scale", 0.0, 1.0, 0.7, 0.1)
            ip_scale = ac2.slider("IP scale", 0.0, 1.0, 0.5, 0.1)
            seed = st.number_input("Seed", value=42, step=1)

        run = st.button(
            "🔁 Restore masked region",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )

    def show_result(placeholder_img, placeholder_metrics, im: Image.Image | None, text: str, caption: str):
        if im is None:
            return
        placeholder_img.image(im, caption=caption, use_column_width=True)
        if text:
            placeholder_metrics.info(text)

    with col_right:
        results_blk = st.container()
        presets_blk = st.container()

        with presets_blk:
            preset_rows = _load_preset_manifest()
            st.markdown("**Preset examples** (tải sẵn, không chạy model)")
            ex_cols = st.columns(3)
            for i, preset in enumerate(preset_rows[:3]):
                with ex_cols[i]:
                    inp_p = EXAMPLES_DIR / preset["input"]
                    ok = inp_p.is_file()
                    if st.button(
                        preset["label"],
                        key=f"preset_btn_{preset['id']}",
                        use_container_width=True,
                        disabled=not ok,
                    ):
                        res_p = EXAMPLES_DIR / preset["result"]
                        if res_p.is_file():
                            st.session_state.demo_result_img = Image.open(res_p).convert("RGB")
                        else:
                            st.session_state.demo_result_img = Image.open(inp_p).convert("RGB")
                        st.session_state.demo_result_metrics = preset.get("metrics", "")
                        st.session_state.preset_label = preset["label"]
                    if not ok:
                        st.caption("Thiếu file ảnh")

        with results_blk:
            st.markdown("### Kết quả")
            result_placeholder = st.empty()
            metrics_placeholder = st.empty()
            dl_slot = st.empty()

            if run and pil_orig is not None:
                rgba = getattr(canvas_result, "image_data", None)
                if isinstance(rgba, np.ndarray):
                    rgba = rgba.astype(np.uint8, copy=False)
                elif rgba is not None:
                    rgba = np.asarray(rgba)

                base_512 = pil_orig.resize((512, 512), Image.Resampling.LANCZOS)
                dm = dict(
                    base_512=base_512,
                    rgba=rgba,
                    display_wh=disp_hw,
                )
                with st.spinner(f"Running {config}…"):
                    result_img, metrics_str = inpaint(
                        dm,
                        config_name=config,
                        pipes=pipes,
                        steps=steps,
                        guidance=guidance,
                        cn_scale=cn_scale,
                        ip_scale=ip_scale,
                        seed=int(seed),
                    )
                st.session_state.demo_result_img = result_img
                st.session_state.demo_result_metrics = metrics_str
                st.session_state.preset_label = ""

            if st.session_state.demo_result_img is not None:
                cap = st.session_state.preset_label + " · preset" if st.session_state.preset_label else "Restored"
                show_result(result_placeholder, metrics_placeholder, st.session_state.demo_result_img, st.session_state.demo_result_metrics, cap)
                raw_png = io.BytesIO()
                st.session_state.demo_result_img.save(raw_png, format="PNG")
                dl_slot.download_button(
                    "⬇️ Download result",
                    raw_png.getvalue(),
                    file_name="inpaint_result.png",
                    mime="image/png",
                    use_container_width=True,
                    key="download_result_png",
                )


if __name__ == "__main__":
    main()
