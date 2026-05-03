# Cursor Prompt — Car Inpainting Streamlit Demo (CS331)

## Task

Build a **Streamlit demo app** (`demo/app.py`) for a CS331 computer vision project on occluded car inpainting. The user uploads a car image, paints over the occluded region with a brush, selects a config (A/B/C/E/F/G), and clicks Restore to see the inpainted result with metrics.

---

## Project context

6 inpainting configs using Stable Diffusion + LoRA + ControlNet + IP-Adapter. Demo must match exactly the models used in research notebooks.

**Models (exact IDs — do NOT substitute):**
- SD base: `runwayml/stable-diffusion-inpainting` ← LoRA was trained on this, not SD v2
- ControlNet: `lllyasviel/control_v11p_sd15_canny`
- IP-Adapter: `h94/IP-Adapter`, subfolder `models`, weight `ip-adapter-plus_sd15.bin`
- LoRA: local path `r8/best/` (PEFT format)
- Scheduler: replace default with `DPMSolverMultistepScheduler` on all pipes

**Configs:**
```python
CONFIGS = {
    "A — Baseline":          dict(lora=False, cn=False, ip=False),
    "B — ControlNet":        dict(lora=False, cn=True,  ip=False),
    "C — CN + IP-Adapter":   dict(lora=False, cn=True,  ip=True),
    "E — LoRA only":         dict(lora=True,  cn=False, ip=False),
    "F — LoRA + ControlNet": dict(lora=True,  cn=True,  ip=False),
    "G — Full stack (best)": dict(lora=True,  cn=True,  ip=True),
}
```

---

## File structure

```
demo/
├── app.py              # Streamlit main
├── model_loader.py     # load & cache pipelines
├── inference.py        # inpaint() + helpers
├── metrics.py          # masked_ssim(), masked_lpips()
├── examples/           # 3–5 pre-run car images + masks
│   ├── car_easy.jpg
│   ├── car_medium.jpg
│   └── car_hard.jpg
└── requirements.txt
```

---

## `model_loader.py`

Load **4 pipeline objects once**, cache with `@st.cache_resource` so Streamlit never reloads on rerun.

```python
@st.cache_resource(show_spinner="Loading models…")
def load_pipelines(lora_path: str) -> dict:
    ...
    return {
        "pipe_base":       ...,   # Config A  — SD only
        "lora_pipe_base":  ...,   # Config E  — SD + LoRA fused
        "pipe_cn":         ...,   # Configs B,C — SD + CN (+ IP loaded)
        "lora_pipe_cn":    ...,   # Configs F,G — SD + LoRA + CN (+ IP loaded)
        "lpips_model":     ...,   # lpips.LPIPS(net="alex")
    }
```

**Building each pipeline:**
```python
# 1. pipe_base (A)
pipe_base = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
)
pipe_base.scheduler = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config)
pipe_base.enable_attention_slicing("auto")
pipe_base.vae.enable_slicing()
pipe_base = pipe_base.to(device)

# 2. lora_pipe_base (E) — deep copy then fuse LoRA
lora_pipe_base = copy.deepcopy(pipe_base)   # copy before .to() is cheaper; adjust if OOM
lora_pipe_base.unet = PeftModel.from_pretrained(lora_pipe_base.unet, lora_path)
lora_pipe_base.unet = lora_pipe_base.unet.merge_and_unload()

# 3. pipe_cn (B, C)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny", torch_dtype=dtype
)
pipe_cn = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
)
pipe_cn.scheduler = DPMSolverMultistepScheduler.from_config(pipe_cn.scheduler.config)
pipe_cn.enable_attention_slicing("auto")
pipe_cn.vae.enable_slicing()
pipe_cn.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter-plus_sd15.bin")
pipe_cn = pipe_cn.to(device)

# 4. lora_pipe_cn (F, G)
lora_pipe_cn = copy.deepcopy(pipe_cn)
lora_pipe_cn.unet = PeftModel.from_pretrained(lora_pipe_cn.unet, lora_path)
lora_pipe_cn.unet = lora_pipe_cn.unet.merge_and_unload()
```

**If VRAM is tight** (< 10 GB), skip `copy.deepcopy` and build each pipeline independently from scratch — comment explains why.

---

## `inference.py`

**Constants (exact values from notebooks):**
```python
CANNY_LOW      = 80
CANNY_HIGH     = 150
MASK_DILATE_PX = 5
PROMPT     = "a car, realistic, high quality, detailed"
NEG_PROMPT = "blurry, distorted, artifacts, deformed"
```

**Helper — Canny edges (mask-aware):**
```python
def extract_canny_masked(image_pil: Image, mask_pil: Image) -> Image:
    img   = np.array(image_pil.convert("RGB"))
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    msk   = np.array(mask_pil.convert("L"))
    if MASK_DILATE_PX > 0:
        k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
              (MASK_DILATE_PX*2+1, MASK_DILATE_PX*2+1))
        msk = cv2.dilate(msk, k, iterations=1)
    edges[msk > 127] = 0          # zero out edges inside mask
    return Image.fromarray(np.stack([edges]*3, axis=2))
```

**Helper — visible patch for IP-Adapter:**
```python
def extract_visible_patch(image_pil: Image, mask_pil: Image) -> Image:
    img     = np.array(image_pil.convert("RGB"))
    msk     = np.array(mask_pil.convert("L"))
    visible = msk < 128
    mean_c  = img[visible].mean(axis=0).astype(np.uint8) \
              if visible.sum() > 0 else np.array([128,128,128], dtype=np.uint8)
    patch   = img.copy()
    patch[~visible] = mean_c
    return Image.fromarray(patch)
```

**Main function:**
```python
def inpaint(
    image_and_mask: dict,      # {"image": np.ndarray RGBA, "mask": np.ndarray L}
                               # from streamlit-drawable-canvas
    config_name: str,
    pipes: dict,               # from load_pipelines()
    steps: int        = 50,
    guidance: float   = 7.5,
    cn_scale: float   = 0.7,
    ip_scale: float   = 0.5,
    seed: int         = 42,
) -> tuple[Image.Image, str]:
```

Logic:
1. `orig_image` = resize background to 512×512
2. `mask` = from canvas stroke alpha → binary PIL mask (white = inpaint region)
3. Guard: if `mask` is all black → return `(orig_image, "⚠️ Draw a mask first")`
4. Select pipeline:
   - `lora=True, cn=True`  → `lora_pipe_cn`
   - `lora=True, cn=False` → `lora_pipe_base`
   - `lora=False, cn=True` → `pipe_cn`
   - `lora=False, cn=False`→ `pipe_base`
5. If `ip=True`: `pipe.set_ip_adapter_scale(ip_scale)` else `pipe.set_ip_adapter_scale(0.0)` (needed to disable on cn pipes that have IP loaded)
6. Build `kwargs`:
   - if `cn`: `control_image=extract_canny_masked(...)`, `controlnet_conditioning_scale=cn_scale`
   - if `ip`: `ip_adapter_image=extract_visible_patch(...)`
7. Run inference with `generator=torch.Generator(device).manual_seed(seed)`
8. Compute metrics with `masked_ssim` and `masked_lpips`
9. Return `(result_image, metrics_string)`

Wrap everything in `try/except torch.cuda.OutOfMemoryError` → return `(orig_image, "❌ CUDA OOM — reduce Steps or use CPU")`

---

## `metrics.py`

```python
def masked_ssim(gt: Image, pred: Image, mask: Image) -> float:
    """SSIM averaged over masked pixels only."""
    # convert to float32 numpy [0,1], shape (H,W,3)
    # skimage.metrics.structural_similarity(full=True) → get smap
    # return smap[mask_bool].mean()

def masked_lpips(gt: Image, pred: Image, mask: Image, model) -> float:
    """LPIPS on masked region (pass lpips.LPIPS model in, don't reload)."""
    # normalize to [-1,1], apply spatial mask as weight, return scalar
```

---

## `app.py` — Streamlit UI

**Drawing canvas** — use `streamlit-drawable-canvas`:
```python
from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="rgba(255, 68, 68, 0.4)",  # semi-transparent red brush
    stroke_width=brush_size,
    stroke_color="rgba(255, 68, 68, 0.9)",
    background_image=uploaded_pil,
    update_streamlit=True,
    height=img_display_h,
    width=img_display_w,
    drawing_mode="freedraw",
    key="canvas",
)
```

**Full layout:**
```
st.set_page_config(layout="wide", page_title="Car Inpainting — CS331")
st.title("🚗 Car Inpainting Demo")
st.caption("CS331 · Diffusion Models with LoRA + ControlNet + IP-Adapter")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # File uploader
    uploaded = st.file_uploader("Upload car image", type=["jpg","jpeg","png"])

    if uploaded:
        # Show drawable canvas with uploaded image as background
        brush_size = st.slider("Brush size", 10, 80, 30)
        canvas_result = st_canvas(...)    # freedraw mode, red brush

    config = st.selectbox("Config", list(CONFIGS.keys()),
                          index=5)   # default G

    with st.expander("Advanced parameters", expanded=False):
        col1, col2 = st.columns(2)
        steps    = col1.slider("Steps",    20, 100, 50, 5)
        guidance = col2.slider("Guidance", 1.0, 15.0, 7.5, 0.5)
        cn_scale = col1.slider("CN scale", 0.0, 1.0, 0.7, 0.1)
        ip_scale = col2.slider("IP scale", 0.0, 1.0, 0.5, 0.1)
        seed     = st.number_input("Seed", value=42, step=1)

    run = st.button("🔁 Restore masked region",
                    type="primary", use_container_width=True,
                    disabled=uploaded is None)

with col_right:
    result_placeholder = st.empty()
    metrics_placeholder = st.empty()

    # Preset examples at bottom
    st.divider()
    st.markdown("**Preset examples** (instant, pre-computed)")
    ex_cols = st.columns(3)
    # 3 example buttons that load image + show cached result
```

**Running inference:**
```python
if run:
    with st.spinner(f"Running {config}…"):
        result_img, metrics_str = inpaint(
            image_and_mask={"image": canvas_result.image_data,
                            "mask":  canvas_result.image_data},
            config_name=config,
            pipes=pipes,          # from load_pipelines()
            steps=steps, guidance=guidance,
            cn_scale=cn_scale, ip_scale=ip_scale, seed=seed,
        )
    result_placeholder.image(result_img, caption="Restored", use_column_width=True)
    metrics_placeholder.info(metrics_str)

    # Download button
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    st.download_button("⬇️ Download result", buf.getvalue(),
                       file_name="inpaint_result.png", mime="image/png")
```

**Preset examples logic:**
- Load 3 pre-run `(input, result, metrics)` tuples from `demo/examples/`
- Show as clickable cards in 3 columns
- Clicking loads the result instantly without running inference
- This is the **backup** if GPU is slow on demo day

---

## `requirements.txt`

```
streamlit>=1.32.0
streamlit-drawable-canvas>=0.9.3
torch>=2.0.0
diffusers>=0.27.0
transformers>=4.35.0
peft>=0.7.0
accelerate>=0.24.0
opencv-python>=4.8.0
scikit-image>=0.21.0
lpips>=0.1.4
Pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

---

## Constraints

1. **`@st.cache_resource`** on `load_pipelines()` — Streamlit reruns on every interaction; without this, models reload every click.

2. **Canvas image_data** is RGBA numpy array (H×W×4). Extract mask from alpha channel:
   ```python
   alpha = canvas_result.image_data[:, :, 3]
   mask_arr = (alpha > 10).astype(np.uint8) * 255
   mask_pil = Image.fromarray(mask_arr, mode="L")
   ```

3. **Canvas size** — scale uploaded image to fit display (max 512px wide), record scale factor, then resize mask back to 512×512 before passing to pipeline.

4. **IP-Adapter on CN pipes only** — `pipe_base` and `lora_pipe_base` do NOT have IP-Adapter loaded. Never call `set_ip_adapter_scale` on them.

5. **No xformers** — use `enable_attention_slicing("auto")`.

6. **LoRA path config** — read from `st.secrets["LORA_PATH"]` with fallback to env var `LORA_WEIGHTS_PATH`, then fallback to `"outputs/lora_weights/r8/best"`.

7. If LoRA path not found on startup → `st.error(...)` + `st.stop()` with clear message.

---

## What NOT to do

- **Never** use `stabilityai/stable-diffusion-2-inpainting` — LoRA trained on SD v1.5
- **Never** reload LoRA in `inpaint()` per call — fuse once in `model_loader.py`
- **Never** use `controlnet_aux.CannyDetector` — use `cv2.Canny` directly
- **Never** call `st.cache_data` on pipelines (use `st.cache_resource`)
- **Never** put `load_pipelines()` call inside a button callback — call at module level so it runs once on startup
