# PROJECT SCOPE — CS331 Occluded Object Reconstruction

> Tài liệu **phạm vi dự án** (scope): mục tiêu, ranh giới, deliverables, tiêu chí thành công.  
> Quy tắc vận hành kỹ thuật (ký hiệu, VRAM, reproducibility) vẫn nằm trong [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

---

## 1. Tóm tắt điều hành

| Mục | Nội dung |
|-----|----------|
| **Vấn đề** | Tái tạo vùng bị che trên ảnh xe sao cho khớp hình dạng, màu sắc và chất liệu bề mặt với phần còn lại. |
| **Phương pháp** | LDM inpainting làm backbone; **ControlNet-Canny** (cấu trúc/hình dạng); **IP-Adapter** (surface/style từ ảnh tham chiếu); fallback **Config D** khi không đủ VRAM. |
| **Nền tảng** | Notebook **Kaggle** (GPU T4 / P100 ~16GB), Python thuần, không dùng Lightning/Trainer. |
| **Đánh giá** | SSIM, LPIPS, FID — **chỉ trên vùng mask**; tập test **200 ảnh**; phân tầng theo occlusion ratio. |

---

## 2. Mục tiêu dự án

### 2.1 Mục tiêu khoa học

1. **Baseline có thể tái lập:** một pipeline inpainting diffusion với prompt cố định, scheduler và metrics thống nhất.
2. **Cải thiện có giải thích:** so sánh ablation **A → B → C** (và **D** khi cần), mỗi bước gắn với một paper (LDM, ControlNet, IP-Adapter).
3. **Điều kiện công bằng:** Canny **C** từ `x_occ` (không dùng `x_gt` cho oracle); mask-aware Canny (xóa cạnh trong vùng mask) để tránh nhiễu từ biên vật che.

### 2.2 Mục tiêu kỹ thuật (deliverable)

| ID | Deliverable | Trạng thái (theo repo) |
|----|-------------|-------------------------|
| D1 | Pipeline tạo dữ liệu synthetic occlusion | `notebooks/01_synthetic-occlusion-pipeline.ipynb` |
| D2 | Config A — baseline inpainting + báo cáo metrics | `notebooks/02`, `results/baseline_sd2/reports/` |
| D3 | Config B — SD + ControlNet-Canny (mask-aware) | `notebooks/03-controlnet-canny.ipynb`, `results-controlnet/` |
| D4 | Config C — + IP-Adapter (và fallback D) | `notebooks/04-controlnet-ip-adapter.ipynb` / `04-ip-adapter.ipynb`, `results-ip-adapter/` |
| D5 | Bảng ablation + phân tích theo occlusion ratio | CSV + (tuỳ chọn) báo cáo khóa học |
| D6 | (Tuỳ chọn) Demo Gradio 2-tab | `PROJECT_CONTEXT.md` §12 |

---

## 3. Phạm vi **trong** (in scope)

- **Đầu vào:** ảnh xe (Stanford Cars + synthetic occlusion), mask nhị phân, `x_occ` / `x_gt` đã resize **512×512** (letterbox).
- **Mô hình:** chỉ inference với checkpoint công khai (Hugging Face); không huấn luyện lại UNet đầy đủ trong phạm vi khóa học trừ khi mở rộng có chủ đích.
- **Metrics:** masked SSIM, masked LPIPS (Alex), FID (clean-fid); báo cáo mean trên 200 ảnh test.
- **Ablation:** Config A–D như bảng dưới; ghi rõ khi chạy **D** (OOM / fallback).

### 3.1 Bốn cấu hình cần so sánh

```
Config A: SD Inpainting only                          ← Baseline
Config B: SD + ControlNet-Canny                       ← +Shape (edges)
Config C: SD + ControlNet + IP-Adapter              ← +Surface (image prompt)
Config D: SD + ControlNet + Text Color Prompt       ← Fallback khi C không đủ VRAM
```

---

## 4. Phạm vi **ngoài** (out of scope — phiên bản hiện tại)

- Huấn luyện lại ControlNet / IP-Adapter trên dataset riêng.
- Video, multi-frame, hoặc occlusion 3D.
- Đánh giá trên ảnh xe thực tế ngoài Stanford Cars (có thể làm **future work**).
- Sản phẩm production: SLA, API scale, bảo mật — không bắt buộc trong CS331 trừ khi mở rộng demo.

---

## 5. Kiến trúc logic (tham chiếu paper)

| Thành phần | Vai trò | Paper / nguồn |
|------------|---------|----------------|
| LDM Inpainting | Backbone, điền vùng mask | Rombach et al. (SD) |
| ControlNet | Điều kiện cấu trúc (Canny **C**) | Zhang & Agrawala, 2302.05543 |
| IP-Adapter | Image prompt, cross-attention tách (λ) | Ye et al., 2308.06721 |

**Ký hiệu:** `x_gt`, `x_occ`, `M`, `C` (Canny từ `x_occ`), `P` (visible patch cho IP-Adapter), `x_hat` — chi tiết trong [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

---

## 6. Checkpoints & nhất quán implementation

> Trong repo, baseline và các notebook nâng cao dùng **cùng họ** SD inpainting + ControlNet v1.1 Canny SD1.5 khi so sánh trực tiếp; mọi thay đổi model ID cần ghi trong CSV summary.

| Thành phần | Checkpoint mẫu (HF) |
|------------|---------------------|
| Inpainting | `runwayml/stable-diffusion-inpainting` |
| ControlNet (Canny) | `lllyasviel/control_v11p_sd15_canny` |
| IP-Adapter | `h94/IP-Adapter` → `ip-adapter-plus_sd15.bin` |

*(Nếu `PROJECT_CONTEXT.md` còn ghi SD2.1 cho một số dòng, ưu tiên **đồng bộ với notebook đang chạy** và cập nhật bảng trong CONTEXT khi đổi model.)*

---

## 7. Dataset & test

- **Nguồn:** Stanford Cars; **occlusion:** đối tượng COCO, IoU mục tiêu **0.15–0.40**.
- **File mỗi sample:** `{id}_gt`, `{id}_occ`, `{id}_mask` (metadata CSV trong `data/synthetic_occ/`).
- **Test:** **200** ảnh, stratified theo class xe (nếu có metadata).
- **Phân tầng báo cáo:** `light < 0.20` | `medium 0.20–0.40` | `heavy > 0.40`.

---

## 8. Tiêu chí thành công

1. **Độ hoàn thành:** có ít nhất một lần chạy đầy đủ **A** và **B**; **C** (hoặc **D** có ghi chú) với summary CSV.
2. **Khoa học:** không dùng `x_gt` để tạo **C**; metrics masked đúng convention.
3. **So sánh:** bảng hoặc đoạn văn so baseline vs B vs C (và D nếu có), kèm SSIM/LPIPS/FID.
4. **Tái lập:** `SEED=42`, scheduler và prompt/negative_prompt ghi trong CSV.

---

## 9. Rủi ro & ứng phó

| Rủi ro | Ứng phó |
|--------|---------|
| OOM khi chạy cả ControlNet + IP-Adapter | `enable_xformers_memory_efficient_attention`, VAE slicing, **Config D** (text color) |
| Grid search quá tốn GPU | Giảm `GRID_SAMPLE`, cố định một scale sau pilot |
| Lệch kết quả giữa máy local / Kaggle | Ghi `diffusers` version, model ID, seed trong report |

---

## 10. Cấu trúc artifact (repo)

```
Occluded Object Reconstruction/
├── PROJECT_CONTEXT.md      ← Quy tắc agent + kỹ thuật (đọc trước khi code)
├── PROJECT_SCOPE.md        ← Phạm vi dự án (file này)
├── notebooks/
│   ├── 01_synthetic-occlusion-pipeline.ipynb
│   ├── 02_baseline-SD2-inpainting.ipynb
│   ├── 03-controlnet-canny.ipynb
│   └── 04-controlnet-ip-adapter.ipynb   (và/hoặc 04-ip-adapter.ipynb)
├── data/synthetic_occ/
└── results/                 ← baseline
    results-controlnet/      ← Config B
    results-ip-adapter/      ← Config C / D
```

---

## 11. Phiên bản tài liệu

| Phiên bản | Ngày | Ghi chú |
|-----------|------|---------|
| 1.0 | 2026-04 | Scope mới: align với notebook 01–04 và thư mục `results*`. |

---

*Kết thúc PROJECT_SCOPE.md — cập nhật khi thay đổi milestone hoặc deliverable.*
