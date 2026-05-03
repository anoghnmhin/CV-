# Demo Streamlit — Car Inpainting (CS331)

Ứng dụng inpainting ô tô che khuất: upload ảnh, vẽ mask trên canvas, chọn cấu hình (A/B/C/E/F/G), nhấn **Restore** để xem ảnh khôi phục và metrics (masked SSIM / LPIPS).

## Yêu cầu hệ thống

| Hạng mục | Gợi ý |
|----------|--------|
| **Python** | 3.10+ (demo đã thử trên 3.12) |
| **VRAM** | Nên có GPU NVIDIA (CUDA). CPU chạy được nhưng rất chậm; nhiều pipeline được load trong bộ nhớ cùng lúc → cần RAM/VRAM đủ lớn (~12 GB VRAM là mức tham khảo). |
| **Ổ đĩa** | **~15 GB trống trở lên** cho Hugging Face cache (Stable Diffusion Inpainting, ControlNet Canny, IP-Adapter…). Cache mặc định nằm tại `~/.cache/huggingface`. Nếu hết chỗ trong lúc tải, gặp lỗi `No space left on device`. |

## 1. Tạo môi trường và cài phụ thuộc

Từ **thư mục gốc repo** (`Occluded Object Reconstruction`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r demo/requirements.txt
```

## 2. Trọng số LoRA (PEFT)

App cần thư mục LoRA đã train (định dạng PEFT) trỏ tới **`.../best`** (hoặc tương đương).

**Thứ tự ưu tiên:**

1. `LORA_PATH` trong file **Streamlit secrets** — xem bước 3  
2. Biến môi trường **`LORA_WEIGHTS_PATH`**  
3. Đường mặc định: **`outputs/lora_weights/r8/best`** (tính từ **thư mục repo**, không phải `demo/`)

Ví dụ copy LoRA vào repo:

```
Occluded Object Reconstruction/
  outputs/
    lora_weights/
      r8/
        best/          ← chứa adapter PEFT + config
```

Nếu thư mục LoRA không tồn tại, Streamlit báo lỗi và dừng.

## 3. Streamlit secrets (khuyến nghị)

Nếu **chưa có** file `secrets.toml`, một số phiên bản Streamlit sẽ lỗi khi đọc `st.secrets`. Tạo file tại một trong các vị trí:

- **`Occluded Object Reconstruction/.streamlit/secrets.toml`** (khuyến nghị), hoặc  
- **`demo/.streamlit/secrets.toml`**

Nội dung ví dụ:

```toml
LORA_PATH = "outputs/lora_weights/r8/best"
```

Có thể dùng đường dẫn **tuyệt đối** tới thư mục LoRA trên máy bạn.

**Lưu ý bảo mật:** không commit `secrets.toml` nếu chứa đường dẫn nhạy cảm — nên thêm `.streamlit/secrets.toml` vào `.gitignore` nếu cần.

## 4. Chạy demo

Luôn chạy từ **thư mục gốc repo** (để đường dẫn LoRA và import đúng):

```bash
source .venv/bin/activate   # nếu dùng venv
streamlit run demo/app.py
```

Mở trình duyệt tại URL hiển thị (thường `http://localhost:8501`).

**Biến môi trường tùy chọn**

- **`LORA_WEIGHTS_PATH`** — ghi đè LoRA khi không dùng / không set `LORA_PATH` trong secrets.
- **`HF_HOME`** hoặc **`HUGGINGFACE_HUB_CACHE`** — chuyển cache model sang ổ khác khi SSD đầy.

## 5. Giao diện nhanh

- Upload ảnh xe → chỉnh kích cỡ cọ → vẽ vùng cần inpaint → chọn config → **Restore masked region**.
- Khối **Preset examples** là ảnh/metrics có sẵn trong `demo/examples/` (dùng khi demo không cần chờ inference).
- Sau khi chạy thành công có nút **Download result** (PNG).

## 6. Sự cố thường gặp

| Triệu chứng | Gợi ý |
|-------------|--------|
| `No module named 'streamlit_drawable_canvas'` | `pip install streamlit-drawable-canvas` (hoặc lại `-r demo/requirements.txt`). |
| `No module named 'peft'` / khác package | Luôn cài **`pip install -r demo/requirements.txt`** trong **đúng venv** rồi chạy lại Streamlit. |
| `StreamlitSecretNotFoundError` / không tìm thấy secrets | Tạo `.streamlit/secrets.toml` như mục 3. |
| `No space left on device` khi khởi động | Giải phóng ổ hoặc trỏ `HF_HOME` sang ổ có nhiều dung trống. |
| Tải model chậm / lần đầu lâu | Bình thường — `from_pretrained` tải từ Hugging Face; các lần sau dùng cache. |

---

Model ID cố định (không đổi trong dự án): `runwayml/stable-diffusion-inpainting`, ControlNet SD1.5 Canny, IP-Adapter Plus SD1.5 — xem `cursorrules_streamlit` / `cursor_prompt_streamlit_demo.md` trong repo gốc.
