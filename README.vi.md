# KerasHub TranslateGemma 4B IT trên Kaggle T4x2 — Văn bản + Hình ảnh

> 🌐 Language / Ngôn ngữ: [English](README.md) | **Tiếng Việt**

Dự án Kaggle công khai và có thể tái lập để chạy **TranslateGemma 4B IT** bằng **KerasHub + JAX** trên bộ tăng tốc **Kaggle GPU T4 x2**.

Máy chủ sử dụng cả hai GPU T4 bằng cách khởi chạy **một worker mô hình độc lập trên mỗi GPU** phía sau một Flask coordinator nhẹ. Dự án hỗ trợ dịch văn bản thông thường và dịch hình ảnh sang văn bản (OCR + dịch) bằng cùng một checkpoint đa phương thức.

**Tác giả:** Đăng Khoa <i.am@dangkhoa.dev>

## Điểm nổi bật

- Mô hình đa phương thức KerasHub `translategemma_4b_it`.
- Hai GPU worker độc lập: GPU 0 và GPU 1.
- Endpoint dịch văn bản và dịch hình ảnh.
- API job đồng bộ và bất đồng bộ.
- Khởi động worker thích ứng dựa trên trạng thái JAX cache và RAM host còn trống.
- JAX persistent compilation cache và generation-length bucketing.
- Warm-up cho các shape sinh văn bản và vision thường gặp.
- Xác thực bằng API key được tạo cục bộ ở lần khởi động đầu tiên.
- Tùy chọn Cloudflare Quick Tunnel để truy cập công khai tạm thời.
- Unit test, smoke test text/vision, benchmark đồng thời T4x2, benchmark dtype và benchmark startup/cache.
- Notebook Kaggle được lưu trực tiếp trong repository; không cần một gói dự án riêng.
- Tài liệu tiếng Anh và tiếng Việt với liên kết chuyển đổi ngôn ngữ.

## Mô hình và phần cứng

Ứng dụng được tối ưu cho môi trường Kaggle **GPU T4 x2**. Mỗi worker tải một checkpoint TranslateGemma 4B IT đa phương thức hoàn chỉnh lên một GPU T4, vì vậy chế độ vision sử dụng gần hết 16 GB VRAM khả dụng trên mỗi GPU.

Preset KerasHub là `translategemma_4b_it`. Keras mô tả đây là mô hình dịch đa phương thức 4,30 tỷ tham số, hỗ trợ đầu vào văn bản và hình ảnh. Tham khảo chính thức: [mô hình TranslateGemma trên Kaggle](https://www.kaggle.com/models/keras/translategemma) và [các preset KerasHub Gemma3CausalLM](https://keras.io/keras_hub/api/models/gemma3/gemma3_causal_lm/). Repository mặc định dùng đường dẫn mô hình Keras được Kaggle mount:

```text
/kaggle/input/models/keras/translategemma/keras/translategemma_4b_it/1
```

Nếu Kaggle mount một thư mục phiên bản khác, `MODEL_AUTO_DISCOVER=true` sẽ tự động tìm các phiên bản mô hình đã được đính kèm.

## Quy trình Kaggle được khuyến nghị

### 1. Import notebook trực tiếp từ GitHub

Trên Kaggle, mở hộp thoại import notebook và chọn **GitHub**. Chọn repository này và import:

```text
notebooks/kaggle-t4x2-text-vision.ipynb
```

Đây là quy trình được khuyến nghị vì Kaggle sẽ tải trực tiếp toàn bộ cell của notebook từ repository.

### 2. Cấu hình phiên Kaggle

Trước khi chạy notebook:

1. Chọn **Accelerator → GPU T4 x2**.
2. Bật **Internet** để cell đầu tiên có thể clone/cập nhật repository và `scripts/setup.sh` có thể tải `cloudflared` khi cần.
3. Thêm mô hình Keras **TranslateGemma** làm Kaggle Model input nếu chưa được đính kèm.

### 3. Chạy notebook từ cell đầu tiên

Cell đầu tiên clone repository này vào:

```text
/kaggle/working/KerasHub-TranslateGemma-4B-IT-Kaggle-GPU-T4x2-Text-Vision
```

Chạy lại cell đó sẽ cập nhật working copy theo nhánh `main` hiện tại. Các cell tiếp theo tạo `.env` từ `.env.example`, kiểm tra cấu hình T4x2, cài đặt/kiểm tra dependency, chạy unit test, khởi động cả hai GPU worker, rồi chạy test text/vision và benchmark.

> Chỉ chạy `git clone` không thể thay thế notebook Kaggle đang mở hoặc tự chèn các cell vào notebook đó. Hãy import notebook của repository trước; cell clone sau đó cung cấp source tree cho các cell đã được Kaggle tải sẵn.

## Cấu trúc repository

```text
.
├── .github/                    # CI và template GitHub
├── assets/                     # Tài nguyên test vision công khai
├── bin/                        # Binary cloudflared tải lúc chạy (được ignore)
├── data/                       # Input mẫu công khai; secret sinh ra được ignore
├── docs/                       # Tài liệu kiến trúc, Kaggle, benchmark (EN/VI)
├── log/                        # Log runtime (được ignore)
├── notebooks/
│   └── kaggle-t4x2-text-vision.ipynb
├── scripts/                    # Setup, lifecycle, test, tunnel, benchmark
├── src/                        # Flask coordinator + TranslateGemma workers
├── state/                      # PID/trạng thái worker runtime (được ignore)
├── tests/                      # Unit test thân thiện với CPU
├── .env.example
├── .gitignore
├── CHANGELOG.md / CHANGELOG.vi.md
├── CODE_OF_CONDUCT.md / CONTRIBUTING.md / SECURITY.md (EN/VI)
├── LICENSE
├── NOTICE.md / THIRD_PARTY_NOTICES.md (EN/VI)
├── README.md / README.vi.md
├── constraints-kaggle-tested.txt
└── requirements.txt
```

## Thiết lập thủ công trong terminal Kaggle

Nếu bạn muốn dùng terminal thay cho notebook:

```bash
git clone https://github.com/dangkhoa2016/KerasHub-TranslateGemma-4B-IT-Kaggle-GPU-T4x2-Text-Vision.git
cd KerasHub-TranslateGemma-4B-IT-Kaggle-GPU-T4x2-Text-Vision
cp .env.example .env
INSTALL_PYTHON_DEPS=1 bash scripts/setup.sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
bash scripts/start.sh
```

Kiểm tra trạng thái:

```bash
bash scripts/status.sh
```

Dừng máy chủ:

```bash
bash scripts/stop.sh
```

## Xác thực API

Xác thực được bật mặc định. Trong lần khởi động đầu tiên, coordinator tạo các giá trị ngẫu nhiên cục bộ tại:

```text
data/api_key.txt
data/restart_secret.txt
```

Cả hai đường dẫn đều được Git ignore. Không commit các file này.

Đọc API key cục bộ:

```bash
API_KEY="$(cat data/api_key.txt)"
```

## Dịch văn bản

```bash
API_KEY="$(cat data/api_key.txt)"

curl -X POST http://127.0.0.1:7860/translate \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "source_lang": "English",
    "target_lang": "Vietnamese",
    "max_new_tokens": 256
  }'
```

Với các job chạy nền, dùng `POST /translate/async` và kiểm tra `GET /result/<job_id>`.

## Dịch hình ảnh

Vision được bật mặc định trong `.env.example`.

```bash
bash scripts/test_vision.sh assets/sample-image-with-text.jpg
```

API endpoint là `POST /translate/image`; biến thể bất đồng bộ có tại `POST /translate/image/async`.

Hình ảnh được gửi dưới dạng base64 trong body JSON. Trước khi giải nén bất kỳ dữ liệu pixel nào, máy chủ kiểm tra kích thước byte đã giải mã và kích thước header theo `MAX_IMAGE_BYTES`, `MAX_IMAGE_WIDTH`, `MAX_IMAGE_HEIGHT` và `MAX_IMAGE_PIXELS`, nhờ đó cũng chặn được decompression bomb. Định dạng hỗ trợ: JPEG, PNG, WEBP, BMP, TIFF, GIF.

## Health endpoints

```text
GET /health/live
GET /health/ready
GET /health/ready?all=1
GET /health/ready?all=1&details=1
```

Thông tin readiness chi tiết yêu cầu API key khi xác thực được bật.

## Thiết kế T4x2

Một process Keras/JAX đơn lẻ không thể đơn giản trải mô hình lên hai GPU T4 này nếu không thay đổi chiến lược thực thi mô hình. Thay vào đó, dự án chạy hai worker độc lập:

```text
Flask coordinator
   ├── worker gpu:0 -> CUDA_VISIBLE_DEVICES=0 -> TranslateGemma 4B IT
   └── worker gpu:1 -> CUDA_VISIBLE_DEVICES=1 -> TranslateGemma 4B IT
```

Request được xếp hàng tập trung và phân phối cho worker đang rảnh. Vì vậy hai job dịch đồng thời có thể chạy cùng lúc trên hai GPU riêng biệt.

## Chiến lược biên dịch JAX

Cấu hình mặc định sử dụng:

```text
GENERATION_LENGTH_BUCKETS=256,512,1024,1536,2048
WARMUP_TEXT_BUCKETS=256
WARMUP_VISION_BUCKETS=512
JAX_COMPILATION_CACHE_DIR=/kaggle/working/.cache/translategemma-jax
WORKER_START_MODE=auto
```

Generation bucketing tránh biên dịch một JAX executable mới cho mỗi độ dài prompt chỉ khác nhau một chút. Chính sách khởi động thích ứng sử dụng staggered launch khi cache lạnh/rỗng, sau đó có thể khởi động song song cả hai worker khi cache đã warm và host còn đủ RAM.

## Benchmark từ lần chạy Kaggle T4x2 đã được xác thực

Các số đo đại diện từ phiên xác thực nguồn được tóm tắt trong [`docs/BENCHMARKS.vi.md`](docs/BENCHMARKS.vi.md). Đây là số liệu tham khảo, không phải cam kết hiệu năng.

## Public tunnel tùy chọn

Sau khi local API sẵn sàng:

```bash
bash scripts/run_tunnel.sh
cat data/tunnel_url.txt
```

Script sử dụng Cloudflare Quick Tunnel tạm thời. Hãy coi URL là tạm thời và luôn bật xác thực API.

## Kiểm thử

Unit test có thể chạy trên CPU:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Test văn bản với mô hình thật:

```bash
bash scripts/test.sh data/input.example.txt
```

Test vision với mô hình thật:

```bash
bash scripts/test_vision.sh assets/sample-image-with-text.jpg
```

Benchmark hai request đồng thời trên T4x2:

```bash
bash scripts/test_concurrency.sh
```

Benchmark dtype tùy chọn:

```bash
RUN_DTYPE_BENCHMARK=1 bash scripts/benchmark_dtype.sh
```

Benchmark startup/cache tùy chọn:

```bash
RUN_STARTUP_BENCHMARK=1 bash scripts/benchmark_startup.sh
```

## Ghi chú bảo mật

- Không bao giờ commit `.env`, key được sinh ra, tunnel URL, runtime state, log hoặc binary đã tải.
- Giữ `API_AUTH_REQUIRED=true` khi expose API qua tunnel.
- Restart endpoint dùng một restart secret riêng.
- Hãy xem phiên Kaggle notebook và mọi public tunnel là tài nguyên tính toán tạm thời, không phải hạ tầng production bền vững.

## Giấy phép

Code gốc của repository này được cấp phép theo **MIT License**. Xem [`LICENSE`](LICENSE). File `LICENSE` tiếng Anh là văn bản license chính thức.

TranslateGemma/Gemma, KerasHub và các thành phần bên thứ ba khác vẫn chịu các giấy phép và điều khoản riêng. Xem [`NOTICE.vi.md`](NOTICE.vi.md) và [`THIRD_PARTY_NOTICES.vi.md`](THIRD_PARTY_NOTICES.vi.md).
