# KerasHub TranslateGemma 4B IT on Kaggle T4x2 — Text + Vision

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

A public, reproducible Kaggle project for running **TranslateGemma 4B IT** with **KerasHub + JAX** on the **Kaggle GPU T4 x2** accelerator.

The server uses both T4 GPUs by starting **one isolated model worker per GPU** behind a lightweight Flask coordinator. It supports normal text translation and image-to-text translation (OCR + translation) through the same multimodal checkpoint.

**Author:** Đăng Khoa <i.am@dangkhoa.dev>

## Highlights

- KerasHub `translategemma_4b_it` multimodal model.
- Two independent GPU workers: GPU 0 and GPU 1.
- Text translation and image translation endpoints.
- Synchronous and asynchronous job APIs.
- Adaptive worker startup based on JAX cache state and available host RAM.
- JAX persistent compilation cache and generation-length bucketing.
- Warm-up for common text and vision generation shapes.
- API-key authentication generated locally on first start.
- Optional Cloudflare Quick Tunnel for temporary public access.
- Unit tests, text/vision smoke tests, T4x2 concurrency benchmark, dtype benchmark, and startup/cache benchmark.
- A Kaggle notebook stored directly in this repository; no separate project bundle is required.
- English and Vietnamese documentation with language-switch links.

## Model and hardware

The application is tuned for the Kaggle **GPU T4 x2** environment. Each worker loads one complete multimodal TranslateGemma 4B IT checkpoint on one T4, so vision mode uses nearly the full 16 GB VRAM available on each GPU.

The KerasHub preset is `translategemma_4b_it`. Keras documents it as a 4.30B-parameter multimodal translation model supporting text and image input. Official references: [Kaggle TranslateGemma model](https://www.kaggle.com/models/keras/translategemma) and [KerasHub Gemma3CausalLM presets](https://keras.io/keras_hub/api/models/gemma3/gemma3_causal_lm/). The repository defaults to the Kaggle-mounted Keras model path:

```text
/kaggle/input/models/keras/translategemma/keras/translategemma_4b_it/1
```

If Kaggle mounts a different version directory, `MODEL_AUTO_DISCOVER=true` searches the attached model versions automatically.

## Recommended Kaggle workflow

### 1. Import the notebook directly from GitHub

On Kaggle, open the notebook import dialog and choose **GitHub**. Select this repository and import:

```text
notebooks/kaggle-t4x2-text-vision.ipynb
```

This is the preferred workflow because Kaggle loads all notebook cells directly from the repository.

### 2. Configure the Kaggle session

Before running the notebook:

1. Set **Accelerator → GPU T4 x2**.
2. Enable **Internet** so the first cell can clone/update the repository and `scripts/setup.sh` can download `cloudflared` when needed.
3. Add the Keras **TranslateGemma** model as a Kaggle Model input if it is not already attached.

### 3. Run the notebook from the first cell

The first cell clones this repository into:

```text
/kaggle/working/KerasHub-TranslateGemma-4B-IT-Kaggle-GPU-T4x2-Text-Vision
```

Re-running that cell updates the working copy to the current `main` branch. The following cells create `.env` from `.env.example`, validate the T4x2 setup, install/check dependencies, run unit tests, start both GPU workers, and run text/vision tests and benchmarks.

> `git clone` alone cannot replace the currently open Kaggle notebook or inject cells into it. Import the repository notebook first; the clone cell then provides the source tree used by those already-loaded cells.

## Repository layout

```text
.
├── .github/                    # CI and GitHub templates
├── assets/                     # Public vision test asset
├── bin/                        # Runtime-downloaded cloudflared binary (ignored)
├── data/                       # Public input example; generated secrets are ignored
├── docs/                       # Architecture, Kaggle usage, and benchmark notes (EN/VI)
├── log/                        # Runtime logs (ignored)
├── notebooks/
│   └── kaggle-t4x2-text-vision.ipynb
├── scripts/                    # Setup, lifecycle, tests, tunnel, benchmarks
├── src/                        # Flask coordinator + TranslateGemma workers
├── state/                      # Runtime PID/worker state (ignored)
├── tests/                      # CPU-friendly unit tests
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

## Manual setup in a Kaggle terminal

If you prefer the terminal instead of the notebook:

```bash
git clone https://github.com/dangkhoa2016/KerasHub-TranslateGemma-4B-IT-Kaggle-GPU-T4x2-Text-Vision.git
cd KerasHub-TranslateGemma-4B-IT-Kaggle-GPU-T4x2-Text-Vision
cp .env.example .env
INSTALL_PYTHON_DEPS=1 bash scripts/setup.sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
bash scripts/start.sh
```

Check status:

```bash
bash scripts/status.sh
```

Stop the server:

```bash
bash scripts/stop.sh
```

## API authentication

Authentication is enabled by default. On first start, the coordinator creates random local values in:

```text
data/api_key.txt
data/restart_secret.txt
```

Both paths are ignored by Git. Do not commit them.

Read the API key locally:

```bash
API_KEY="$(cat data/api_key.txt)"
```

## Text translation

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

For background jobs, use `POST /translate/async` and poll `GET /result/<job_id>`.

## Image translation

Vision is enabled by default in `.env.example`.

```bash
bash scripts/test_vision.sh assets/sample-image-with-text.jpg
```

The API endpoint is `POST /translate/image`; an asynchronous variant is available at `POST /translate/image/async`.

Images are submitted as base64 in the JSON body. Before decoding any pixel data, the server validates the decoded byte size and the header dimensions against `MAX_IMAGE_BYTES`, `MAX_IMAGE_WIDTH`, `MAX_IMAGE_HEIGHT`, and `MAX_IMAGE_PIXELS`, which also guards against decompression bombs. Supported formats: JPEG, PNG, WEBP, BMP, TIFF, GIF.

## Health endpoints

```text
GET /health/live
GET /health/ready
GET /health/ready?all=1
GET /health/ready?all=1&details=1
```

Detailed readiness information requires the API key when authentication is enabled.

## T4x2 design

A single Keras/JAX process cannot simply span these two T4 GPUs without changing the model execution strategy. This project instead runs two isolated workers:

```text
Flask coordinator
   ├── worker gpu:0 -> CUDA_VISIBLE_DEVICES=0 -> TranslateGemma 4B IT
   └── worker gpu:1 -> CUDA_VISIBLE_DEVICES=1 -> TranslateGemma 4B IT
```

Requests are queued centrally and distributed to available workers. Two concurrent translation jobs can therefore execute at the same time on separate GPUs.

## JAX compilation strategy

The default configuration uses:

```text
GENERATION_LENGTH_BUCKETS=256,512,1024,1536,2048
WARMUP_TEXT_BUCKETS=256
WARMUP_VISION_BUCKETS=512
JAX_COMPILATION_CACHE_DIR=/kaggle/working/.cache/translategemma-jax
WORKER_START_MODE=auto
```

Generation bucketing avoids compiling a new JAX executable for every slightly different prompt length. The adaptive startup policy uses a staggered launch for a cold/empty cache, then can start both workers in parallel when the cache is warm and sufficient host RAM is available.

## Benchmarks from the validated Kaggle T4x2 run

Representative measurements from the source validation session are summarized in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). They are reference measurements, not performance guarantees.

## Optional public tunnel

After the local API is ready:

```bash
bash scripts/run_tunnel.sh
cat data/tunnel_url.txt
```

The script uses a temporary Cloudflare Quick Tunnel. Treat the URL as ephemeral and keep API authentication enabled.

## Tests

CPU-friendly unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Real-model text test:

```bash
bash scripts/test.sh data/input.example.txt
```

Real-model vision test:

```bash
bash scripts/test_vision.sh assets/sample-image-with-text.jpg
```

Two-request T4x2 concurrency benchmark:

```bash
bash scripts/test_concurrency.sh
```

Optional dtype benchmark:

```bash
RUN_DTYPE_BENCHMARK=1 bash scripts/benchmark_dtype.sh
```

Optional startup/cache benchmark:

```bash
RUN_STARTUP_BENCHMARK=1 bash scripts/benchmark_startup.sh
```

## Security notes

- Never commit `.env`, generated keys, tunnel URLs, runtime state, logs, or downloaded binaries.
- Keep `API_AUTH_REQUIRED=true` when exposing the API through a tunnel.
- The restart endpoint uses a separate restart secret.
- Treat a Kaggle notebook session and any public tunnel as temporary compute, not as durable production infrastructure.

## License

The original code in this repository is licensed under the **MIT License**. See [`LICENSE`](LICENSE).

TranslateGemma/Gemma, KerasHub, and other third-party components remain subject to their own licenses and terms. See [`NOTICE.md`](NOTICE.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
