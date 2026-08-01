# Architecture

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](ARCHITECTURE.vi.md)

## Process model

The application separates the HTTP coordinator from model workers. Each model worker receives an explicit GPU assignment and loads its own KerasHub/JAX model instance.

```text
Client
  |
  v
Flask coordinator / job store
  |---------------------------|
  v                           v
worker gpu:0                worker gpu:1
CUDA device 0               CUDA device 1
TranslateGemma 4B IT        TranslateGemma 4B IT
```

This process isolation is intentional: it provides predictable T4 memory ownership and allows two requests to run concurrently without requiring a multi-device sharding implementation.

## Request lifecycle

1. The coordinator validates the payload.
2. A job enters the bounded queue.
3. An available worker claims the job.
4. The worker performs text or multimodal generation.
5. The result is written to the in-memory job store.
6. Synchronous requests return immediately when ready or return a job identifier if the request timeout is exceeded.

## Generation-shape stabilization

JAX compilation cost is sensitive to input/output shapes. The worker maps requested total generation lengths onto a limited set of configured buckets. Common text and vision buckets can be compiled during warm-up, and JAX's persistent compilation cache is kept under `/kaggle/working/.cache` rather than inside the Git working tree.

## Adaptive startup

With `WORKER_START_MODE=auto`:

- a cold/insufficient cache favors staggered startup to reduce simultaneous compilation pressure;
- a warm cache plus sufficient available host RAM allows parallel startup of both workers.

The startup policy is visible in detailed health metadata.
