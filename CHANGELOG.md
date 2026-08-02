# Changelog

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

## 1.0.0 — Initial public release

- Run the TranslateGemma 4B IT multimodal model on a Kaggle GPU T4 x2.
- Start one isolated model worker process per GPU behind a lightweight Flask coordinator.
- Expose synchronous and asynchronous text and image translation endpoints.
- Authenticate requests with an API key and a restart secret generated locally on first start.
- Validate payloads strictly and retain results in a bounded store with a TTL.
- Stabilize JAX compilation with generation-length bucketing and warm-up.
- Keep a persistent JAX compilation cache outside the repository working tree.
- Adapt worker startup to the JAX cache state and available host RAM.
- Monitor worker health, restart crashed workers, and clean up stale processes.
- Provide unit tests, text/vision smoke tests, and concurrency, dtype, and startup benchmarks.
- Ship a repository-backed Kaggle notebook that needs no separate project bundle.
- Document the project in English and Vietnamese with language-switch links.
