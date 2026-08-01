# Reference benchmarks

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](BENCHMARKS.vi.md)

These measurements came from the validated Kaggle T4x2 session used as the source for this public repository. They are intended to demonstrate that both T4 GPUs were active and to document the startup/cache behavior. They are not performance guarantees; Kaggle images, load, model versions, and prompt shapes can change results.

## Two-request concurrency

Configuration: BF16, generation bucketing enabled, 128 output-token budget, two identical concurrent text requests.

| Phase | Wall time for both requests | Average CPU | Peak sampled RAM | GPU 0 avg util | GPU 1 avg util |
|---|---:|---:|---:|---:|---:|
| PRIME | 4.271 s | 51.99% | 6039.8 MiB | 100% | 100% |
| HOT | 4.176 s | 53.79% | 6050.9 MiB | 100% | 100% |

Both phases reported workers `gpu:0` and `gpu:1`, confirming that the pair of requests ran across both devices.

## BF16 vs FP16 hot phase

| Dtype | PRIME | HOT |
|---|---:|---:|
| bfloat16 | 4.229 s | 4.198 s |
| float16 | 4.248 s | 4.207 s |

The measured difference was negligible in this workload, so BF16 remains the default.

## Startup and persistent JAX cache

| Scenario | Time until 2/2 workers ready |
|---|---:|
| Cold cache + staggered startup | 250.38 s |
| Warm cache + staggered startup | 201.45 s |
| Warm cache + parallel startup | 103.12 s |
| Restored default auto mode with warm cache | 102.02 s |

This is the motivation for the default adaptive startup policy: use conservative startup while the compilation cache is cold, then use parallel startup when the cache is warm and host memory is sufficient.
