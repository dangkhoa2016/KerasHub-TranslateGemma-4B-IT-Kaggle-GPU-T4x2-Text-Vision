# Contributing

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CONTRIBUTING.vi.md)

Thank you for contributing. The repository prioritizes changes that help it run reliably on Kaggle GPU T4x2, preserve reproducibility, and never leak the model or credentials.

## Proposal process

1. Fork the repository and create a dedicated branch for your change.
2. Keep each pull request focused on a single clear goal.
3. Update the README/CHANGELOG when behavior or configuration changes.
4. Run the checks before submitting a pull request:

```bash
python3 -m py_compile src/server.py src/translategemma/*.py src/translategemma/*/*.py scripts/*.py tests/*.py
for file in scripts/*.sh tests/*.sh; do bash -n "$file"; done
bash tests/test_setup_env.sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Some integration tests need Flask, a GPU, and the checkpoint attached in Kaggle. Note which tests you could not run and why.

## Content rules

Do not commit:

- Model weights, tokenizer caches, or Hugging Face/Keras cache directories.
- `.env`, API keys, restart secrets, Hugging Face tokens, or Cloudflare credentials.
- Downloaded binaries, log files, PIDs, or tunnel URLs.
- Private datasets or user content not authorized for sharing.

## Pull request

A pull request description should state:

- The problem it solves.
- Architecture or configuration changes.
- Test results on CPU and, if possible, Kaggle T4x2.
- Impact on RAM/VRAM, startup time, and backward compatibility.

## Repository metadata

Set the GitHub "About" section for this repository to:

Description:

```text
Run KerasHub TranslateGemma 4B IT on Kaggle T4x2 with dual-GPU workers, text/image translation, JAX compilation cache and Flask REST API.
```

Topics:

```text
translategemma gemma keras keras-hub jax kaggle gpu nvidia-t4 multimodal translation image-translation flask rest-api python
```

By submitting a contribution, you agree that your contribution is licensed under the repository's MIT License.
