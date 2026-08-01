> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](pull_request_template.vi.md)

## Goal

Describe the problem and the main changes.

## Validation

- [ ] `python3 -m compileall -q src tests scripts`
- [ ] `bash -n` passes for every `scripts/*.sh`
- [ ] Unit tests were run, or skipped tests are explained
- [ ] `python3 scripts/validate_public_repo.py` passes
- [ ] No model weights, caches, logs, downloaded binaries, or credentials are included
- [ ] README/CHANGELOG were updated when behavior changed
- [ ] English/Vietnamese Markdown pairs remain synchronized when documentation changed

## Operational impact

Describe any effect on Kaggle T4x2, RAM/VRAM, startup time, and API compatibility.
