> 🌐 Language / Ngôn ngữ: [English](pull_request_template.md) | **Tiếng Việt**

## Mục tiêu

Mô tả vấn đề và các thay đổi chính.

## Xác thực

- [ ] `python3 -m compileall -q src tests scripts`
- [ ] `bash -n` thành công với mọi `scripts/*.sh`
- [ ] Đã chạy unit test, hoặc đã giải thích các test bị bỏ qua
- [ ] `python3 scripts/validate_public_repo.py` thành công
- [ ] Không bao gồm model weights, cache, log, binary đã tải hoặc credential
- [ ] README/CHANGELOG đã được cập nhật khi hành vi thay đổi
- [ ] Các cặp Markdown English/Tiếng Việt vẫn đồng bộ khi tài liệu thay đổi

## Ảnh hưởng vận hành

Mô tả mọi ảnh hưởng đến Kaggle T4x2, RAM/VRAM, thời gian khởi động và khả năng tương thích API.
