# Đóng góp

> 🌐 Language / Ngôn ngữ: [English](CONTRIBUTING.md) | **Tiếng Việt**

Cảm ơn bạn đã đóng góp. Repository ưu tiên các thay đổi giúp dự án chạy ổn định trên Kaggle GPU T4x2, duy trì khả năng tái lập và không bao giờ làm lộ mô hình hoặc credential.

## Quy trình đề xuất

1. Fork repository và tạo một branch riêng cho thay đổi của bạn.
2. Giữ mỗi pull request tập trung vào một mục tiêu rõ ràng.
3. Cập nhật README/CHANGELOG khi hành vi hoặc cấu hình thay đổi.
4. Chạy các kiểm tra trước khi gửi pull request:

```bash
python3 -m py_compile src/server.py src/translategemma/*.py src/translategemma/*/*.py scripts/*.py tests/*.py
for file in scripts/*.sh tests/*.sh; do bash -n "$file"; done
bash tests/test_setup_env.sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Một số integration test cần Flask, GPU và checkpoint được đính kèm trong Kaggle. Hãy ghi rõ test nào bạn không thể chạy và lý do.

## Quy tắc nội dung

Không commit:

- Model weights, tokenizer cache hoặc thư mục cache Hugging Face/Keras.
- `.env`, API key, restart secret, Hugging Face token hoặc Cloudflare credential.
- Binary đã tải, file log, PID hoặc tunnel URL.
- Dataset riêng tư hoặc nội dung người dùng chưa được phép chia sẻ.

## Pull request

Mô tả pull request nên nêu rõ:

- Vấn đề mà thay đổi giải quyết.
- Thay đổi kiến trúc hoặc cấu hình.
- Kết quả test trên CPU và, nếu có thể, Kaggle T4x2.
- Ảnh hưởng đến RAM/VRAM, thời gian khởi động và khả năng tương thích ngược.

## Metadata repository

Thiết lập mục "About" trên GitHub cho repository này như sau:

Mô tả:

```text
Run KerasHub TranslateGemma 4B IT on Kaggle T4x2 with dual-GPU workers, text/image translation, JAX compilation cache and Flask REST API.
```

Topics:

```text
translategemma gemma keras keras-hub jax kaggle gpu nvidia-t4 multimodal translation image-translation flask rest-api python
```

Khi gửi đóng góp, bạn đồng ý rằng phần đóng góp của mình được cấp phép theo MIT License của repository.
