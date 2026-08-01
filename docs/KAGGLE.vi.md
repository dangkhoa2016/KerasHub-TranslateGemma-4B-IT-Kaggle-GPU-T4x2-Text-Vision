# Sử dụng Kaggle

> 🌐 Language / Ngôn ngữ: [English](KAGGLE.md) | **Tiếng Việt**

## Import notebook

Dùng hộp thoại import notebook của Kaggle và chọn nguồn **GitHub**. Chọn repository này và import `notebooks/kaggle-t4x2-text-vision.ipynb`.

Bản thân notebook chứa toàn bộ cell điều phối. Cell đầu tiên clone hoặc làm mới repository dưới `/kaggle/working`, nhờ đó các cell còn lại có một source tree bình thường để thực thi.

Một shell command như `git clone` chỉ tạo file trong notebook runtime; nó không khiến Kaggle thay notebook đang mở bằng một file `.ipynb` khác. Vì vậy GitHub notebook import và repository cloning được dùng cùng nhau.

## Cấu hình phiên bắt buộc

- Accelerator: **GPU T4 x2**
- Internet: **On**
- Model input: Keras TranslateGemma với preset `translategemma_4b_it`

## Chạy lại sau khi repository được cập nhật

Chạy lại cell đầu tiên của notebook. Khi Git working copy đã tồn tại, cell sẽ fetch `origin/main` và reset working tree về revision đó. Các file runtime cục bộ được sinh ra vẫn bị Git ignore; notebook sẽ tạo lại `.env` từ `.env.example` khi cần.

## Tự động tìm mô hình

Đường dẫn mount mặc định là `/kaggle/input/models/keras/translategemma/keras/translategemma_4b_it/1`. Nếu đúng thư mục phiên bản này không tồn tại, `scripts/setup.sh` sẽ tìm các version directory đã được đính kèm khi `MODEL_AUTO_DISCOVER=true`.
