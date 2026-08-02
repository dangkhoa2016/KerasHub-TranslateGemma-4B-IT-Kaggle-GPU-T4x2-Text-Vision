# Nhật ký thay đổi

> 🌐 Language / Ngôn ngữ: [English](CHANGELOG.md) | **Tiếng Việt**

## 1.0.0 — Bản phát hành public đầu tiên

- Chạy mô hình đa phương thức TranslateGemma 4B IT trên Kaggle GPU T4 x2.
- Khởi động một worker model tách biệt trên mỗi GPU, đứng sau bộ điều phối Flask gọn nhẹ.
- Cung cấp API dịch văn bản và hình ảnh theo chế độ đồng bộ lẫn bất đồng bộ.
- Xác thực yêu cầu bằng API key và restart secret được tạo cục bộ ngay lần chạy đầu tiên.
- Kiểm tra payload chặt chẽ và lưu kết quả trong store giới hạn kèm TTL.
- Ổn định quá trình biên dịch JAX bằng cách phân nhóm độ dài sinh token và warm-up.
- Giữ cache biên dịch JAX bền vững nằm ngoài cây làm việc của repository.
- Thích ứng cách khởi động worker theo trạng thái JAX cache và RAM khả dụng.
- Theo dõi sức khỏe worker, khởi động lại worker bị sập và dọn dẹp tiến trình mồ côi.
- Cung cấp unit test, smoke test văn bản/hình ảnh cùng benchmark độ đồng thời, dtype và khởi động.
- Cung cấp notebook Kaggle dựa trên repository, không cần bundle riêng.
- Soạn tài liệu song ngữ Anh – Việt kèm liên kết chuyển ngôn ngữ.
