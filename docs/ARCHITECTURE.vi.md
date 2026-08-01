# Kiến trúc

> 🌐 Language / Ngôn ngữ: [English](ARCHITECTURE.md) | **Tiếng Việt**

## Mô hình process

Ứng dụng tách HTTP coordinator khỏi các model worker. Mỗi model worker được gán GPU rõ ràng và tải instance KerasHub/JAX riêng.

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

Việc cô lập process này là có chủ đích: nó giúp quyền sở hữu bộ nhớ T4 có thể dự đoán được và cho phép hai request chạy đồng thời mà không cần triển khai sharding đa thiết bị.

## Vòng đời request

1. Coordinator xác thực payload.
2. Một job được đưa vào bounded queue.
3. Một worker đang rảnh nhận job.
4. Worker thực hiện sinh văn bản hoặc sinh đa phương thức.
5. Kết quả được ghi vào in-memory job store.
6. Request đồng bộ trả về ngay khi sẵn sàng hoặc trả về job identifier nếu vượt quá request timeout.

## Ổn định shape sinh dữ liệu

Chi phí biên dịch JAX nhạy với shape đầu vào/đầu ra. Worker ánh xạ tổng độ dài generation được yêu cầu vào một tập bucket cấu hình giới hạn. Các bucket text và vision phổ biến có thể được biên dịch trong warm-up, còn JAX persistent compilation cache được lưu dưới `/kaggle/working/.cache` thay vì bên trong Git working tree.

## Khởi động thích ứng

Với `WORKER_START_MODE=auto`:

- cache lạnh/không đủ ưu tiên staggered startup để giảm áp lực biên dịch đồng thời;
- cache đã warm cùng với host RAM còn đủ cho phép khởi động song song cả hai worker.

Chính sách khởi động có thể xem trong health metadata chi tiết.
