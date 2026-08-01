# Benchmark tham khảo

> 🌐 Language / Ngôn ngữ: [English](BENCHMARKS.md) | **Tiếng Việt**

Các số đo này đến từ phiên Kaggle T4x2 đã được xác thực và được dùng làm nguồn cho public repository này. Mục đích là chứng minh cả hai GPU T4 đều hoạt động và ghi lại hành vi startup/cache. Đây không phải cam kết hiệu năng; Kaggle image, tải hệ thống, phiên bản mô hình và shape của prompt có thể làm thay đổi kết quả.

## Hai request đồng thời

Cấu hình: BF16, bật generation bucketing, ngân sách 128 output token, hai request văn bản giống nhau chạy đồng thời.

| Giai đoạn | Wall time cho cả hai request | CPU trung bình | RAM lấy mẫu cao nhất | GPU 0 util TB | GPU 1 util TB |
|---|---:|---:|---:|---:|---:|
| PRIME | 4.271 s | 51.99% | 6039.8 MiB | 100% | 100% |
| HOT | 4.176 s | 53.79% | 6050.9 MiB | 100% | 100% |

Cả hai giai đoạn đều báo cáo worker `gpu:0` và `gpu:1`, xác nhận cặp request đã chạy trên cả hai thiết bị.

## Giai đoạn nóng BF16 so với FP16

| Dtype | PRIME | HOT |
|---|---:|---:|
| bfloat16 | 4.229 s | 4.198 s |
| float16 | 4.248 s | 4.207 s |

Chênh lệch đo được trong workload này không đáng kể, vì vậy BF16 vẫn là mặc định.

## Startup và JAX cache bền vững

| Kịch bản | Thời gian đến khi 2/2 worker sẵn sàng |
|---|---:|
| Cache lạnh + staggered startup | 250.38 s |
| Cache warm + staggered startup | 201.45 s |
| Cache warm + parallel startup | 103.12 s |
| Khôi phục auto mode mặc định với cache warm | 102.02 s |

Đây là lý do cho chính sách khởi động thích ứng mặc định: dùng startup thận trọng khi compilation cache còn lạnh, sau đó chuyển sang parallel startup khi cache đã warm và host còn đủ bộ nhớ.
