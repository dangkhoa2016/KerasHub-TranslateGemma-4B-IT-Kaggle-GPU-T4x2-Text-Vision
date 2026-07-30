# Chính sách bảo mật

> 🌐 Language / Ngôn ngữ: [English](SECURITY.md) | **Tiếng Việt**

## Phạm vi được hỗ trợ

Bản vá bảo mật chỉ được áp dụng cho phiên bản mới nhất trên branch mặc định của repository. Các commit cũ có thể không được cập nhật.

## Báo cáo lỗ hổng

Không đăng API key, restart secret, private tunnel URL, Hugging Face token hoặc chi tiết khai thác nhạy cảm trong public issue.

Ưu tiên dùng **GitHub → Security → Advisories → Report a vulnerability** để gửi báo cáo riêng cho chủ repository. Nếu Private Vulnerability Reporting chưa được bật, hãy liên hệ riêng với **Đăng Khoa <i.am@dangkhoa.dev>**.

Một báo cáo hữu ích nên bao gồm:

- Phiên bản hoặc commit bị ảnh hưởng.
- Cấu hình triển khai liên quan, sau khi đã loại bỏ mọi secret.
- Các bước tái hiện tối thiểu.
- Ảnh hưởng dự kiến và đề xuất sửa lỗi, nếu có.

## Xử lý credential bị lộ

Nếu credential từng được commit, chỉ xóa file ở một commit mới là chưa đủ. Thay vào đó:

1. Thu hồi hoặc xoay vòng credential ngay lập tức.
2. Xóa credential khỏi toàn bộ Git history khi cần.
3. Kiểm tra access log và mọi secret liên quan.
4. Bật GitHub secret scanning và push protection.

Các file runtime như `.env`, `data/api_key.txt`, `data/restart_secret.txt`, log và cache đã được `.gitignore` bao phủ, nhưng người vận hành vẫn phải kiểm tra trước mỗi lần push.

## Giới hạn bảo mật

Dự án được thiết kế cho demo/thử nghiệm trên Kaggle, không phải hạ tầng production quy mô lớn. Cloudflare Quick Tunnel tạo một URL công khai; hãy luôn bật xác thực API và không bao giờ coi một URL ngẫu nhiên là cơ chế bảo mật.
