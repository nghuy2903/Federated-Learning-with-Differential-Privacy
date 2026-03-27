📑 QUẢN LÝ DỰ ÁN: FL + DP SECURITY SYSTEM
1. Thông tin chung
Đề tài: Nghiên cứu mô hình kết hợp Federated Learning (FL) và Differential Privacy (DP) để xây dựng hệ thống bảo mật dữ liệu trên nền tảng số.
Thời hạn hoàn thành: ~1.5 tháng (Dự kiến kết thúc vào cuối tháng 04/2026).
Đội ngũ: 4 thành viên (Trưởng nhóm phụ trách AI; 3 thành viên hỗ trợ hạ tầng, dữ liệu và giao diện).
2. Mục tiêu kỹ thuậtKiến trúc: Federated Learning (Flower Framework) với kịch bản Horizontal FL.
Bảo mật: Differential Privacy (Thư viện Opacus) tích hợp vào quá trình huấn luyện cục bộ (Local Training).
Dữ liệu: MNIST chia theo kịch bản Non-IID (dữ liệu không đồng nhất giữa các máy khách).
Môi trường thực thi: 
* Giai đoạn đầu: Mô phỏng 1 Server - N Clients trên 1 máy (Localhost).
* Giai đoạn cuối: Triển khai trên mạng lưới 4 máy tính vật lý thật.
3. Cấu hình hệ thống & Môi trườngPhần cứng: Máy chủ/Máy khách có Card NVIDIA (Hỗ trợ CUDA).
Quản lý môi trường: Miniconda (Python 3.10).
Thư viện lõi: * torch, torchvision (Deep Learning).flwr (Federated Learning).opacus (Differential Privacy).
Môi trường Conda: fl_dp_env.
4. Kế hoạch triển khai (Roadmap 6 tuần)
🟢 Giai đoạn 1: Thiết lập & Nghiên cứu lý thuyết (Tuần 1 - Đã xong)
[x] Chốt đề tài và báo cáo tiến độ với hội đồng.
[x] Thiết lập cấu trúc thư mục dự án.
[x] Cài đặt môi trường Miniconda, PyTorch CUDA và các thư viện liên quan.
🟡 Giai đoạn 2: Huấn luyện mô phỏng (Simulation) (Tuần 2-3 - Đang thực hiện)
[x] Viết mã nguồn cho model.py (CNN).
[x] Hoàn thiện utils.py để chia dữ liệu MNIST theo dạng Non-IID.
[x] Chạy thông luồng Federated Learning cơ bản (1 Server - 2 Clients giả lập).
[x] Tích hợp Opacus vào Client để thực hiện bảo mật DP-SGD.
[x] Kiểm chứng sự thay đổi của Accuracy khi thay đổi ngân sách riêng tư $\epsilon$.
⚪ Giai đoạn 3: Triển khai hệ thống vật lý (Tuần 4)
[ ] Cấu hình mạng LAN cho 4 máy tính.
[ ] Đồng bộ hóa môi trường Python trên các máy thành viên.
[ ] Thực hiện huấn luyện liên hợp thực tế trên mạng lưới thiết bị thật.
⚪ Giai đoạn 4: Đánh giá & Thu thập dữ liệu (Tuần 5)
[ ] Chạy thực nghiệm với nhiều mức $\epsilon$ khác nhau.
[ ] Vẽ biểu đồ so sánh: Accuracy vs Privacy Budget, Convergence Rate.
[ ] Xây dựng Dashboard hiển thị kết quả thời gian thực (Streamlit).
⚪ Giai đoạn 5: Hoàn thiện & Đóng gói (Tuần 6)[ ] Viết báo cáo tổng kết NCKH.
[ ] Chuẩn bị kịch bản Demo và Slide bảo vệ cuối cùng.
5. Phân công công việc (Team 4 người)
Trưởng nhóm (AI Lead): Thiết kế thuật toán, tích hợp DP, xử lý lỗi core.
Thành viên 2 (Data Specialist): Tiền xử lý MNIST, viết script chia Non-IID, quản lý logs.
Thành viên 3 (Infrastructure Lead): Thiết lập mạng LAN, quản lý kết nối 4 máy, cấu hình IP.
Thành viên 4 (UI & Reporter): Làm Dashboard Streamlit, viết báo cáo và vẽ biểu đồ kết quả.
6. Ghi chú quan trọng
Luôn kích hoạt môi trường conda activate fl_dp_env trước khi chạy code.
Khi huấn luyện trên 1 máy (Simulation), dùng địa chỉ 127.0.0.1:8080.
Khi triển khai thật, cần cập nhật IP Server vào file client.py.