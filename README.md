# Federated Learning + Differential Privacy (MNIST)

![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-ff6f00?logo=flower&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)
![Opacus](https://img.shields.io/badge/Opacus-Differential%20Privacy-6a1b9a)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ed?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit&logoColor=white)

## 1) Project Overview

Project này triển khai **Federated Learning (FL)** cho bài toán phân loại ảnh số viết tay **MNIST** với kiến trúc **1 Central Server + 4 Clients**.  
Mỗi client huấn luyện mô hình cục bộ trên dữ liệu riêng, chỉ gửi cập nhật trọng số về server thay vì chia sẻ dữ liệu gốc.

Để tăng tính riêng tư, hệ thống tích hợp **Differential Privacy (DP)** bằng **Opacus** trong quá trình local training. Cách tiếp cận này giúp giảm rủi ro rò rỉ thông tin từ gradient/model updates trong khi vẫn duy trì hiệu quả học liên hợp.

---

## 2) Architecture Diagram

```text
                       +----------------------+
                       |   Central FL Server  |
                       |      (Flower)        |
                       +----------+-----------+
                                  ^
                                  | Aggregated model updates
                                  |
        ---------------------------------------------------------------
        |                         |                       |            |
        v                         v                       v            v
+---------------+         +---------------+       +---------------+  +---------------+
|   Client 01   |         |   Client 02   |       |   Client 03   |  |   Client 04   |
| PyTorch+DP    |         | PyTorch+DP    |       | PyTorch+DP    |  | PyTorch+DP    |
| local .pt data|         | local .pt data|       | local .pt data|  | local .pt data|
+-------+-------+         +-------+-------+       +-------+-------+  +-------+-------+
        \_________________________|_______________________|_____________________/
                                  |
                                  v
                       Local private training data
```

---

## 3) Prerequisites

Trước khi chạy dự án, hãy đảm bảo bạn có:

- 🐳 **Docker Desktop** (bắt buộc cho luồng chạy container)
- 🌿 **Git** (clone/pull source code)
- 🐍 **Python 3.10+** (tùy chọn, chỉ cần nếu muốn chạy ngoài Docker)

---

## 4) Quick Start (Golden Steps)

### A. Start FL Server

Tại thư mục gốc dự án, chạy:

**`docker-compose up fl_server`**

---

### B. Build Client Image

Build image một lần trước khi chạy clients:

**`docker build -t fl-client .`**

---

### C. Run 4 Clients

Chạy từng client (mở 4 terminal hoặc 4 máy khác nhau):

```bash
docker run --rm -it ^
  -e CLIENT_ID=[ID] ^
  -e SERVER_ADDRESS=[SERVER_IP]:8080 ^
  -v ${PWD}/client_data:/app/client_data ^
  fl-client
```

> Thay:
> - `[ID]` bằng `1`, `2`, `3`, hoặc `4`
> - `[SERVER_IP]` bằng IP máy đang chạy server (ví dụ `192.168.1.10`)

---

## 5) Project Structure

```text
FL-DP-PROJECT/
|-- src/
|   |-- server.py
|   |-- client.py
|   |-- model.py
|   |-- utils.py
|   |-- dashboard.py
|   |-- simulate.py
|   |-- plot_results.py
|   |-- plot_comparison.py
|   `-- plot_meeting_results.py
|-- client_data/              # Place per-client .pt files here
|-- results/                  # Training logs/metrics/plots output
|-- notebooks/
|   `-- Client.ipynb
|-- docker-compose.yml
|-- Dockerfile
|-- requirements.txt
|-- environment.yaml
`-- README.md
```

---

## 6) Client ID Assignment

| Client ID | Vai trò | Dữ liệu cục bộ |
|---|---|---|
| 1 | Client node 01 | `client_data/client_1.pt` |
| 2 | Client node 02 | `client_data/client_2.pt` |
| 3 | Client node 03 | `client_data/client_3.pt` |
| 4 | Client node 04 | `client_data/client_4.pt` |

---

## 7) Important Notes

- 📦 **Bắt buộc** đặt các file dữ liệu `.pt` vào thư mục **`/client_data/`** trước khi chạy clients.
- 🔥 Nếu chạy qua mạng Wi-Fi nội bộ giữa nhiều máy, có thể cần **tắt Firewall** (hoặc mở đúng port) để clients kết nối server ổn định.
- 📊 Streamlit Dashboard chạy tại cổng **`8501`** (ví dụ: [http://localhost:8501](http://localhost:8501)).
- 🌐 Khi chạy đa máy, luôn đảm bảo biến **`SERVER_ADDRESS=[SERVER_IP]:8080`** trỏ đúng IP của máy server.

---

## 8) Useful Commands

- Start dashboard:
  - **`streamlit run src/dashboard.py --server.port 8501`**
- Stop all running containers:
  - **`docker ps -q | xargs docker stop`**

Chúc bạn triển khai FL + DP thuận lợi! 🚀