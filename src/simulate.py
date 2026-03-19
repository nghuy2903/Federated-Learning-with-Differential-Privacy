import os
import sys
import flwr as fl
import torch
from torch.utils.data import Subset

# --- BƯỚC QUAN TRỌNG: FIX LỖI MODULENOTFOUNDERROR ---
# Lấy đường dẫn tuyệt đối của thư mục chứa file simulate.py (chính là thư mục src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Thêm thư mục src vào đầu danh sách tìm kiếm của Python
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Bây giờ mới import các module local
from model import Net
from utils import get_mnist_data, partition_data_non_iid, get_dataloader
from client import MNISTClient

# 1. Hàm tạo Client cho Simulation
def client_fn(cid: str):
    # Khởi tạo model và dữ liệu bên trong hàm này
    # Sử dụng CPU để debug cho ổn định
    device = torch.device("cpu")
    
    # Tải dữ liệu và cắt nhỏ để test cực nhanh (200 ảnh)
    train_data, _ = get_mnist_data()
    small_data = Subset(train_data, range(200)) 
    
    # Chia dữ liệu Non-IID
    all_partitions = partition_data_non_iid(small_data, num_clients=2)
    client_train_data = all_partitions[int(cid)]
    train_loader = get_dataloader(client_train_data, batch_size=32)
    
    my_numpy_client = MNISTClient(client_id=int(cid), train_loader=train_loader, device=device)
    
    # Dùng hàm này, nó sẽ tự động nhận diện và chuyển đổi
    return fl.client.to_client(my_numpy_client)

# 2. Hàm gom chỉ số (Aggregation)
def weighted_average(metrics):
    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if sum(examples) == 0: return {"accuracy": 0.0}
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    # 3. Cấu hình Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # 4. Chạy mô phỏng
    print("\n" + "="*50)
    print("--- ĐANG CHẠY MÔ PHỎNG (SIMULATION MODE) ---")
    print("="*50 + "\n")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=1), # Chạy 1 vòng để check lỗi cho nhanh
        strategy=strategy,
        # Giới hạn tài nguyên để tránh lỗi Ray trên Windows
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )