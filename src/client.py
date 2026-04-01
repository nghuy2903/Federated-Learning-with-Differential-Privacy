import flwr as fl
import json
import os
import torch
import sys
from datetime import datetime
from model import Net
from utils import get_mnist_data, partition_data_non_iid, get_dataloader
from opacus import PrivacyEngine

class MNISTClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, device="cpu"):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.model = Net().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        # Thiết lập Privacy Engine
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            target_epsilon=5.0,
            target_delta=1e-5,
            epochs=3,
            max_grad_norm=1.0,
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def _save_parameter_inspector_sample(self, weights_clean, weights_noisy):
        if not os.path.exists("results"):
            os.makedirs("results")

        sample_payload = {
            "client_id": int(self.client_id),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "layer_name": "fc2.weight",
            "weights_clean": [float(v) for v in weights_clean],
            "weights_noisy": [float(v) for v in weights_noisy],
        }

        output_path = f"results/parameter_inspector_client_{self.client_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample_payload, f, indent=4)

    def fit(self, parameters, config):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        self.model.train()
        inspector_clean_sample = None
        inspector_noisy_sample = None

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = torch.nn.functional.nll_loss(self.model(images), labels)
            loss.backward()

            # Lấy bản sao "clean/reference" trước bước cập nhật có DP noise
            fc2_weight_before = self.model.state_dict()["fc2.weight"].detach().cpu().flatten()
            self.optimizer.step()

            # Trọng số thực tế sau cập nhật (đã chịu tác động của DP)
            fc2_weight_after = self.model.state_dict()["fc2.weight"].detach().cpu().flatten()

            inspector_clean_sample = fc2_weight_before[:100].tolist()
            inspector_noisy_sample = fc2_weight_after[:100].tolist()

        if inspector_clean_sample is not None and inspector_noisy_sample is not None:
            self._save_parameter_inspector_sample(inspector_clean_sample, inspector_noisy_sample)
        
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        return self.get_parameters(config), len(self.train_loader.dataset), {"epsilon": float(epsilon)}

    def evaluate(self, parameters, config):
        # Tương tự như fit, cập nhật lại tham số trước khi eval
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        self.model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += torch.nn.functional.nll_loss(outputs, labels).item()
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        return float(loss) / len(self.train_loader), total, {"accuracy": float(accuracy), "epsilon": float(epsilon)}

if __name__ == "__main__":
    # 1. Lấy ID từ dòng lệnh (mặc định là 0 nếu không nhập)
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # 1b. Lấy IP server từ dòng lệnh (mặc định localhost nếu không nhập)
    server_ip = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    server_address = f"{server_ip}:8080"
    
    # 2. Cấu hình thiết bị (CPU/GPU) - Tôi để CPU cho ổn định như bạn đã test
    device = torch.device("cpu")
    
    # 3. Chuẩn bị dữ liệu cho Client này
    train_data, _ = get_mnist_data()
    # Chia Non-IID thành 2 phần (cho client 0 và client 1)
    all_partitions = partition_data_non_iid(train_data, num_clients=2)
    client_train_data = all_partitions[cid]
    train_loader = get_dataloader(client_train_data, batch_size=32)
    
    # 4. Khởi tạo và chạy Client
    client = MNISTClient(client_id=cid, train_loader=train_loader, device=device)
    
    print(f"--- ĐANG KHỞI CHẠY CLIENT {cid} ---")
    print(f"[*] Đang kết nối tới Server: {server_address}")
    try:
        fl.client.start_numpy_client(server_address=server_address, client=client)
    except Exception as e:
        print("[!] Không thể kết nối tới Server Flower.")
        print(f"[!] Địa chỉ đã thử: {server_address}")
        print("[!] Hãy kiểm tra lại IP Server (LAN IPv4) hoặc Firewall trên máy Server (cổng 8080).")
        print(f"[!] Chi tiết lỗi: {e}")