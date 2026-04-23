import flwr as fl
import json
import os
import torch
import sys
from datetime import datetime
from model import Net
from utils import get_dataloader
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

    @staticmethod
    def _extract_server_round(config) -> int:
        round_value = config.get("server_round", config.get("round", 0))
        try:
            return int(round_value)
        except (TypeError, ValueError):
            return 0

    def _save_parameter_inspector_sample(
        self,
        server_round,
        weights_clean,
        weights_noisy,
        local_accuracy,
        local_loss,
    ):
        if not os.path.exists("results"):
            os.makedirs("results")

        sample_payload = {
            "client_id": int(self.client_id),
            "server_round": int(server_round),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "layer_name": "fc2.weight",
            "history_accuracy": [float(local_accuracy)],
            "history_loss": [float(local_loss)],
            "weights_clean": [float(v) for v in weights_clean],
            "weights_noisy": [float(v) for v in weights_noisy],
        }

        output_path = f"results/client_{self.client_id}_round_{server_round}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample_payload, f, indent=4)
    
    @staticmethod
    def _build_parameter_inspector_payload(
        client_id,
        server_round,
        weights_clean,
        weights_noisy,
        local_accuracy,
        local_loss,
    ):
        return {
            "client_id": int(client_id),
            "server_round": int(server_round),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "layer_name": "fc2.weight",
            "history_accuracy": [float(local_accuracy)],
            "history_loss": [float(local_loss)],
            "weights_clean": [float(v) for v in weights_clean],
            "weights_noisy": [float(v) for v in weights_noisy],
        }

    def fit(self, parameters, config):
        server_round = self._extract_server_round(config)
        params_dict = zip(self.model.parameters(), parameters)
        for p, v in params_dict:
            p.data = torch.from_numpy(v).to(self.device)

        self.model.train()
        inspector_clean_sample = None
        inspector_noisy_sample = None
        correct_predictions = 0
        total_samples = 0
        total_loss = 0.0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = torch.nn.functional.nll_loss(outputs, labels)
            loss.backward()
            total_loss += float(loss.item())
            predicted = torch.max(outputs.data, 1)[1]
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # # Lấy bản sao "clean/reference" trước bước cập nhật có DP noise
            # fc2_weight_before = self.model.state_dict()["fc2.weight"].detach().cpu().flatten()
            # self.optimizer.step()

            # # Trọng số thực tế sau cập nhật (đã chịu tác động của DP)
            # fc2_weight_after = self.model.state_dict()["fc2.weight"].detach().cpu().flatten()
            # Tự động tìm lớp fc2 dù có bị Opacus wrap hay không
            target_layer = self.model._module.fc2 if hasattr(self.model, "_module") else self.model.fc2

            # Lấy trọng số trước khi cập nhật
            fc2_weight_before = target_layer.weight.data.detach().cpu().flatten()

            self.optimizer.step()

            # Lấy trọng số sau khi cập nhật (đã có nhiễu DP)
            fc2_weight_after = target_layer.weight.data.detach().cpu().flatten()

            inspector_clean_sample = fc2_weight_before[:100].tolist()
            inspector_noisy_sample = fc2_weight_after[:100].tolist()

        # if inspector_clean_sample is not None and inspector_noisy_sample is not None:
        #     self._save_parameter_inspector_sample(inspector_clean_sample, inspector_noisy_sample)

        local_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        local_accuracy = (correct_predictions / total_samples) if total_samples > 0 else 0.0

        inspector_payload = None
        if inspector_clean_sample is not None and inspector_noisy_sample is not None:
            inspector_payload = self._build_parameter_inspector_payload(
                client_id=self.client_id,
                server_round=server_round,
                weights_clean=inspector_clean_sample,
                weights_noisy=inspector_noisy_sample,
                local_accuracy=local_accuracy,
                local_loss=local_loss,
            )
            try:
                self._save_parameter_inspector_sample(
                    server_round=server_round,
                    weights_clean=inspector_clean_sample,
                    weights_noisy=inspector_noisy_sample,
                    local_accuracy=local_accuracy,
                    local_loss=local_loss,
                )
            except Exception as e:
                print(f"[!] Cảnh báo Inspector: {e}") # Chỉ hiện cảnh báo, không làm sập Client
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        fit_metrics = {"epsilon": float(epsilon)}
        if inspector_payload is not None:
            fit_metrics["parameter_inspector_payload"] = json.dumps(inspector_payload)
        return self.get_parameters(config), len(self.train_loader.dataset), fit_metrics

    def evaluate(self, parameters, config):
        # Tương tự như fit, cập nhật lại tham số trước khi eval
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Lưu global model mới nhất để Dashboard có thể suy luận ngay trên client.
        if not os.path.exists("results"):
            os.makedirs("results")
        torch.save(self.model.state_dict(), "results/global_model_latest.pth")
        
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

    # 1b. Lấy server host từ dòng lệnh (mặc định fl_server cho Docker Compose)
    server_ip = sys.argv[2] if len(sys.argv) > 2 else "fl_server"
    server_address = f"{server_ip}:8080"
    
    # 2. Cấu hình thiết bị (CPU/GPU)
    device = torch.device("cpu")
    
    # 3. Chuẩn bị dữ liệu cho Client này (ưu tiên mount path trong Docker)
    docker_data_path = f"/app/client_data/client_{cid}_data.pt"
    local_data_path = f"client_data/client_{cid}_data.pt"
    data_path = docker_data_path if os.path.exists(docker_data_path) else local_data_path
    
    if not os.path.exists(data_path):
        print(f"[!] Lỗi: Không tìm thấy file dữ liệu tại {data_path}")
        sys.exit(1)

    try:
        # Nạp đối tượng Subset hoặc Dataset từ file .pt
        client_train_data = torch.load(data_path, weights_only=False)
        
        # Tạo DataLoader từ dữ liệu đã nạp
        # Lưu ý: get_dataloader là hàm bạn đã định nghĩa trong utils.py
        train_loader = get_dataloader(client_train_data, batch_size=32)
        
        print(f"[*] Đã nạp thành công dữ liệu từ: {data_path}")
        print(f"[*] Số lượng mẫu huấn luyện: {len(client_train_data)}")
        
    except Exception as e:
        print(f"[!] Lỗi khi nạp dữ liệu: {e}")
        sys.exit(1)
    
    # 4. Khởi tạo và chạy Client
    client = MNISTClient(client_id=cid, train_loader=train_loader, device=device)
    
    print(f"--- ĐANG KHỞI CHẠY CLIENT {cid} ---")
    print(f"[*] Đang kết nối tới Server: {server_address}")
    try:
        fl.client.start_numpy_client(server_address=server_address, client=client)
    except Exception as e:
        print("[!] Không thể kết nối tới Server Flower.")
        print(f"[!] Địa chỉ đã thử: {server_address}")
        print("[!] Hãy kiểm tra lại server host/IP hoặc Firewall trên máy Server (cổng 8080).")
        print(f"[!] Chi tiết lỗi: {e}")