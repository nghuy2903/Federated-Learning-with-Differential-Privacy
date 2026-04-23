import flwr as fl
import json
import os
import socket
import torch
from datetime import datetime
from collections import OrderedDict
from typing import Optional
from model import Net

# In ra IPv4 LAN để client dễ nhập đúng IP server
def get_lan_ipv4() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Không cần kết nối thực sự; chỉ để OS chọn interface LAN phù hợp
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()

# (Giữ nguyên hàm weighted_average của lượt trước ở đây)
def weighted_average(metrics):
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples) if sum(examples) > 0 else 1

    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    epsilons = [num_examples * m.get("epsilon", 0.0) for num_examples, m in metrics]
    
    return {
        "accuracy": sum(accuracies) / total_examples,
        "avg_epsilon": sum(epsilons) / total_examples
    }
class EarlyStoppingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, patience=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience  # Số vòng "chịu đựng" tối đa nếu không cải thiện
        self.best_acc = 0.0       # Lưu độ chính xác cao nhất
        self.strikes = 0          # Đếm số vòng dậm chân tại chỗ
        self.stop_training = False # Cờ hiệu dừng hệ thống

    @staticmethod
    def _ensure_results_dir() -> None:
        if not os.path.exists("results"):
            os.makedirs("results")

    @staticmethod
    def _get_inspector_path(server_round: int) -> str:
        return f"results/parameter_inspector_server_round_{server_round}.json"

    @staticmethod
    def _update_round_history_fields(
        server_round: int,
        accuracy: float,
        loss: float,
        epsilon: Optional[float],
    ) -> None:
        EarlyStoppingFedAvg._ensure_results_dir()

        inspector_path = EarlyStoppingFedAvg._get_inspector_path(server_round)
        payload = {}
        if os.path.exists(inspector_path):
            try:
                with open(inspector_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                payload = {}

        payload["server_round"] = int(server_round)
        payload["history_accuracy"] = [float(accuracy)]
        payload["history_loss"] = [float(loss)]
        payload["history_epsilon"] = [] if epsilon is None else [float(epsilon)]

        with open(inspector_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

    @staticmethod
    def _save_client_metric_payloads(server_round: int, results) -> None:
        EarlyStoppingFedAvg._ensure_results_dir()

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics if fit_res.metrics is not None else {}
            raw_payload = metrics.get("parameter_inspector_payload")
            if raw_payload is None:
                continue

            try:
                payload = json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
                if not isinstance(payload, dict):
                    raise TypeError("Inspector payload khong phai JSON object hop le")
                client_id = str(payload.get("client_id", client_proxy.cid))
                output_path = f"results/client_{client_id}_round_{server_round}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=4)
            except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                print(
                    f"[!] Khong the luu payload inspector tu client {client_proxy.cid} "
                    f"tai round {server_round}: {exc}"
                )

    def aggregate_fit(self, server_round, results, failures):
        # 1. Gọi hàm gốc để lấy bộ trọng số đã được trung bình cộng từ các Client
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if failures:
            print(f"[!] Round {server_round}: co {len(failures)} client fit bi loi, tiep tuc aggregate.")

        # Đồng bộ JSON do client gửi qua metrics về thư mục results của server.
        if results:
            self._save_client_metric_payloads(server_round=server_round, results=results)
        
        # 2. Tiến hành lưu file nếu quá trình gom thành công
        if aggregated_parameters is not None:
            print(f"[*] Đang trích xuất và lưu Global Model vòng {server_round}...")
            
            # Chuyển đổi định dạng byte của Flower sang mảng NumPy
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Lưu snapshot tham số để minh họa cơ chế giảm nhiễu sau aggregate
            model_keys = list(Net().state_dict().keys())
            if "fc2.weight" in model_keys:
                fc2_idx = model_keys.index("fc2.weight")
                aggregated_fc2_sample = ndarrays[fc2_idx].reshape(-1)[:100].tolist()
                client_fc2_samples = []

                for client_proxy, fit_res in results:
                    client_ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    client_fc2_sample = client_ndarrays[fc2_idx].reshape(-1)[:100].tolist()
                    client_fc2_samples.append(
                        {
                            "client_id": str(client_proxy.cid),
                            "weights_noisy": [float(v) for v in client_fc2_sample],
                        }
                    )

                self._ensure_results_dir()

                inspector_payload = {
                    "server_round": int(server_round),
                    "layer_name": "fc2.weight",
                    "client_samples_noisy": client_fc2_samples,
                    "aggregated_sample": [float(v) for v in aggregated_fc2_sample],
                    # Các trường history sẽ được cập nhật sau evaluate của chính vòng này.
                    "history_accuracy": [],
                    "history_loss": [],
                    "history_epsilon": [],
                }

                inspector_path = self._get_inspector_path(server_round)
                with open(inspector_path, "w", encoding="utf-8") as f:
                    json.dump(inspector_payload, f, indent=4)
            
            # Khởi tạo một vỏ mô hình rỗng
            model = Net()
            
            # Ghép trọng số NumPy vào cấu trúc của PyTorch
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            
            # Đảm bảo thư mục results tồn tại và lưu mô hình
            self._ensure_results_dir()
            save_path = "results/global_model_latest.pth"
            torch.save(model.state_dict(), save_path)
            
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        # Tính toán kết quả vòng hiện tại
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if failures:
            print(f"[!] Round {server_round}: co {len(failures)} client evaluate bi loi, tiep tuc tong hop.")

        if metrics and "accuracy" in metrics and loss is not None:
            acc = float(metrics["accuracy"])
            eps_raw = metrics.get("avg_epsilon")
            eps = None if eps_raw is None else float(eps_raw)
            eps_text = f"{eps:.4f}" if eps is not None else "N/A"
            print(f"\n---> [Vòng {server_round}] Độ chính xác: {acc:.4f} | Tiêu thụ Epsilon: {eps_text} <---")

            # Ghi bổ sung lịch sử của vòng vào file JSON round tương ứng.
            self._update_round_history_fields(
                server_round=server_round,
                accuracy=acc,
                loss=float(loss),
                epsilon=eps,
            )
            
            # Kiểm tra xem mô hình có cải thiện ít nhất 0.1% (0.001) hay không
            if acc > self.best_acc + 0.001:
                self.best_acc = acc
                self.strikes = 0 # Reset lại bộ đếm
            else:
                self.strikes += 1
                print(f"[!] Cảnh báo: Độ chính xác không tăng ({self.strikes}/{self.patience})")
                
            # Nếu chạm giới hạn chịu đựng -> Kích hoạt dừng sớm
            if self.strikes >= self.patience:
                print(f"\n[!!!] KÍCH HOẠT EARLY STOPPING TẠI VÒNG {server_round} [!!!]")
                print(f"Lý do: Bảo toàn ngân sách Epsilon vì mô hình đã đạt đỉnh ({self.best_acc:.4f}).")
                self.stop_training = True
                
        return loss, metrics

    # Can thiệp vào quá trình chọn Client để ép Server dừng lại
    def configure_fit(self, server_round, parameters, client_manager):
        if self.stop_training:
            return [] # Trả về mảng rỗng -> Không chọn Client nào -> Server ngắt vòng Fit
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.stop_training:
            return [] # Không chọn Client nào -> Server ngắt vòng Evaluate
        return super().configure_evaluate(server_round, parameters, client_manager)

def main():
    # Định nghĩa chiến lược hợp nhất FedAvg
    strategy = EarlyStoppingFedAvg(
        patience=3, # Nếu 3 vòng liên tiếp accuracy không tăng -> Dừng
        fraction_fit=1.0,
        min_fit_clients = 3,
        min_available_clients = 3,
        min_evaluate_clients = 3,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # 4. CHẠY SERVER VỚI TỐI ĐA 20 VÒNG
    print("--- SERVER KHỞI ĐỘNG: TỐI ĐA 20 VÒNG + EARLY STOPPING ---")
    print(f"[*] LAN IPv4 của Server: {get_lan_ipv4()}")
    print("[*] Server đang lắng nghe trên 0.0.0.0:8080 (LAN/Wi-Fi).")
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3), #Thay số vòng bằng 3 để lưu model
        strategy=strategy,
    )

    # 5. Lưu kết quả
    print("\n--- ĐANG LƯU KẾT QUẢ HUẤN LUYỆN ---")
    if not os.path.exists('results'):
        os.makedirs('results')

    acc_history = history.metrics_distributed.get("accuracy", [])
    eps_history = history.metrics_distributed_fit.get("avg_epsilon", [])
    results_data = {
        "history_accuracy": [acc for _, acc in acc_history],
        "history_epsilon": [eps for _, eps in eps_history],
        "history_loss": [loss for _, loss in history.losses_distributed]
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/experiment_20rounds_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results_data, f, indent=4)

    print(f"--- Đã lưu kết quả tại: {filename} ---")

if __name__ == "__main__":
    main()