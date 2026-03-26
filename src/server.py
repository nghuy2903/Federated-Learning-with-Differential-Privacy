import flwr as fl
import json
import os
import torch
from datetime import datetime
from collections import OrderedDict
from model import Net

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

    def aggregate_fit(self, server_round, results, failures):
        # 1. Gọi hàm gốc để lấy bộ trọng số đã được trung bình cộng từ các Client
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # 2. Tiến hành lưu file nếu quá trình gom thành công
        if aggregated_parameters is not None:
            print(f"[*] Đang trích xuất và lưu Global Model vòng {server_round}...")
            
            # Chuyển đổi định dạng byte của Flower sang mảng NumPy
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Khởi tạo một vỏ mô hình rỗng
            model = Net()
            
            # Ghép trọng số NumPy vào cấu trúc của PyTorch
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            
            # Đảm bảo thư mục results tồn tại và lưu mô hình
            if not os.path.exists('results'):
                os.makedirs('results')
            save_path = "results/global_model_latest.pth"
            torch.save(model.state_dict(), save_path)
            
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        # Tính toán kết quả vòng hiện tại
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if metrics and "accuracy" in metrics:
            acc = metrics["accuracy"]
            eps = metrics.get('avg_epsilon', 0)
            print(f"\n---> [Vòng {server_round}] Độ chính xác: {acc:.4f} | Tiêu thụ Epsilon: {eps:.4f} <---")
            
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
        min_fit_clients=2,
        min_available_clients=2,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # 4. CHẠY SERVER VỚI TỐI ĐA 20 VÒNG
    print("--- SERVER KHỞI ĐỘNG: TỐI ĐA 20 VÒNG + EARLY STOPPING ---")
    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
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