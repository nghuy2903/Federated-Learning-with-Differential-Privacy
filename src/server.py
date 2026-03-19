import flwr as fl
import json
import os
from datetime import datetime

# (Giữ nguyên hàm weighted_average của lượt trước ở đây)
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    epsilons = [num_examples * m.get("epsilon", 0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "avg_epsilon": sum(epsilons) / sum(examples)
    }

def main():
    # Định nghĩa chiến lược hợp nhất FedAvg
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,      # Sử dụng 100% client đang kết nối để huấn luyện
        min_fit_clients=2,     # Đợi ít nhất 2 client mới bắt đầu huấn luyện
        min_available_clients=2, # Đợi ít nhất 2 client có mặt trên mạng
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Khởi chạy Flower Server   
    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # 3. LƯU KẾT QUẢ VÀO FILE
    print("--- ĐANG LƯU KẾT QUẢ HUẤN LUYỆN ---")
    
    # Tạo thư mục results nếu chưa có
    if not os.path.exists('results'):
        os.makedirs('results')

    results_data = {
        "history_accuracy": [acc for _, acc in history.metrics_distributed["accuracy"]],
        "history_epsilon": [eps for _, eps in history.metrics_distributed["avg_epsilon"]],
        "history_loss": [loss for _, loss in history.losses_distributed]
    }

    # Lưu thành file JSON có đính kèm thời gian
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/experiment_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results_data, f, indent=4)

    print(f"--- Đã lưu kết quả tại: {filename} ---")

if __name__ == "__main__":
    main()