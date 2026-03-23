import json
import matplotlib.pyplot as plt
import os

def plot_tradeoff():
    # 1. CẤU HÌNH ĐƯỜNG DẪN FILE 
    # Thay đổi các chuỗi này thành tên file thực tế trong máy của bạn
    experiments = {
        "No DP (Baseline)": "results/experiment_20260319_151946.json",   # File có Accuracy 94%
        "DP (Epsilon = 10)": "results/experiment_20260319_173042.json", # File có Accuracy 73.6%
        "DP (Epsilon = 2)":  "results/experiment_20260319_174923.json"   # File có Accuracy 73.7%
    }

    # Tạo khung hình với 2 biểu đồ nằm ngang nhau
    plt.figure(figsize=(14, 6))

    # Cấu hình màu và kiểu đường (Xanh lá, Cam, Đỏ)
    styles = ['g-o', 'orange', 'r--^']

    # --- BIỂU ĐỒ 1: ACCURACY ---
    plt.subplot(1, 2, 1)
    for (label, filepath), style in zip(experiments.items(), styles):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Trục X là số vòng (Round)
                rounds = range(1, len(data['history_accuracy']) + 1)
                plt.plot(rounds, data['history_accuracy'], style, label=label, linewidth=2, markersize=8)
        else:
            print(f"[Cảnh báo] Không tìm thấy file: {filepath}")
            return

    plt.title('Sự thay đổi của Accuracy qua các vòng huấn luyện', fontsize=14, pad=15)
    plt.xlabel('Vòng huấn luyện (Federated Rounds)', fontsize=12)
    plt.ylabel('Độ chính xác (Accuracy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)

    # --- BIỂU ĐỒ 2: LOSS ---
    plt.subplot(1, 2, 2)
    for (label, filepath), style in zip(experiments.items(), styles):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                rounds = range(1, len(data['history_loss']) + 1)
                if style == 'orange': # Custom chút cho đường nét đứt
                    plt.plot(rounds, data['history_loss'], color='orange', linestyle='-.', marker='s', label=label, linewidth=2, markersize=8)
                else:
                    plt.plot(rounds, data['history_loss'], style, label=label, linewidth=2, markersize=8)

    plt.title('Sự thay đổi của Hàm mất mát (Loss)', fontsize=14, pad=15)
    plt.xlabel('Vòng huấn luyện (Federated Rounds)', fontsize=12)
    plt.ylabel('Hàm mất mát (Loss)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)

    # Lưu và hiển thị
    plt.tight_layout()
    save_path = 'results/privacy_utility_tradeoff.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n--- Đã xuất biểu đồ thành công tại: {save_path} ---")
    plt.show()

if __name__ == "__main__":
    plot_tradeoff()