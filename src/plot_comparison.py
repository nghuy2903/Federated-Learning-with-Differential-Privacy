import json
import matplotlib.pyplot as plt
import os

def plot_tradeoff():
    # 1. CẤU HÌNH ĐƯỜNG DẪN FILE 
    # MẸO: Hãy thay tên file 20 vòng bạn vừa chạy vào một trong các dòng dưới đây
    experiments = {
        "DP (Kết quả 20 vòng)": "results/experiment_20rounds_20260323_184229.json", # <-- ĐIỀN TÊN FILE CỦA BẠN VÀO ĐÂY
        "DP (Epsilon = 10)": "results/experiment_20260319_173042.json",
        "DP (Epsilon = 2)":  "results/experiment_20260319_174923.json"
    }

    # 2. KHỞI TẠO KHUNG HÌNH (Mở rộng ra để chứa 3 biểu đồ)
    plt.figure(figsize=(18, 5))
    styles = ['b-o', 'orange', 'r--^'] # Xanh dương, Cam, Đỏ nét đứt

    # Đọc trước dữ liệu để tránh mở file nhiều lần
    loaded_data = {}
    for (label, filepath), style in zip(experiments.items(), styles):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                loaded_data[label] = (json.load(f), style)
        else:
            print(f"[Cảnh báo] Không tìm thấy file: {filepath} -> Sẽ bỏ qua và vẽ các file còn lại.")

    # Nếu không có file nào được tải, dừng chương trình
    if not loaded_data:
        print("Không có dữ liệu để vẽ. Vui lòng kiểm tra lại đường dẫn file.")
        return

    # --- BIỂU ĐỒ 1: ACCURACY ---
    plt.subplot(1, 3, 1)
    for label, (data, style) in loaded_data.items():
        if 'history_accuracy' in data:
            rounds = range(1, len(data['history_accuracy']) + 1)
            plt.plot(rounds, data['history_accuracy'], style, label=label, linewidth=2, markersize=6)
    
    plt.title('Độ chính xác (Accuracy)', fontsize=14, pad=10)
    plt.xlabel('Vòng huấn luyện (Federated Rounds)', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # --- BIỂU ĐỒ 2: LOSS ---
    plt.subplot(1, 3, 2)
    for label, (data, style) in loaded_data.items():
        if 'history_loss' in data:
            rounds = range(1, len(data['history_loss']) + 1)
            # Nếu là màu cam thì dùng nét đứt vuông cho dễ phân biệt
            if style == 'orange':
                plt.plot(rounds, data['history_loss'], color='orange', linestyle='-.', marker='s', label=label, linewidth=2, markersize=6)
            else:
                plt.plot(rounds, data['history_loss'], style, label=label, linewidth=2, markersize=6)
                
    plt.title('Hàm mất mát (Loss)', fontsize=14, pad=10)
    plt.xlabel('Vòng huấn luyện (Federated Rounds)', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # --- BIỂU ĐỒ 3: EPSILON (MỚI) ---
    plt.subplot(1, 3, 3)
    for label, (data, style) in loaded_data.items():
        if 'history_epsilon' in data:
            rounds = range(1, len(data['history_epsilon']) + 1)
            plt.plot(rounds, data['history_epsilon'], style, label=label, linewidth=2, markersize=6)
            
    plt.title('Ngân sách bảo mật tiêu thụ (Epsilon)', fontsize=14, pad=10)
    plt.xlabel('Vòng huấn luyện (Federated Rounds)', fontsize=11)
    plt.ylabel('Epsilon', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # 3. LƯU VÀ HIỂN THỊ
    plt.tight_layout()
    save_path = 'results/privacy_utility_tradeoff.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n--- Đã xuất biểu đồ thành công tại: {save_path} ---")
    plt.show()

if __name__ == "__main__":
    plot_tradeoff()