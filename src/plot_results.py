import json
import matplotlib.pyplot as plt
import glob
import os

def plot_latest_experiment():
    # Tìm file json mới nhất trong thư mục results
    list_of_files = glob.glob('results/*.json')
    if not list_of_files:
        print("Không tìm thấy file kết quả nào!")
        return
    latest_file = max(list_of_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        data = json.load(f)

    rounds = range(1, len(data['history_accuracy']) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Vẽ đường Accuracy
    color = 'tab:red'
    ax1.set_xlabel('Federated Rounds')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(rounds, data['history_accuracy'], color=color, marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    # Vẽ đường Epsilon trên cùng một biểu đồ (trục y thứ 2)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Epsilon (Privacy Leakage)', color=color)
    ax2.plot(rounds, data['history_epsilon'], color=color, marker='s', linestyle='--', label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Trade-off between Accuracy and Privacy (DP-FedAvg)\nSource: {os.path.basename(latest_file)}')
    fig.tight_layout()
    plt.savefig('results/latest_plot.png')
    plt.show()

if __name__ == "__main__":
    plot_latest_experiment()