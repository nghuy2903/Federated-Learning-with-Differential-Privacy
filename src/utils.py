import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

def get_mnist_data():
    """Tải bộ dữ liệu MNIST và chuẩn hóa."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    return train_dataset, test_dataset

def partition_data_non_iid(dataset, num_clients=3):
    """
    Chia dữ liệu theo dạng Non-IID: Mỗi Client sẽ nhận được các nhóm nhãn khác nhau.
    Ví dụ: Client 0 nhận nhãn 0,1; Client 1 nhận nhãn 2,3...
    """
    # Lấy tất cả các nhãn trong tập dữ liệu
    if isinstance(dataset, Subset):
        # Lấy toàn bộ nhãn từ dataset gốc
        full_dataset_targets = np.array(dataset.dataset.targets)
        # Chỉ lấy nhãn của những vị trí nằm trong Subset
        labels = full_dataset_targets[dataset.indices]
        indices = np.array(dataset.indices)
    else:
        # Nếu là Dataset gốc (MNIST)
        labels = np.array(dataset.targets)
        indices = np.arange(len(labels))
    
    # Sắp xếp chỉ số dựa trên nhãn
    # Lưu ý: Sắp xếp các chỉ số THỰC TẾ của dataset
    sorted_indices = indices[np.argsort(labels)]
    
    partition_size = len(sorted_indices) // num_clients
    partitions = []
    
    for i in range(num_clients):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size if i != num_clients - 1 else len(sorted_indices)
        
        # Lấy các chỉ số con đã sắp xếp
        subset_indices = sorted_indices[start_idx:end_idx]
        
        # Nếu dataset đầu vào đã là Subset, ta phải lấy dataset gốc của nó 
        # để tránh việc lồng Subset vào Subset nhiều lần
        actual_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        partitions.append(Subset(actual_dataset, subset_indices))
        
    return partitions

def get_dataloader(dataset, batch_size=32, shuffle=True):
    """Tạo DataLoader cho một tập dữ liệu con."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Kiểm tra nhanh file utils.py
if __name__ == "__main__":
    train_data, _ = get_mnist_data()
    clients_data = partition_data_non_iid(train_data, num_clients=3)
    
    for i, data in enumerate(clients_data):
        print(f"Client {i} có {len(data)} mẫu dữ liệu.")
        # Kiểm tra thử nhãn của một vài mẫu đầu tiên để xem tính Non-IID
        sample_labels = [data[j][1] for j in range(10)]
        print(f"Nhãn mẫu của Client {i}: {sample_labels}")