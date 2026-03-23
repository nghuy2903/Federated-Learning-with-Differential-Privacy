import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from collections import Counter

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
    
    # Chia thử cho 2 client giống như bạn đang cấu hình trong mô phỏng
    num_clients_test = 2
    clients_data = partition_data_non_iid(train_data, num_clients=num_clients_test)
    
    print("\n" + "="*50)
    print("KIỂM TRA PHÂN BỐ DỮ LIỆU NON-IID")
    print("="*50)
    
    for i, data in enumerate(clients_data):
        print(f"\n--- Client {i} có tổng cộng {len(data)} mẫu dữ liệu ---")
        
        # Duyệt qua toàn bộ tập dữ liệu của Client này để lấy nhãn (label)
        # Quá trình này có thể mất 1-2 giây
        all_labels = [label for _, label in data]
        
        # Đếm số lượng của từng nhãn
        label_counts = Counter(all_labels)
        
        # Sắp xếp lại dictionary theo thứ tự nhãn (0, 1, 2... 9) cho dễ nhìn
        sorted_counts = dict(sorted(label_counts.items()))
        
        # In ra kết quả
        print(f"-> Số lượng nhãn khác nhau: {len(sorted_counts)} nhãn")
        print(f"-> Phân bố chi tiết (Nhãn: Số lượng):")
        
        # In đẹp từng dòng
        for label, count in sorted_counts.items():
            print(f"   + Nhãn {label}: {count} ảnh")
            
    print("\n" + "="*50)