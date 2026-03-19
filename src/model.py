import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Lớp tích chập 1: 1 kênh vào (ảnh xám), 32 bộ lọc, kích thước 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        # Lớp tích chập 2: 32 đầu vào, 64 bộ lọc, kích thước 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        
        # Lớp Dropout để tránh quá khớp (Overfitting)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Lớp kết nối đầy đủ (Fully Connected)
        # 9216 là số đặc trưng sau khi qua các lớp Conv và Pooling
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 đầu ra tương ứng 10 chữ số (0-9)

    def forward(self, x):
        # Luồng dữ liệu qua các lớp
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Trả về xác suất của từng lớp bằng Log Softmax
        output = F.log_softmax(x, dim=1)
        return output

# Đoạn code kiểm tra nhanh mô hình
if __name__ == "__main__":
    model = Net()
    print("Cấu trúc mô hình CNN:")
    print(model)
    
    # Thử nghiệm với một tensor giả lập (1 ảnh MNIST 28x28)
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"\nKích thước đầu ra: {output.shape} (Phải là [1, 10])")