import flwr as fl
import torch
import sys
from model import Net
from utils import get_mnist_data, partition_data_non_iid, get_dataloader
from opacus import PrivacyEngine
import warnings

# Tắt các cảnh báo không cần thiết từ Opacus
warnings.filterwarnings("ignore", category=UserWarning)

# torch.backends.cudnn.enabled = False 
# torch.backends.cudnn.benchmark = False

# 1. Cấu hình thiết bị (GPU/CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 2. Tải model và dữ liệu
model = Net().to(device)
train_data, _ = get_mnist_data()

# Nhận ID của client từ dòng lệnh (0 hoặc 1)
client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
all_partitions = partition_data_non_iid(train_data, num_clients=2)
client_train_data = all_partitions[client_id]
train_loader = get_dataloader(client_train_data, batch_size=32)

# 3. Cấu hình Differential Privacy (DP)
# Epsilon là ngân sách bảo mật. Càng thấp = Càng bảo mật nhưng độ chính xác giảm.
TARGET_EPSILON = 10.0 
TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 1.0

class MNISTClient(fl.client.NumPyClient):
    def __init__(self):
        super().__init__()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Khởi tạo Privacy Engine
        self.privacy_engine = PrivacyEngine()
        
        # Biến model và optimizer thành "bản bảo mật"
        # Lưu ý: Opacus sẽ tự động điều chỉnh noise_multiplier để đạt được TARGET_EPSILON
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            target_epsilon=TARGET_EPSILON,
            target_delta=TARGET_DELTA,
            epochs=3, # Số vòng huấn luyện dự kiến
            max_grad_norm=MAX_GRAD_NORM,
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Cập nhật tham số từ Server
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Huấn luyện cục bộ với DP
        self.model.train()
        for images, labels in self.train_loader:
            images, labels = images.to(device), labels.to(device)
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = torch.nn.functional.nll_loss(output, labels)
            loss.backward()
            self.optimizer.step()
        
        # Tính toán mức độ bảo mật đã tiêu thụ (Epsilon thực tế)
        current_epsilon = self.privacy_engine.get_epsilon(delta=TARGET_DELTA)
        print(f"--- Client {client_id}: Training hoàn tất. Epsilon tiêu thụ: {current_epsilon:.2f} ---")
        
        return self.get_parameters(config), len(self.train_loader.dataset), {"epsilon": current_epsilon}

    def evaluate(self, parameters, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        # 2. Đánh giá (Test)
        model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in train_loader: # Hoặc dùng test_loader nếu bạn đã định nghĩa
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss += torch.nn.functional.nll_loss(outputs, labels).item()
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"--- Client {client_id} đánh giá: Accuracy = {accuracy:.4f} ---")
        
        # PHẢI TRẢ VỀ ĐÚNG KHÓA "accuracy" Ở ĐÂY
        return float(loss) / len(train_loader), total, {"accuracy": float(accuracy)}

if __name__ == "__main__":
    print(f"--- CLIENT {client_id} (BẢO MẬT DP) ĐANG KẾT NỐI ---")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MNISTClient())