import flwr as fl
import torch
import sys
from model import Net
from utils import get_mnist_data, partition_data_non_iid, get_dataloader
from opacus import PrivacyEngine

class MNISTClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, device="cpu"):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.model = Net().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        # Thiết lập Privacy Engine
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            target_epsilon=5.0,
            target_delta=1e-5,
            epochs=3,
            max_grad_norm=1.0,
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        self.model.train()
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = torch.nn.functional.nll_loss(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
        
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        return self.get_parameters(config), len(self.train_loader.dataset), {"epsilon": float(epsilon)}

    def evaluate(self, parameters, config):
        # Tương tự như fit, cập nhật lại tham số trước khi eval
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        self.model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += torch.nn.functional.nll_loss(outputs, labels).item()
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        return float(loss) / len(self.train_loader), total, {"accuracy": float(accuracy), "epsilon": float(epsilon)}

if __name__ == "__main__":
    # 1. Lấy ID từ dòng lệnh (mặc định là 0 nếu không nhập)
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    # 2. Cấu hình thiết bị (CPU/GPU) - Tôi để CPU cho ổn định như bạn đã test
    device = torch.device("cpu")
    
    # 3. Chuẩn bị dữ liệu cho Client này
    train_data, _ = get_mnist_data()
    # Chia Non-IID thành 2 phần (cho client 0 và client 1)
    all_partitions = partition_data_non_iid(train_data, num_clients=2)
    client_train_data = all_partitions[cid]
    train_loader = get_dataloader(client_train_data, batch_size=32)
    
    # 4. Khởi tạo và chạy Client
    client = MNISTClient(client_id=cid, train_loader=train_loader, device=device)
    
    print(f"--- ĐANG KHỞI CHẠY CLIENT {cid} ---")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)