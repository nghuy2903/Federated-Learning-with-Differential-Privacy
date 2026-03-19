import flwr as fl
from client import MNISTClient # Import class từ file bạn đã viết

def client_fn(cid: str):
    return MNISTClient().to_client() # cid là ID của client

if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=1), # Chỉ chạy 1 vòng để test lỗi
        strategy=fl.server.strategy.FedAvg(),
    )