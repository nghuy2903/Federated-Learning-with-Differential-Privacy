import json
import os
import random
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from torchvision import datasets, transforms

from model import Net


ROUND_FILE_PATTERN = "parameter_inspector_server_round_*.json"
RESULTS_DIR = os.path.join("results")
ROUND_FILE_REGEX = re.compile(r"^parameter_inspector_server_round_(\d+)\.json$")
SERVER_CLIENT_FILE_REGEX = re.compile(r"^client_(.+)_round_(\d+)\.json$")
DP_EXPERIMENT_FILENAME = "experiment_20rounds_20260505_172719.json"
NODP_EXPERIMENT_FILENAME = "experiment_nodp_last20rounds_20260505_184731.json"
COMPARISON_ROUNDS = 20


def ensure_results_dir(results_dir: str) -> None:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


def load_json_file(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_result_files(results_dir: str) -> List[str]:
    if not os.path.exists(results_dir):
        return []
    result_files: List[str] = []
    for filename in os.listdir(results_dir):
        if ROUND_FILE_REGEX.match(filename):
            result_files.append(os.path.join(results_dir, filename))
    return result_files


def extract_round_number(file_path: str) -> Optional[int]:
    matched = ROUND_FILE_REGEX.match(os.path.basename(file_path))
    if matched is None:
        return None
    round_number = int(matched.group(1))
    return round_number if round_number > 0 else None


def load_round_metrics(file_path: str) -> Dict[str, Optional[float]]:
    data = load_json_file(file_path)

    def pick_metric(metric_key: str, round_number: int) -> Optional[float]:
        if metric_key not in data:
            return None

        metric_value = data[metric_key]
        if isinstance(metric_value, list):
            if not metric_value:
                return None
            if len(metric_value) >= round_number:
                return float(metric_value[round_number - 1])
            return float(metric_value[-1])
        return float(metric_value)

    round_number = extract_round_number(file_path)
    if round_number is None:
        raise ValueError("Không trích xuất được số vòng từ file.")

    accuracy = pick_metric("history_accuracy", round_number)
    loss = pick_metric("history_loss", round_number)
    epsilon = pick_metric("history_epsilon", round_number)

    if accuracy is None or loss is None:
        raise KeyError("File JSON thiếu 'history_accuracy' hoặc 'history_loss' hợp lệ.")

    return {
        "accuracy": float(accuracy),
        "loss": float(loss),
        "epsilon": epsilon,
    }

def build_server_history_dataframe(results_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, Optional[float]]] = []
    files = find_result_files(results_dir)

    if not files:
        return pd.DataFrame()

    for file_path in files:
        round_number = extract_round_number(file_path)
        if round_number is None:
            continue

        try:
            metrics = load_round_metrics(file_path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            st.warning(f"Bỏ qua lỗi `{os.path.basename(file_path)}`: {exc}")
            continue

        rows.append(
            {
                "Round": round_number,
                "Accuracy": float(metrics["accuracy"]),
                "Loss": float(metrics["loss"]),
                "Epsilon": metrics["epsilon"],
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(by="Round").reset_index(drop=True)

    # Epsilon tu server/client da la gia tri tich luy den round hien tai,
    # vi vay chi can hien thi truc tiep theo tung round.
    df["Epsilon"] = pd.to_numeric(df["Epsilon"], errors="coerce").fillna(0.0)
    return df


def build_client_history_dataframe(results_dir: str) -> pd.DataFrame:
    history_path = os.path.join(results_dir, "experiment_global_history.json")
    if not os.path.exists(history_path):
        return pd.DataFrame()

    try:
        history_payload = load_json_file(history_path)
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Không thể đọc `experiment_global_history.json`: {exc}")
        return pd.DataFrame()

    accuracy_history = history_payload.get("history_accuracy", [])
    loss_history = history_payload.get("history_loss", [])
    epsilon_history = history_payload.get("history_epsilon", [])
    max_len = max(len(accuracy_history), len(loss_history), len(epsilon_history), 0)
    if max_len == 0:
        return pd.DataFrame()

    rows = []
    for idx in range(max_len):
        rows.append(
            {
                "Round": idx + 1,
                "Accuracy": float(accuracy_history[idx]) if idx < len(accuracy_history) else 0.0,
                "Loss": float(loss_history[idx]) if idx < len(loss_history) else 0.0,
                "Epsilon": float(epsilon_history[idx]) if idx < len(epsilon_history) else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    # epsilon_history la epsilon tich luy theo round -> khong cong don them.
    df["Epsilon"] = pd.to_numeric(df["Epsilon"], errors="coerce").fillna(0.0)
    return df


def _normalize_history_to_20_rounds(values: List[float], rounds: int = COMPARISON_ROUNDS) -> List[float]:
    normalized = pd.to_numeric(pd.Series(values), errors="coerce").tolist()
    normalized = [float(v) if pd.notna(v) else float("nan") for v in normalized][-rounds:]
    if len(normalized) < rounds:
        normalized = [float("nan")] * (rounds - len(normalized)) + normalized
    return normalized


def _load_experiment_metrics(file_path: str) -> Tuple[List[float], List[float], List[float]]:
    payload = load_json_file(file_path)
    accuracy = payload.get("history_accuracy", [])
    loss = payload.get("history_loss", [])
    epsilon = payload.get("history_epsilon", [])
    return (
        _normalize_history_to_20_rounds(accuracy),
        _normalize_history_to_20_rounds(loss),
        _normalize_history_to_20_rounds(epsilon),
    )



import os
import json
from typing import List
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def render_dp_nodp_comparison_charts(results_dir: str) -> None:
    dp_path = os.path.join(results_dir, DP_EXPERIMENT_FILENAME)
    nodp_path = os.path.join(results_dir, NODP_EXPERIMENT_FILENAME)

    missing_files: List[str] = []
    if not os.path.exists(dp_path):
        missing_files.append(DP_EXPERIMENT_FILENAME)
    if not os.path.exists(nodp_path):
        missing_files.append(NODP_EXPERIMENT_FILENAME)

    if missing_files:
        st.error(
            "Không tìm thấy file kết quả cần so sánh trong `results/`: "
            + ", ".join(f"`{name}`" for name in missing_files)
        )
        return

    try:
        dp_acc, dp_loss, dp_eps = _load_experiment_metrics(dp_path)
        nodp_acc, nodp_loss, nodp_eps = _load_experiment_metrics(nodp_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        st.error(f"Không thể đọc file kết quả thực nghiệm: {exc}")
        return

    rounds = list(range(1, COMPARISON_ROUNDS + 1))

    # --- 1. THAY THẾ THÀNH Ô CHỌN DUY NHẤT (SELECTBOX) ---
    selected_metric = st.selectbox(
        label="Chọn chỉ số muốn quan sát:",
        options=["Accuracy", "Loss", "Privacy Budget"]
    )

    # --- 2. TỰ ĐỘNG GÁN DỮ LIỆU VÀ TIÊU ĐỀ THEO LỰA CHỌN ---
    if selected_metric == "Accuracy":
        y_dp = dp_acc
        y_nodp = nodp_acc
        title = "So sánh Accuracy"
        ylabel = "Accuracy"
    elif selected_metric == "Loss":
        y_dp = dp_loss
        y_nodp = nodp_loss
        title = "So sánh Loss"
        ylabel = "NLL Loss"
    else:  # Privacy Budget
        y_dp = dp_eps
        y_nodp = nodp_eps
        title = "So sánh Privacy Budget (Epsilon)"
        ylabel = "Epsilon"

    # Định dạng hover: hiển thị giá trị làm tròn 2 số thập phân (:.2f)
    hover_template = "<b>Round:</b> %{x}<br><b>Giá trị:</b> %{y:.2f}<extra></extra>"

    # --- 3. LOGIC VẼ MỘT BIỂU ĐỒ DUY NHẤT ---
    fig = go.Figure()

    # Đường biểu diễn dữ liệu có DP
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=y_dp,
            name="DP",
            line=dict(color="#1f77b4", width=2.5),
            mode="lines+markers",
            hovertemplate=hover_template,
        )
    )

    # Đường biểu diễn dữ liệu không có DP (No-DP)
    fig.add_trace(
        go.Scatter(
            x=rounds,
            y=y_nodp,
            name="No-DP",
            line=dict(color="#d62728", width=2.5, dash="dash"),
            mode="lines+markers",
            hovertemplate=hover_template,
        )
    )

    # Cấu hình giao diện biểu đồ
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Round",
        yaxis_title=ylabel,
        hovermode="x unified",  # Tạo đường kẻ dọc đồng bộ khi di chuột
        margin=dict(l=40, r=40, t=50, b=40),
        height=450,  # Tăng nhẹ chiều cao để biểu đồ đơn nhìn rõ ràng, đẹp mắt hơn
    )

    # Hiển thị biểu đồ lên Streamlit
    st.plotly_chart(fig, use_container_width=True)


@st.cache_resource(show_spinner=False)
def load_global_model(model_path: str) -> Net:
    model = Net()
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    if any(key.startswith("_module.") for key in checkpoint.keys()):
        cleaned_state_dict = OrderedDict(
            (key.replace("_module.", "", 1), value) for key, value in checkpoint.items()
        )
    else:
        cleaned_state_dict = checkpoint

    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_mnist_test_dataset() -> datasets.MNIST:
    return datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )


def render_live_inference_section() -> None:

    if "selected_inference_digit" not in st.session_state:
        st.session_state["selected_inference_digit"] = None
    st.markdown("**Chọn nhãn muốn kiểm tra**")
    for row_start in (0, 5):
        row_cols = st.columns(5)
        for col_idx, digit in enumerate(range(row_start, row_start + 5)):
            with row_cols[col_idx]:
                if st.button(f"{digit}", key=f"infer_digit_{digit}", use_container_width=True):
                    st.session_state["selected_inference_digit"] = digit
    selected_digit = st.session_state["selected_inference_digit"]
    if selected_digit is None:
        st.info("Vui lòng chọn một chữ số để bắt đầu suy luận.")
        return

    model_path = os.path.join(RESULTS_DIR, "global_model_latest.pth")
    if not os.path.exists(model_path):
        st.error("Khong tim thay model global moi nhat tai `results/global_model_latest.pth`.")
        return

    st.info("Đang sử dụng mô hình Global mới nhất để dự đoán.")

    try:
        model = load_global_model(model_path)
    except Exception as exc:
        st.error(f"Khong the load model tu `results/global_model_latest.pth`: {exc}")
        return

    try:
        mnist_test = load_mnist_test_dataset()
    except Exception as exc:
        st.error(f"Khong the tai tap MNIST test: {exc}")
        return

    if len(mnist_test) == 0:
        st.error("Tap MNIST test rong, khong the suy luan.")
        return

    label_indices = [idx for idx, target in enumerate(mnist_test.targets) if int(target) == selected_digit]
    if not label_indices:
        st.error(f"Khong tim thay anh nao co nhan `{selected_digit}` trong tap test.")
        return
    sample_index = random.choice(label_indices)
    image_tensor, label = mnist_test[sample_index]

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
        probabilities = torch.exp(logits)
        confidence, prediction = torch.max(probabilities, dim=1)

    st.image(
        image_tensor.squeeze(0).numpy(),
        caption=f"Ảnh MNIST với nhãn đã chọn: {selected_digit}",
        use_container_width=False,
        clamp=True,
    )
    prediction_digit = int(prediction.item())
    confidence_percent = float(confidence.item()) * 100.0

    st.markdown(f"### Kết quả dự đoán: **{prediction_digit}** (Xác suất: **{confidence_percent:.2f}%**)")
    if prediction_digit == selected_digit:
        st.success("Dự đoán CHÍNH XÁC so với nhãn đã chọn.")
    else:
        st.warning(
            "Dự đoán KHÔNG KHỚP với nhãn đã chọn."
        )

    st.markdown("**Phân phối xác suất cho các lớp 0-9**")
    probability_values = probabilities.squeeze(0).detach().cpu().tolist()
    probability_df = pd.DataFrame(
        {
            "Digit": [str(i) for i in range(10)],
            "Probability": probability_values,
        }
    ).set_index("Digit")
    st.bar_chart(probability_df)


def detect_dashboard_mode(results_dir: str) -> str:
    client_history_path = os.path.join(results_dir, "experiment_global_history.json")
    if os.path.exists(client_history_path):
        return "client"
    return "server"


def find_server_client_payload_files(results_dir: str) -> List[str]:
    if not os.path.exists(results_dir):
        return []
    paths: List[str] = []
    for filename in os.listdir(results_dir):
        if SERVER_CLIENT_FILE_REGEX.match(filename):
            paths.append(os.path.join(results_dir, filename))
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths


def find_local_client_payload_files(results_dir: str) -> List[str]:
    if not os.path.exists(results_dir):
        return []
    paths: List[str] = []
    for filename in os.listdir(results_dir):
        if filename.startswith("parameter_inspector_client_") and filename.endswith(".json"):
            paths.append(os.path.join(results_dir, filename))
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths


def load_parameter_inspector_data(file_path: str) -> Dict[str, List[float]]:
    data = load_json_file(file_path)

    if "weights_clean" not in data or "weights_noisy" not in data:
        raise KeyError("File inspector thieu 'weights_clean' hoac 'weights_noisy'.")

    clean = [float(v) for v in data["weights_clean"][:100]]
    noisy = [float(v) for v in data["weights_noisy"][:100]]
    if len(clean) < 100 or len(noisy) < 100:
        raise ValueError("Can it nhat 100 gia tri cho moi nhom tham so.")

    return {
        "weights_clean": clean,
        "weights_noisy": noisy,
        "layer_name": str(data.get("layer_name", "fc2.weight")),
        "client_id": str(data.get("client_id", "N/A")),
    }


def reshape_to_10x10(values: List[float]) -> List[List[float]]:
    return [values[i * 10:(i + 1) * 10] for i in range(10)]


def build_server_client_comparison_dataframe(file_paths: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        matched = SERVER_CLIENT_FILE_REGEX.match(file_name)
        if matched is None:
            continue

        fallback_client_id, round_text = matched.groups()
        try:
            payload = load_json_file(file_path)
            clean_weights = payload.get("weights_clean", [])
            noisy_weights = payload.get("weights_noisy", [])
            pair_count = min(len(clean_weights), len(noisy_weights))
            avg_abs_noise = 0.0
            if pair_count > 0:
                avg_abs_noise = sum(
                    abs(float(noisy_weights[idx]) - float(clean_weights[idx]))
                    for idx in range(pair_count)
                ) / pair_count

            rows.append(
                {
                    "ClientID": str(payload.get("client_id", fallback_client_id)),
                    "Round": int(payload.get("server_round", int(round_text))),
                    "AvgAbsNoise": float(avg_abs_noise),
                    "LocalAccuracy": float(payload.get("history_accuracy", [0.0])[0]),
                    "LocalLoss": float(payload.get("history_loss", [0.0])[0]),
                }
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["Round", "ClientID"]).reset_index(drop=True)


def render_dp_inspector_tab(runtime_mode: str) -> None:

    if runtime_mode == "server":
        inspector_files = find_server_client_payload_files(RESULTS_DIR)
        if not inspector_files:
            st.info("Chua co file `client_{id}_round_{round}.json` trong `results/`.")
            return

        comparison_df = build_server_client_comparison_dataframe(inspector_files)
        if not comparison_df.empty:
            st.markdown("**So sánh Client theo Round (Server View)**")
            st.dataframe(comparison_df, use_container_width=True)
            chart_df = comparison_df.groupby("ClientID", as_index=True)["AvgAbsNoise"].mean().to_frame()
            st.markdown("**Average |Noisy - Clean| theo Client**")
            st.bar_chart(chart_df)
        else:
            st.warning("Khong the tao bang so sanh tu file client payload.")

        selected_inspector = st.selectbox(
            "Chon file payload client tren server",
            options=inspector_files,
            format_func=lambda p: os.path.basename(p),
        )
    else:
        inspector_files = find_local_client_payload_files(RESULTS_DIR)
        if not inspector_files:
            st.info("Chua co file local `parameter_inspector_client_*.json`.")
            return

        selected_inspector = st.selectbox(
            "Chọn file Parameter Inspector local",
            options=inspector_files,
            format_func=lambda p: os.path.basename(p),
        )

    if not inspector_files:
        st.info(
            "Chua co file Parameter Inspector. "
            "Hay chay client de tao `results/parameter_inspector_client_*.json`."
        )
        return

    try:
        inspector_data = load_parameter_inspector_data(selected_inspector)
    except Exception as exc:
        st.error(f"Khong the tai du lieu Parameter Inspector: {exc}")
        return

    clean_matrix = reshape_to_10x10(inspector_data["weights_clean"])
    noisy_matrix = reshape_to_10x10(inspector_data["weights_noisy"])

    st.write(
        f"Layer minh họa: `{inspector_data['layer_name']}` | "
        f"Client: `{inspector_data['client_id']}`"
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Tham số gốc Raw Parameters**")
        fig_clean, ax_clean = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            clean_matrix,
            cmap="Blues",
            center=0.0,
            ax=ax_clean,
            cbar=True,
            square=True,
        )
        ax_clean.set_xlabel("Column")
        ax_clean.set_ylabel("Row")
        st.pyplot(fig_clean, use_container_width=True)
        plt.close(fig_clean)

    with col_right:
        st.markdown("**Tham số đã thêm nhiễu DP Noisy Parameters**")
        fig_noisy, ax_noisy = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            noisy_matrix,
            cmap="Reds",
            center=0.0,
            ax=ax_noisy,
            cbar=True,
            square=True,
        )
        ax_noisy.set_xlabel("Column")
        ax_noisy.set_ylabel("Row")
        st.pyplot(fig_noisy, use_container_width=True)
        plt.close(fig_noisy)

    st.markdown("**Phân phối tham số: Raw vs DP Noisy**")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    sns.histplot(
        inspector_data["weights_clean"],
        color="#1f77b4",
        alpha=0.55,
        kde=True,
        label="Raw Parameters",
        ax=ax_hist,
    )
    sns.histplot(
        inspector_data["weights_noisy"],
        color="#d62728",
        alpha=0.5,
        kde=True,
        label="DP Noisy Parameters",
        ax=ax_hist,
    )
    ax_hist.set_xlabel("Weight Value")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend()
    st.pyplot(fig_hist, use_container_width=True)
    plt.close(fig_hist)


def main() -> None:
    ensure_results_dir(RESULTS_DIR)
    st.set_page_config(
        page_title="Federated Learning & Differential Privacy Dashboard",
        layout="wide",
    )

    st.title("Federated Learning & Differential Privacy Dashboard")

    st.sidebar.header("Data & Simulation Controls")
    st.sidebar.caption("Mở tab kiểm chứng mô hình để chọn nhãn và kiểm tra dự đoán trên ảnh MNIST tương ứng.")
    runtime_mode = detect_dashboard_mode(RESULTS_DIR)

    overview_tab, dp_tab, inference_tab = st.tabs(
        ["Monitoring FL+DP", "Cơ chế bảo mật DP", " Kiểm chứng mô hình"]
    )

    with overview_tab:
        st.subheader("So sánh kết quả huấn luyện giữa DP vs No-DP")
        render_dp_nodp_comparison_charts(RESULTS_DIR)

    with dp_tab:
        render_dp_inspector_tab(runtime_mode=runtime_mode)

    with inference_tab:
        render_live_inference_section()


if __name__ == "__main__":
    main()
