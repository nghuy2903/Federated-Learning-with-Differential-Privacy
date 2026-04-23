import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

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
        raise ValueError("Khong trich xuat duoc so round tu ten file.")

    accuracy = pick_metric("history_accuracy", round_number)
    loss = pick_metric("history_loss", round_number)
    epsilon = pick_metric("history_epsilon", round_number)

    if accuracy is None or loss is None:
        raise KeyError("File JSON thieu 'history_accuracy' hoac 'history_loss' hop le.")

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
            st.warning(f"Bo qua file loi `{os.path.basename(file_path)}`: {exc}")
            continue

        rows.append(
            {
                "Round": round_number,
                "Accuracy": float(metrics["accuracy"]),
                "Loss": float(metrics["loss"]),
                "EpsilonPerRound": metrics["epsilon"],
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(by="Round").reset_index(drop=True)

    # Missing epsilon is treated as 0 to keep cumulative budget stable.
    df["EpsilonPerRound"] = pd.to_numeric(df["EpsilonPerRound"], errors="coerce").fillna(0.0)
    df["CumulativeEpsilon"] = df["EpsilonPerRound"].cumsum()
    return df


def build_client_history_dataframe(results_dir: str) -> pd.DataFrame:
    history_path = os.path.join(results_dir, "experiment_global_history.json")
    if not os.path.exists(history_path):
        return pd.DataFrame()

    try:
        history_payload = load_json_file(history_path)
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Khong the doc `experiment_global_history.json`: {exc}")
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
                "EpsilonPerRound": float(epsilon_history[idx]) if idx < len(epsilon_history) else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df["CumulativeEpsilon"] = df["EpsilonPerRound"].cumsum()
    return df


@st.cache_resource(show_spinner=False)
def load_global_model(model_path: str) -> Net:
    model = Net()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
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
    st.subheader("🚀 Kiểm chứng Mô hình (Inference)")
    st.write("Chọn chữ số (0-9) để kiểm tra mô hình trên một ảnh MNIST tương ứng.")

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

    st.markdown(f"### Kết quả dự đoán: **{prediction_digit}** (Độ tự tin: **{confidence_percent:.2f}%**)")
    if prediction_digit == selected_digit:
        st.success("Dự đoán CHÍNH XÁC so với nhãn đã chọn.")
    else:
        st.warning(
            "Dự đoán KHÔNG KHỚP với nhãn đã chọn. Điều này có thể xảy ra khi mô hình còn sai số hoặc bị ảnh hưởng bởi nhiễu DP."
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
    st.subheader("Cơ chế bảo mật DP")

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
            "Chon file Parameter Inspector local",
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
        f"Layer minh hoa: `{inspector_data['layer_name']}` | "
        f"Client: `{inspector_data['client_id']}`"
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Tham số gốc (Raw Parameters)**")
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
        st.markdown("**Tham số đã thêm nhiễu (DP Noisy Parameters)**")
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
    st.caption("Real-time simulation for advisor meeting (FL + DP training history)")

    st.sidebar.header("Data & Simulation Controls")
    st.sidebar.markdown("### 🚀 Kiểm chứng Mô hình (Inference)")
    st.sidebar.caption("Mở tab Inference để chọn nhãn và kiểm tra dự đoán trên ảnh MNIST tương ứng.")
    runtime_mode = detect_dashboard_mode(RESULTS_DIR)
    st.sidebar.caption(f"Mode: {'Client View' if runtime_mode == 'client' else 'Server View'}")

    df = build_client_history_dataframe(RESULTS_DIR) if runtime_mode == "client" else build_server_history_dataframe(RESULTS_DIR)

    try:
        total_rounds = int(df["Round"].max()) if not df.empty else 0
    except (ValueError, TypeError, KeyError):
        total_rounds = 0

    if total_rounds > 1:
        current_round = st.sidebar.slider(
            "Simulate Training Round",
            min_value=1,
            max_value=total_rounds,
            value=total_rounds,
            step=1,
        )
    elif total_rounds == 1:
        st.sidebar.info("Showing Round 1")
        current_round = 1
    else:
        current_round = 0

    if current_round > 0:
        current_idx = current_round - 1
        df_current = df.iloc[:current_round].copy()

        current_accuracy = float(df.iloc[current_idx]["Accuracy"])
        current_loss = float(df.iloc[current_idx]["Loss"])
        current_epsilon = float(df.iloc[current_idx]["CumulativeEpsilon"])
    else:
        df_current = pd.DataFrame()
        current_accuracy = 0.0
        current_loss = 0.0
        current_epsilon = 0.0

    overview_tab, dp_tab, inference_tab = st.tabs(
        ["Monitoring FL+DP", "Cơ chế bảo mật DP", "🚀 Kiểm chứng Mô hình (Inference)"]
    )

    with overview_tab:
        if current_round == 0:
            st.warning(
                "Khong tim thay du lieu history hop le trong `results/` "
                f"voi pattern `{ROUND_FILE_PATTERN}` hoac `experiment_global_history.json`."
            )
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Round", f"{current_round}")
            col2.metric("Accuracy", f"{current_accuracy:.2%}")
            col3.metric("Privacy Budget (Epsilon)", f"{current_epsilon:.4f}")
            col4.metric("Global Loss", f"{current_loss:.6f}")

            st.divider()

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.subheader("Model Convergence")
                st.line_chart(
                    data=df_current.set_index("Round")[["Accuracy", "Loss"]],
                    use_container_width=True,
                )

            with chart_col2:
                st.subheader("Privacy Budget Consumption")
                st.line_chart(
                    data=df_current.set_index("Round")[["CumulativeEpsilon"]],
                    use_container_width=True,
                )

            st.caption(
                f"Displaying rounds 1 to {current_round} from files "
                f"`{ROUND_FILE_PATTERN}` (max rounds available: {total_rounds})."
            )

    with dp_tab:
        render_dp_inspector_tab(runtime_mode=runtime_mode)

    with inference_tab:
        render_live_inference_section()


if __name__ == "__main__":
    main()
