import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from torchvision import datasets, transforms

from model import Net


RESULTS_DIR = Path("results")
ROUND_FILE_PATTERN = "parameter_inspector_server_round_*.json"


def find_result_files(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    return list(results_dir.glob(ROUND_FILE_PATTERN))


def extract_round_number(json_path: Path) -> Optional[int]:
    stem = json_path.stem
    prefix = "parameter_inspector_server_round_"
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix):]
    try:
        round_number = int(suffix)
    except ValueError:
        return None
    return round_number if round_number > 0 else None


def load_round_metrics(json_path: Path) -> Dict[str, Optional[float]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

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

    round_number = extract_round_number(json_path)
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

def build_history_dataframe(results_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Optional[float]]] = []
    files = find_result_files(results_dir)

    if not files:
        return pd.DataFrame()

    for json_path in files:
        round_number = extract_round_number(json_path)
        if round_number is None:
            continue

        try:
            metrics = load_round_metrics(json_path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            st.warning(f"Bo qua file loi `{json_path.name}`: {exc}")
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


@st.cache_resource(show_spinner=False)
def load_global_model(model_path: Path) -> Net:
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
    st.write("Nhấn nút để chạy suy luận với 1 ảnh MNIST ngẫu nhiên từ tập test.")

    if not st.button("Dự đoán ảnh MNIST ngẫu nhiên", use_container_width=True):
        return

    model_path = RESULTS_DIR / "global_model_latest.pth"
    if not model_path.exists():
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

    sample_index = random.randint(0, len(mnist_test) - 1)
    image_tensor, label = mnist_test[sample_index]

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
        probabilities = torch.exp(logits)
        confidence, prediction = torch.max(probabilities, dim=1)

    st.image(
        image_tensor.squeeze(0).numpy(),
        caption=f"Anh test MNIST ngau nhien (label that: {label})",
        use_container_width=False,
        clamp=True,
    )
    st.success(f"AI du doan day la so: **{int(prediction.item())}**")
    st.info(f"Do tu tin (Probability): **{float(confidence.item()):.2%}**")


def find_parameter_inspector_files(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    return sorted(
        results_dir.glob("parameter_inspector_client_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_parameter_inspector_data(json_path: Path) -> Dict[str, List[float]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

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


def render_dp_inspector_tab() -> None:
    st.subheader("Cơ chế bảo mật DP")
    inspector_files = find_parameter_inspector_files(RESULTS_DIR)
    if not inspector_files:
        st.info(
            "Chua co file Parameter Inspector. "
            "Hay chay client de tao `results/parameter_inspector_client_*.json`."
        )
        return

    selected_inspector = st.selectbox(
        "Chon file Parameter Inspector",
        options=inspector_files,
        format_func=lambda p: p.name,
    )

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
    st.set_page_config(
        page_title="Federated Learning & Differential Privacy Dashboard",
        layout="wide",
    )

    st.title("Federated Learning & Differential Privacy Dashboard")
    st.caption("Real-time simulation for advisor meeting (FL + DP training history)")

    st.sidebar.header("Data & Simulation Controls")
    st.sidebar.markdown("### 🚀 Kiểm chứng Mô hình (Inference)")
    st.sidebar.caption("Mo tab Inference de du doan anh MNIST ngau nhien.")

    df = build_history_dataframe(RESULTS_DIR)

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
                f"voi pattern `{ROUND_FILE_PATTERN}`."
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
        render_dp_inspector_tab()

    with inference_tab:
        render_live_inference_section()


if __name__ == "__main__":
    main()
