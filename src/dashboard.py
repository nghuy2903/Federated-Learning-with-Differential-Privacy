import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


RESULTS_DIR = Path("results")
REQUIRED_KEYS = ("history_accuracy", "history_epsilon", "history_loss")


def find_result_files(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_history(json_path: Path) -> Dict[str, List[float]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(f"File JSON thieu truong: {', '.join(missing)}")

    history: Dict[str, List[float]] = {}
    for key in REQUIRED_KEYS:
        values = data[key]
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"'{key}' phai la list khong rong.")
        history[key] = [float(v) for v in values]

    min_len = min(len(history["history_accuracy"]), len(history["history_epsilon"]), len(history["history_loss"]))
    if min_len == 0:
        raise ValueError("Khong co du lieu hop le de hien thi.")

    if not (
        len(history["history_accuracy"])
        == len(history["history_epsilon"])
        == len(history["history_loss"])
    ):
        st.warning(
            f"Do dai du lieu khong dong nhat. Dashboard se su dung {min_len} round dau tien."
        )

    for key in REQUIRED_KEYS:
        history[key] = history[key][:min_len]

    return history


def build_dataframe(history: Dict[str, List[float]]) -> pd.DataFrame:
    rounds = list(range(1, len(history["history_accuracy"]) + 1))
    epsilon_per_round = history["history_epsilon"]

    cumulative_epsilon: List[float] = []
    running = 0.0
    for eps in epsilon_per_round:
        running += eps
        cumulative_epsilon.append(running)

    return pd.DataFrame(
        {
            "Round": rounds,
            "Accuracy": history["history_accuracy"],
            "Loss": history["history_loss"],
            "EpsilonPerRound": epsilon_per_round,
            "CumulativeEpsilon": cumulative_epsilon,
        }
    )


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

    result_files = find_result_files(RESULTS_DIR)
    if not result_files:
        st.error("Khong tim thay file JSON trong thu muc 'results/'. Vui long chay train truoc.")
        st.stop()

    st.sidebar.header("Data & Simulation Controls")
    selected_file = st.sidebar.selectbox(
        "Select Training History JSON",
        options=result_files,
        index=0,
        format_func=lambda p: p.name,
    )

    try:
        history = load_history(selected_file)
        df = build_dataframe(history)
    except Exception as exc:
        st.error(f"Khong the tai du lieu tu file da chon: {exc}")
        st.stop()

    max_round = int(df["Round"].max())
    current_round = st.sidebar.slider(
        "Simulate Training Round",
        min_value=1,
        max_value=max_round,
        value=max_round,
        step=1,
    )

    current_idx = current_round - 1
    df_current = df.iloc[:current_round].copy()

    current_accuracy = float(df.iloc[current_idx]["Accuracy"])
    current_loss = float(df.iloc[current_idx]["Loss"])
    current_epsilon = float(df.iloc[current_idx]["CumulativeEpsilon"])

    overview_tab, dp_tab = st.tabs(["Monitoring FL+DP", "Cơ chế bảo mật DP"])

    with overview_tab:
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
            f"Displaying rounds 1 to {current_round} from `{selected_file.name}` "
            f"(max rounds available: {max_round})."
        )

    with dp_tab:
        render_dp_inspector_tab()


if __name__ == "__main__":
    main()
