import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
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


if __name__ == "__main__":
    main()
