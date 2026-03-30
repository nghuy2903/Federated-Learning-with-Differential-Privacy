import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns


def load_history(json_path: Path) -> Dict[str, List[float]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Khong tim thay file JSON: {json_path}")

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"File JSON khong hop le: {json_path}") from exc

    required_keys = ["history_accuracy", "history_epsilon", "history_loss"]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise KeyError(
            "JSON thieu truong bat buoc: " + ", ".join(missing_keys)
        )

    history = {}
    for key in required_keys:
        values = data[key]
        if not isinstance(values, list):
            raise TypeError(f"Gia tri cua '{key}' phai la list.")
        if len(values) == 0:
            raise ValueError(f"Du lieu '{key}' rong, khong the ve bieu do.")

        try:
            history[key] = [float(v) for v in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Du lieu trong '{key}' phai chuyen duoc sang so.") from exc

    return history


def prepare_series(history: Dict[str, List[float]]) -> Dict[str, List[float]]:
    accuracy = history["history_accuracy"]
    epsilon_spend = history["history_epsilon"]
    loss = history["history_loss"]

    min_len = min(len(accuracy), len(epsilon_spend), len(loss))
    if min_len == 0:
        raise ValueError("Khong co du lieu hop le de ve bieu do.")

    if not (len(accuracy) == len(epsilon_spend) == len(loss)):
        print(
            "[Canh bao] Do dai cac chuoi khong dong nhat. "
            f"Se cat ve {min_len} vong dau tien."
        )

    accuracy = accuracy[:min_len]
    epsilon_spend = epsilon_spend[:min_len]
    loss = loss[:min_len]
    rounds = list(range(1, min_len + 1))

    cumulative_epsilon = []
    running_sum = 0.0
    for eps in epsilon_spend:
        running_sum += eps
        cumulative_epsilon.append(running_sum)

    return {
        "rounds": rounds,
        "accuracy": accuracy,
        "loss": loss,
        "cumulative_epsilon": cumulative_epsilon,
    }


def create_figure(series: Dict[str, List[float]], output_path: Path) -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
            "figure.dpi": 120,
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # Subplot 1: Accuracy vs Privacy Budget
    ax1 = axes[0]
    sns.lineplot(
        x=series["cumulative_epsilon"],
        y=series["accuracy"],
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        ax=ax1,
        label="Global Accuracy",
    )
    ax1.set_title(r"Accuracy vs. Privacy Budget ($\epsilon$)", pad=12)
    ax1.set_xlabel(r"Cumulative $\epsilon$")
    ax1.set_ylabel("Global Accuracy")
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax1.legend(loc="best", frameon=True)

    final_eps = series["cumulative_epsilon"][-1]
    final_acc = series["accuracy"][-1]
    ax1.annotate(
        f"Final ({final_eps:.3g}, {final_acc:.3f})",
        xy=(final_eps, final_acc),
        xytext=(15, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#555555", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
    )

    # Subplot 2: Convergence with dual axis
    ax2 = axes[1]
    line_acc = sns.lineplot(
        x=series["rounds"],
        y=series["accuracy"],
        marker="o",
        linewidth=2.2,
        color="#2ca02c",
        ax=ax2,
        label="Global Accuracy",
    )
    ax2.set_title("Model Convergence (MNIST + Non-IID)", pad=12)
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Global Accuracy", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    ax2b = ax2.twinx()
    line_loss = sns.lineplot(
        x=series["rounds"],
        y=series["loss"],
        linestyle=":",
        linewidth=2.4,
        color="#d62728",
        ax=ax2b,
        label="Global Loss",
        legend= False
    )
    ax2b.set_ylabel("Global Loss", color="#d62728")
    ax2b.tick_params(axis="y", labelcolor="#d62728")
    ax2b.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2b.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    lines = [line_acc.lines[0], line_loss.lines[0]]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="best", frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ve bieu do ket qua huan luyen Federated Learning."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Duong dan file JSON ket qua, vi du results/experiment_20rounds_xxx.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/advisor_meeting_plots.png",
        help="Duong dan anh dau ra (mac dinh: results/advisor_meeting_plots.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        history = load_history(input_path)
        series = prepare_series(history)
        create_figure(series, output_path)
        print(f"[OK] Da luu bieu do tai: {output_path}")
    except Exception as exc:  # Friendly error for CLI usage
        print(f"[Loi] Khong the tao bieu do: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
