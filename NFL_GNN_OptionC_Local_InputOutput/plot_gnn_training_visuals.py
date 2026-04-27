import argparse
import json
import os
import re
from typing import Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches

from nfl_trajectory_gnn_pipeline import ModelConfig, OBS_FEATURE_NAMES


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _find_first_existing(paths) -> Optional[str]:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _load_config_from_artifact_dir(artifact_dir: Optional[str]) -> ModelConfig:
    config = ModelConfig()
    if not artifact_dir:
        return config

    config_path = os.path.join(artifact_dir, "config.json")
    if not os.path.exists(config_path):
        return config

    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for key, value in payload.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def _resolve_training_history_path(artifact_dir: Optional[str]) -> Optional[str]:
    if not artifact_dir:
        return None
    return _find_first_existing(
        [
            os.path.join(artifact_dir, "training_history.csv"),
            os.path.join(artifact_dir, "training_history_partial.csv"),
        ]
    )


def _resolve_holdout_summary_paths(holdout_dir: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not holdout_dir:
        return None, None
    csv_path = _find_first_existing([os.path.join(holdout_dir, "holdout_summary.csv")])
    json_path = _find_first_existing([os.path.join(holdout_dir, "holdout_summary.json")])
    return csv_path, json_path


def _extract_week_number(week_key: str) -> Optional[int]:
    match = re.search(r"_w(\d{2})", str(week_key).lower())
    if not match:
        return None
    return int(match.group(1))


def _apply_academic_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "grid.color": "#d0d0d0",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.45,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def plot_training_curve(training_history_path: str, output_path: str) -> None:
    history = pd.read_csv(training_history_path)
    if history.empty:
        raise ValueError(f"Training history is empty: {training_history_path}")

    _apply_academic_style()
    epochs = history["epoch"]
    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax2 = ax1.twinx()

    if "train_loss" in history.columns:
        ax1.plot(
            epochs,
            history["train_loss"],
            color="#1f77b4",
            linewidth=2.1,
            marker="o",
            markersize=4.2,
            label="Train Loss",
        )
    if "val_loss" in history.columns:
        ax1.plot(
            epochs,
            history["val_loss"],
            color="#ff7f0e",
            linewidth=2.1,
            marker="s",
            markersize=4.2,
            label="Validation Loss",
        )
    if "val_kaggle_rmse" in history.columns:
        ax2.plot(
            epochs,
            history["val_kaggle_rmse"],
            color="#2ca02c",
            linewidth=2.2,
            marker="^",
            markersize=4.8,
            label="Validation RMSE",
        )

    ax1.set_title("GNN Training Curve", pad=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("RMSE")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.grid(axis="y")
    ax1.grid(axis="x", alpha=0.18)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles1 or handles2:
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_rmse_per_week(holdout_csv_path: str, holdout_json_path: Optional[str], output_path: str) -> None:
    summary = pd.read_csv(holdout_csv_path)
    if summary.empty:
        raise ValueError(f"Holdout summary is empty: {holdout_csv_path}")

    summary["week_num"] = summary["week_key"].map(_extract_week_number)
    summary = summary.sort_values(["week_num", "week_key"]).reset_index(drop=True)

    overall_rmse = None
    if holdout_json_path and os.path.exists(holdout_json_path):
        with open(holdout_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        overall_rmse = payload.get("overall_kaggle_rmse")

    _apply_academic_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bars = ax.bar(
        summary["week_key"],
        summary["kaggle_rmse"],
        color="#4c78a8",
        edgecolor="#1f1f1f",
        linewidth=0.8,
        alpha=0.88,
    )

    for bar, value in zip(bars, summary["kaggle_rmse"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.002,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    if overall_rmse is not None:
        ax.axhline(
            overall_rmse,
            color="#b22222",
            linestyle="--",
            linewidth=1.8,
            label=f"Overall RMSE = {overall_rmse:.4f}",
        )
        ax.legend(loc="upper right", frameon=True)

    ax.set_title("Weekly Holdout RMSE", pad=12)
    ax.set_xlabel("Holdout Week")
    ax.set_ylabel("Kaggle-style RMSE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    color: str,
    *,
    fontsize: float = 10.5,
    weight: str = "bold",
    edgecolor: str = "#222222",
) -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2.0, y + h / 2.0, label, ha="center", va="center", fontsize=fontsize, weight=weight)


def _add_stage_panel(ax, x: float, y: float, w: float, h: float, title: str, subtitle: str) -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.0,
        edgecolor="#d4d4d4",
        facecolor="#fafafa",
        zorder=0,
    )
    ax.add_patch(rect)
    ax.text(x + 0.015, y + h - 0.03, title, ha="left", va="center", fontsize=10.5, weight="bold", color="#111827")
    ax.text(x + 0.015, y + h - 0.055, subtitle, ha="left", va="center", fontsize=8.8, color="#6b7280")


def _arrow(ax, x1: float, y1: float, x2: float, y2: float, text: Optional[str] = None) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"),
    )
    if text:
        ax.text((x1 + x2) / 2.0, (y1 + y2) / 2.0 + 0.025, text, ha="center", va="bottom", fontsize=9, color="#333333")


def plot_graph_schematic(config: ModelConfig, output_path: str) -> None:
    _apply_academic_style()
    fig, ax = plt.subplots(figsize=(15.6, 6.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fill_main = "#eeeeee"
    fill_aux = "#f7f7f7"
    edge = "#2f2f2f"

    y_main = 0.56
    h_main = 0.125
    blocks = [
        (0.04, 0.130, "Tracking Input\n$x,y,v,a,\\mathrm{dir},\\mathrm{o}$", 13.8),
        (0.215, 0.225, "Graph Construction\naligned kinematics and interaction graph", 11.8),
        (0.490, 0.120, "Feature Encoder\nnode and edge embeddings", 12.8),
        (0.660, 0.165, "Spatial-Temporal Encoder\ngraph attention and motion history", 11.4),
        (0.875, 0.100, "Trajectory Decoder\nfuture coordinate rollout", 11.6),
    ]

    for x, width, label, fontsize in blocks:
        _add_box(
            ax,
            x,
            y_main,
            width,
            h_main,
            label,
            fill_main,
            fontsize=fontsize,
            edgecolor=edge,
        )

    for idx in range(len(blocks) - 1):
        x1, w1 = blocks[idx][0], blocks[idx][1]
        x2 = blocks[idx + 1][0]
        _arrow(ax, x1 + w1 + 0.006, y_main + h_main / 2.0, x2 - 0.006, y_main + h_main / 2.0, None)

    nodefeat_x, nodefeat_y, nodefeat_w, nodefeat_h = 0.455, 0.24, 0.18, 0.16
    edgefeat_x, edgefeat_y, edgefeat_w, edgefeat_h = 0.705, 0.24, 0.18, 0.16

    _add_box(
        ax,
        nodefeat_x,
        nodefeat_y,
        nodefeat_w,
        nodefeat_h,
        "Node Features\nposition, velocity, acceleration, role",
        fill_aux,
        fontsize=12.0,
        edgecolor=edge,
    )
    _add_box(
        ax,
        edgefeat_x,
        edgefeat_y,
        edgefeat_w,
        edgefeat_h,
        "Edge Features\nrelative distance, velocity, and closing speed",
        fill_aux,
        fontsize=11.7,
        edgecolor=edge,
    )

    encoder_center_x = blocks[2][0] + blocks[2][1] / 2.0
    encoder_bottom_y = y_main
    nodefeat_top_x = nodefeat_x + nodefeat_w / 2.0
    edgefeat_top_x = edgefeat_x + edgefeat_w / 2.0
    feature_join_y = 0.475

    ax.plot(
        [nodefeat_top_x, nodefeat_top_x, encoder_center_x],
        [nodefeat_y + nodefeat_h, feature_join_y, feature_join_y],
        linestyle="--",
        linewidth=1.5,
        color="#666666",
    )
    ax.plot(
        [edgefeat_top_x, edgefeat_top_x, encoder_center_x],
        [edgefeat_y + edgefeat_h, feature_join_y, feature_join_y],
        linestyle="--",
        linewidth=1.5,
        color="#666666",
    )
    ax.plot(
        [encoder_center_x, encoder_center_x],
        [feature_join_y, encoder_bottom_y],
        linestyle="--",
        linewidth=1.5,
        color="#666666",
    )

    ax.text(0.5, 0.955, "GNN Graph Schematic", ha="center", va="center", fontsize=18, weight="bold", color="#1f2937")
    ax.text(
        0.5,
        0.915,
        "Architecture of the proposed GNN-based trajectory prediction framework.",
        ha="center",
        va="center",
        fontsize=10.8,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GNN training visualizations from saved training records and model structure.")
    parser.add_argument("--artifact_dir", default="artifacts_recommended_1_13_14_18", help="Directory containing training_history.csv and config.json")
    parser.add_argument("--holdout_dir", default="holdout_predictions_recommended_14_18", help="Directory containing holdout_summary.csv/json")
    parser.add_argument("--output_dir", default="gnn_visualizations", help="Directory to save output figures")
    args = parser.parse_args()

    output_dir = _ensure_dir(os.path.abspath(args.output_dir))
    artifact_dir = args.artifact_dir if os.path.exists(args.artifact_dir) else None
    holdout_dir = args.holdout_dir if os.path.exists(args.holdout_dir) else None

    config = _load_config_from_artifact_dir(artifact_dir)
    history_path = _resolve_training_history_path(artifact_dir)
    holdout_csv_path, holdout_json_path = _resolve_holdout_summary_paths(holdout_dir)

    outputs: Dict[str, str] = {}

    if history_path:
        training_curve_path = os.path.join(output_dir, "gnn_training_curve.png")
        plot_training_curve(history_path, training_curve_path)
        outputs["training_curve"] = training_curve_path
    else:
        print("[WARN] No training history CSV found. Skipping training curve.")

    if holdout_csv_path:
        rmse_bar_path = os.path.join(output_dir, "gnn_rmse_per_week.png")
        plot_rmse_per_week(holdout_csv_path, holdout_json_path, rmse_bar_path)
        outputs["rmse_per_week"] = rmse_bar_path
    else:
        print("[WARN] No holdout_summary.csv found. Skipping weekly RMSE bar chart.")

    graph_path = os.path.join(output_dir, "gnn_graph_schematic.png")
    plot_graph_schematic(config, graph_path)
    outputs["graph_schematic"] = graph_path

    print("Generated figures:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
