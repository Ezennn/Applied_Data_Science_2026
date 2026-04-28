from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

REQUIRED_INPUT_COLUMNS = {
    "game_id",
    "play_id",
    "nfl_id",
    "frame_id",
    "player_to_predict",
    "x",
    "y",
    "s",
    "a",
    "dir",
    "o",
    "num_frames_output",
    "ball_land_x",
    "ball_land_y",
}

REQUIRED_OUTPUT_COLUMNS = {"game_id", "play_id", "nfl_id", "frame_id", "x", "y"}

# Metadata columns are kept for interpretation/evaluation tables only.
# They are not used as raw model features.
OPTIONAL_PLAYER_METADATA_COLUMNS = [
    "display_name",
    "player_name",
    "position",
    "player_position",
    "player_role",
    "role",
    "player_side",
    "side",
    "club",
    "team",
    "team_abbr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a simple XGBoost baseline for NFL Big Data Bowl 2026 "
            "using preprocessed standardized files from weeks 1-13."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="train_preprocessed",
        help="Directory containing input_2023_wXX_standardized.csv and output_2023_wXX_standardized.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_output",
        help="Directory to save metrics, predictions, models, and metadata",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="+",
        default=list(range(1, 14)),
        help="Weeks to load. Default: 1 2 ... 13",
    )
    parser.add_argument(
        "--val-weeks",
        type=int,
        nargs="+",
        default=[13],
        help="Held-out validation weeks. Default: 13",
    )
    parser.add_argument(
        "--last-k",
        type=int,
        default=5,
        help="How many final input frames to use as history features. Default: 5",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for XGBoost. Default: 42",
    )
    parser.add_argument(
        "--top-n-player-table",
        type=int,
        default=5,
        help="How many best/worst player-level rows to print. Default: 5",
    )
    parser.add_argument(
        "--max-horizon-plot",
        type=int,
        default=20,
        help=(
            "Maximum prediction horizon to show in the horizon plot. "
            "Use 0 to plot all horizons. Default: 20"
        ),
    )
    parser.add_argument(
        "--min-horizon-count-ratio",
        type=float,
        default=0.10,
        help=(
            "Only plot horizons with at least this fraction of the maximum horizon sample count. "
            "Default: 0.10"
        ),
    )
    return parser.parse_args()


def check_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def add_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["dir", "o"]:
        radians = np.deg2rad(out[col].astype(float))
        out[f"{col}_sin"] = np.sin(radians)
        out[f"{col}_cos"] = np.cos(radians)
    return out


def _pad_to_last_k(group: pd.DataFrame, last_k: int) -> pd.DataFrame:
    group = group.sort_values("frame_id").reset_index(drop=True)
    if len(group) >= last_k:
        return group.iloc[-last_k:].reset_index(drop=True)

    first_row = group.iloc[[0]].copy()
    pad_count = last_k - len(group)
    pad = pd.concat([first_row] * pad_count, ignore_index=True)
    return pd.concat([pad, group], ignore_index=True).reset_index(drop=True)


def _history_features(player_hist: pd.DataFrame, last_k: int) -> dict:
    hist = _pad_to_last_k(player_hist, last_k)
    hist = hist.sort_values("frame_id").reset_index(drop=True)

    last_row = hist.iloc[-1]
    first_row = hist.iloc[0]

    feat = {
        "last_x": float(last_row["x"]),
        "last_y": float(last_row["y"]),
        "last_s": float(last_row["s"]),
        "last_a": float(last_row["a"]),
        "ball_land_x": float(last_row["ball_land_x"]),
        "ball_land_y": float(last_row["ball_land_y"]),
        "land_dx": float(last_row["ball_land_x"] - last_row["x"]),
        "land_dy": float(last_row["ball_land_y"] - last_row["y"]),
        "num_frames_output": float(last_row["num_frames_output"]),
        "mean_s_last_k": float(hist["s"].mean()),
        "mean_a_last_k": float(hist["a"].mean()),
        "delta_x_over_window": float(last_row["x"] - first_row["x"]),
        "delta_y_over_window": float(last_row["y"] - first_row["y"]),
    }

    if len(hist) >= 2:
        prev_row = hist.iloc[-2]
        feat["delta_x_last_step"] = float(last_row["x"] - prev_row["x"])
        feat["delta_y_last_step"] = float(last_row["y"] - prev_row["y"])
    else:
        feat["delta_x_last_step"] = 0.0
        feat["delta_y_last_step"] = 0.0

    if "absolute_yardline_number" in hist.columns:
        feat["absolute_yardline_number"] = float(last_row["absolute_yardline_number"])

    if "was_flipped" in hist.columns:
        feat["was_flipped"] = float(bool(last_row["was_flipped"]))



    if "player_weight" in hist.columns:
        feat["player_weight"] = float(last_row["player_weight"])

    for i, row in hist.iterrows():
        prefix = f"hist_{i + 1}"
        feat[f"{prefix}_x_rel"] = float(row["x"] - last_row["x"])
        feat[f"{prefix}_y_rel"] = float(row["y"] - last_row["y"])
        feat[f"{prefix}_s"] = float(row["s"])
        feat[f"{prefix}_a"] = float(row["a"])
        feat[f"{prefix}_dir_sin"] = float(row["dir_sin"])
        feat[f"{prefix}_dir_cos"] = float(row["dir_cos"])
        feat[f"{prefix}_o_sin"] = float(row["o_sin"])
        feat[f"{prefix}_o_cos"] = float(row["o_cos"])

    return feat


def build_training_table(input_df: pd.DataFrame, output_df: pd.DataFrame, last_k: int = 5) -> pd.DataFrame:
    input_df = add_angle_features(input_df)
    input_df = input_df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"]).copy()
    output_df = output_df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"]).copy()

    target_input = input_df[input_df["player_to_predict"] == 1].copy()

    rows = []
    grouped_target_input = target_input.groupby(["game_id", "play_id", "nfl_id"], sort=False)
    grouped_output = output_df.groupby(["game_id", "play_id", "nfl_id"], sort=False)

    for key, player_hist in grouped_target_input:
        if key not in grouped_output.groups:
            continue

        future = grouped_output.get_group(key).sort_values("frame_id").reset_index(drop=True)
        feat = _history_features(player_hist, last_k=last_k)

        last_obs = player_hist.sort_values("frame_id").iloc[-1]
        last_x = float(last_obs["x"])
        last_y = float(last_obs["y"])
        week = int(last_obs["source_week"])

        game_id, play_id, nfl_id = key
        play_key = f"{game_id}_{play_id}"

        for _, future_row in future.iterrows():
            row = {
                "source_week": week,
                "game_id": game_id,
                "play_id": play_id,
                "nfl_id": nfl_id,
                "play_key": play_key,
                "future_frame_id": int(future_row["frame_id"]),
                "horizon_t": int(future_row["frame_id"]),
                "horizon_frac": float(future_row["frame_id"]) / max(float(feat["num_frames_output"]), 1.0),
                "target_x": float(future_row["x"]),
                "target_y": float(future_row["y"]),
                "target_dx": float(future_row["x"] - last_x),
                "target_dy": float(future_row["y"] - last_y),
            }
            for meta_col in OPTIONAL_PLAYER_METADATA_COLUMNS:
                if meta_col in last_obs.index:
                    row[meta_col] = last_obs[meta_col]

            row.update(feat)
            rows.append(row)

    if not rows:
        raise ValueError("No training rows were built. Check that input/output keys match.")

    return pd.DataFrame(rows)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    ignore = {
        "source_week",
        "game_id",
        "play_id",
        "nfl_id",
        "play_key",
        "future_frame_id",
        "target_x",
        "target_y",
        "target_dx",
        "target_dy",
    }
    ignore.update(OPTIONAL_PLAYER_METADATA_COLUMNS)

    # Keep the model input numeric. Metadata columns are kept only for evaluation output.
    return [
        c for c in df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
    ]


def make_xgb_model(random_state: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
    )

def compute_prediction_metrics(df: pd.DataFrame) -> dict:
    dx2 = (df["pred_x"] - df["target_x"]) ** 2
    dy2 = (df["pred_y"] - df["target_y"]) ** 2

    rmse_x = float(np.sqrt(np.mean(dx2)))
    rmse_y = float(np.sqrt(np.mean(dy2)))
    kaggle_score = float(np.sqrt(np.mean((dx2 + dy2) / 2.0)))

    return {
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "kaggle_score": kaggle_score,
        "n_rows": int(len(df)),
    }


def evaluate_validation_predictions(val_predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    overall = compute_prediction_metrics(val_predictions)
    overall["val_weeks"] = sorted(val_predictions["source_week"].unique().tolist())

    by_week_rows = []
    for week, group in val_predictions.groupby("source_week", sort=True):
        row = {"week": int(week)}
        row.update(compute_prediction_metrics(group))
        by_week_rows.append(row)

    by_week_df = pd.DataFrame(by_week_rows)
    return overall, by_week_df


def add_prediction_error_columns(val_predictions: pd.DataFrame) -> pd.DataFrame:
    """Add per-row error columns used for report-level evaluation."""
    out = val_predictions.copy()

    out["error_x"] = out["pred_x"] - out["target_x"]
    out["error_y"] = out["pred_y"] - out["target_y"]
    out["abs_error_x"] = out["error_x"].abs()
    out["abs_error_y"] = out["error_y"].abs()
    out["squared_error_x"] = out["error_x"] ** 2
    out["squared_error_y"] = out["error_y"] ** 2

    # Spatial error in yards. This is easier to interpret than coordinate-wise RMSE.
    out["euclidean_error"] = np.sqrt(out["squared_error_x"] + out["squared_error_y"])

    # Per-row contribution in the same coordinate scale as the official score.
    # The overall official score is computed after averaging squared errors, so this
    # column is mainly useful for sorting individual predictions, not replacing the
    # official aggregate score.
    out["row_coordinate_error"] = np.sqrt(
        (out["squared_error_x"] + out["squared_error_y"]) / 2.0
    )

    return out


def compute_error_distribution_summary(val_predictions_with_errors: pd.DataFrame) -> pd.DataFrame:
    """Summarise the distribution of point-level prediction errors."""
    df = val_predictions_with_errors

    summary = {
        "n_rows": int(len(df)),
        "best_euclidean_error": float(df["euclidean_error"].min()),
        "mean_euclidean_error": float(df["euclidean_error"].mean()),
        "median_euclidean_error": float(df["euclidean_error"].median()),
        "p75_euclidean_error": float(df["euclidean_error"].quantile(0.75)),
        "p90_euclidean_error": float(df["euclidean_error"].quantile(0.90)),
        "p95_euclidean_error": float(df["euclidean_error"].quantile(0.95)),
        "worst_euclidean_error": float(df["euclidean_error"].max()),
        "mean_abs_error_x": float(df["abs_error_x"].mean()),
        "mean_abs_error_y": float(df["abs_error_y"].mean()),
    }

    return pd.DataFrame([summary])


def compute_error_by_horizon(val_predictions_with_errors: pd.DataFrame) -> pd.DataFrame:
    """Evaluate how prediction error changes as the future horizon increases."""
    df = val_predictions_with_errors
    horizon_col = "horizon_t" if "horizon_t" in df.columns else "future_frame_id"

    rows = []
    for horizon, group in df.groupby(horizon_col, sort=True):
        row = {horizon_col: int(horizon)}
        row.update(compute_prediction_metrics(group))
        row.update(
            {
                "mean_euclidean_error": float(group["euclidean_error"].mean()),
                "median_euclidean_error": float(group["euclidean_error"].median()),
                "p90_euclidean_error": float(group["euclidean_error"].quantile(0.90)),
                "best_euclidean_error": float(group["euclidean_error"].min()),
                "worst_euclidean_error": float(group["euclidean_error"].max()),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def compute_error_by_play(val_predictions_with_errors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prediction errors at play level to identify best and worst plays."""
    df = val_predictions_with_errors

    group_cols = ["source_week", "game_id", "play_id"]
    if "play_key" in df.columns:
        group_cols.append("play_key")

    by_play = (
        df.groupby(group_cols, sort=False)
        .agg(
            n_predictions=("euclidean_error", "size"),
            mean_euclidean_error=("euclidean_error", "mean"),
            median_euclidean_error=("euclidean_error", "median"),
            p90_euclidean_error=("euclidean_error", lambda x: x.quantile(0.90)),
            max_euclidean_error=("euclidean_error", "max"),
            mean_abs_error_x=("abs_error_x", "mean"),
            mean_abs_error_y=("abs_error_y", "mean"),
            mean_row_coordinate_error=("row_coordinate_error", "mean"),
        )
        .reset_index()
    )

    return by_play

def compute_error_by_player(val_predictions_with_errors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prediction errors by player-play case for best/worst examples."""
    df = val_predictions_with_errors

    meta_cols = [c for c in OPTIONAL_PLAYER_METADATA_COLUMNS if c in df.columns]
    group_cols = ["source_week", "game_id", "play_id", "nfl_id"] + meta_cols

    rows = []
    for keys, group in df.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        metrics = compute_prediction_metrics(group)

        best_idx = group["euclidean_error"].idxmin()
        worst_idx = group["euclidean_error"].idxmax()
        best_row = group.loc[best_idx]
        worst_row = group.loc[worst_idx]

        row.update(
            {
                "n_predicted_frames": int(len(group)),
                "first_future_frame": int(group["future_frame_id"].min()),
                "last_future_frame": int(group["future_frame_id"].max()),
                "num_frames_output": int(round(float(group["num_frames_output"].max())))
                if "num_frames_output" in group.columns
                else int(len(group)),
                "official_score": float(metrics["kaggle_score"]),
                "mean_euclidean_error": float(group["euclidean_error"].mean()),
                "median_euclidean_error": float(group["euclidean_error"].median()),
                "max_euclidean_error": float(group["euclidean_error"].max()),
                "best_frame_id": int(best_row["future_frame_id"]),
                "best_frame_error": float(best_row["euclidean_error"]),
                "worst_frame_id": int(worst_row["future_frame_id"]),
                "worst_frame_error": float(worst_row["euclidean_error"]),
                "mean_abs_error_x": float(group["abs_error_x"].mean()),
                "mean_abs_error_y": float(group["abs_error_y"].mean()),
            }
        )
        rows.append(row)

    by_player = pd.DataFrame(rows)
    return by_player


def _format_extreme_player_table(df: pd.DataFrame, top_n: int, ascending: bool) -> pd.DataFrame:
    """Select and format best/worst player-play rows for console printing."""
    selected = df.sort_values("official_score", ascending=ascending).head(top_n).copy()
    selected.insert(0, "rank", range(1, len(selected) + 1))

    preferred_cols = [
        "rank",
        "source_week",
        "game_id",
        "play_id",
        "nfl_id",
        "display_name",
        "player_name",
        "position",
        "player_position",
        "player_role",
        "role",
        "player_side",
        "side",
        "club",
        "team",
        "team_abbr",
        "n_predicted_frames",
        "official_score",
        "mean_euclidean_error",
        "median_euclidean_error",
        "max_euclidean_error",
        "best_frame_id",
        "best_frame_error",
        "worst_frame_id",
        "worst_frame_error",
    ]
    preferred_cols = [c for c in preferred_cols if c in selected.columns]
    selected = selected[preferred_cols]

    numeric_cols = selected.select_dtypes(include=[np.number]).columns
    selected[numeric_cols] = selected[numeric_cols].round(4)
    return selected


def print_best_worst_player_tables(error_by_player_df: pd.DataFrame, top_n: int = 5) -> None:
    """Print best/worst player-play examples directly in the console; do not save them."""
    best_table = _format_extreme_player_table(error_by_player_df, top_n=top_n, ascending=True)
    worst_table = _format_extreme_player_table(error_by_player_df, top_n=top_n, ascending=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print(f"\nBest {top_n} player-play prediction cases by official score:")
    print(best_table.to_string(index=False))

    print(f"\nWorst {top_n} player-play prediction cases by official score:")
    print(worst_table.to_string(index=False))


def save_additional_evaluation_plots(
    val_predictions_with_errors: pd.DataFrame,
    error_by_horizon_df: pd.DataFrame,
    output_dir: Path,
    max_horizon_plot: int = 20,
    min_horizon_count_ratio: float = 0.10,
) -> None:
    """Save visualisations for the final report.

    The full horizon statistics are still saved to CSV. The plot is intentionally
    limited to horizons with enough examples, because very late horizons often have
    much smaller sample sizes and can make the trend look noisy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # The full distribution has a long tail. Clipping at the 99th percentile makes
    # the main body of the distribution easier to read while the CSV still keeps
    # the true maximum error.
    error_upper = float(val_predictions_with_errors["euclidean_error"].quantile(0.99))
    errors_to_plot = val_predictions_with_errors.loc[
        val_predictions_with_errors["euclidean_error"] <= error_upper,
        "euclidean_error",
    ]

    plt.figure(figsize=(8, 5))
    plt.hist(errors_to_plot, bins=50)
    plt.axvline(val_predictions_with_errors["euclidean_error"].median(), linestyle="--", label="Median")
    plt.axvline(val_predictions_with_errors["euclidean_error"].mean(), linestyle="-", label="Mean")
    plt.xlabel("Euclidean prediction error (yards)")
    plt.ylabel("Number of predictions")
    plt.title("Distribution of validation prediction errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution.png", dpi=300)
    plt.close()

    horizon_col = "horizon_t" if "horizon_t" in error_by_horizon_df.columns else "future_frame_id"
    horizon_plot_df = error_by_horizon_df.copy()

    if max_horizon_plot and max_horizon_plot > 0:
        horizon_plot_df = horizon_plot_df[horizon_plot_df[horizon_col] <= max_horizon_plot]

    max_count = horizon_plot_df["n_rows"].max() if len(horizon_plot_df) else 0
    min_count = max_count * min_horizon_count_ratio
    if max_count > 0:
        horizon_plot_df = horizon_plot_df[horizon_plot_df["n_rows"] >= min_count]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Bars show how many validation rows are available at each prediction horizon.
    bars = ax2.bar(
        horizon_plot_df[horizon_col],
        horizon_plot_df["n_rows"],
        alpha=0.20,
        label="Number of validation rows",
    )
    ax2.set_ylabel("Number of validation rows")

    # Line shows the official validation score at each prediction horizon.
    line, = ax1.plot(
        horizon_plot_df[horizon_col],
        horizon_plot_df["kaggle_score"],
        marker="o",
        label="Official validation score",
    )
    ax1.set_xlabel("Prediction horizon / future frame")
    ax1.set_ylabel("Official validation score")
    ax1.grid(True)

    # Keep the line visually in front of the bars.
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    title = "Validation score and sample size by prediction horizon"
    if max_horizon_plot and max_horizon_plot > 0:
        title += f" (up to frame {max_horizon_plot})"
    ax1.set_title(title)

    # Combined legend for line and bars.
    ax1.legend(
        [line, bars],
        [line.get_label(), bars.get_label()],
        loc="upper left"
    )

    fig.tight_layout()
    plt.savefig(output_dir / "error_by_prediction_horizon.png", dpi=300)
    plt.close()


def run_additional_evaluation(
    val_predictions: pd.DataFrame,
    output_dir: Path,
    top_n_player_table: int = 5,
    max_horizon_plot: int = 20,
    min_horizon_count_ratio: float = 0.10,
) -> dict:
    """Run report-oriented validation analysis beyond the official score."""
    print("Running additional validation analysis...")

    val_predictions_with_errors = add_prediction_error_columns(val_predictions)
    error_summary_df = compute_error_distribution_summary(val_predictions_with_errors)
    error_by_horizon_df = compute_error_by_horizon(val_predictions_with_errors)
    error_by_player_df = compute_error_by_player(val_predictions_with_errors)

    val_predictions_with_errors.to_csv(
        output_dir / "validation_predictions_with_errors.csv", index=False
    )
    error_summary_df.to_csv(output_dir / "error_distribution_summary.csv", index=False)
    error_by_horizon_df.to_csv(output_dir / "error_by_prediction_horizon.csv", index=False)

    save_additional_evaluation_plots(
        val_predictions_with_errors=val_predictions_with_errors,
        error_by_horizon_df=error_by_horizon_df,
        output_dir=output_dir,
        max_horizon_plot=max_horizon_plot,
        min_horizon_count_ratio=min_horizon_count_ratio,
    )

    best_error = float(error_summary_df.loc[0, "best_euclidean_error"])
    median_error = float(error_summary_df.loc[0, "median_euclidean_error"])
    mean_error = float(error_summary_df.loc[0, "mean_euclidean_error"])
    worst_error = float(error_summary_df.loc[0, "worst_euclidean_error"])

    print("Additional validation summary:")
    print(f"  Best point error:   {best_error:.5f} yards")
    print(f"  Median point error: {median_error:.5f} yards")
    print(f"  Mean point error:   {mean_error:.5f} yards")
    print(f"  Worst point error:  {worst_error:.5f} yards")

    print_best_worst_player_tables(
        error_by_player_df=error_by_player_df,
        top_n=top_n_player_table,
    )

    return {
        "validation_predictions_with_errors": "validation_predictions_with_errors.csv",
        "error_distribution_summary": "error_distribution_summary.csv",
        "error_by_prediction_horizon": "error_by_prediction_horizon.csv",
        "error_distribution_plot": "error_distribution.png",
        "error_by_prediction_horizon_plot": "error_by_prediction_horizon.png",
    }


def train_and_evaluate(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    val_weeks: Sequence[int],
    random_state: int,
) -> Tuple[XGBRegressor, XGBRegressor, dict, pd.DataFrame, pd.DataFrame]:
    tr = train_df[~train_df["source_week"].isin(val_weeks)].reset_index(drop=True)
    va = train_df[train_df["source_week"].isin(val_weeks)].reset_index(drop=True)

    if len(tr) == 0 or len(va) == 0:
        raise ValueError(
            "Train/validation split is empty. Check --weeks and --val-weeks settings."
        )

    x_tr = tr[list(feature_cols)]
    x_va = va[list(feature_cols)]
    y_dx_tr = tr["target_dx"]
    y_dy_tr = tr["target_dy"]

    model_dx = make_xgb_model(random_state=random_state)
    model_dy = make_xgb_model(random_state=random_state)

    model_dx.fit(x_tr, y_dx_tr)
    model_dy.fit(x_tr, y_dy_tr)

    pred_dx = model_dx.predict(x_va)
    pred_dy = model_dy.predict(x_va)

    pred_x = va["last_x"].to_numpy() + pred_dx
    pred_y = va["last_y"].to_numpy() + pred_dy

    prediction_cols = [
        "source_week",
        "game_id",
        "play_id",
        "nfl_id",
        "play_key",
        "future_frame_id",
        "horizon_t",
        "horizon_frac",
        "num_frames_output",
        "last_x",
        "last_y",
        "target_x",
        "target_y",
    ]
    prediction_cols.extend([c for c in OPTIONAL_PLAYER_METADATA_COLUMNS if c in va.columns])
    prediction_cols = [c for c in prediction_cols if c in va.columns]

    val_predictions = va[prediction_cols].copy()
    val_predictions["pred_x"] = pred_x
    val_predictions["pred_y"] = pred_y

    metrics, by_week_df = evaluate_validation_predictions(val_predictions)

    metrics.update({
        "train_weeks": sorted(tr["source_week"].unique().tolist()),
        "n_train_rows": int(len(tr)),
        "n_val_rows": int(len(va)),
        "n_features": int(len(feature_cols)),
    })

    return model_dx, model_dy, metrics, val_predictions, by_week_df


def retrain_on_all_data(train_df: pd.DataFrame, feature_cols: Sequence[str], random_state: int):
    x_all = train_df[list(feature_cols)]
    y_dx = train_df["target_dx"]
    y_dy = train_df["target_dy"]

    model_dx = make_xgb_model(random_state=random_state)
    model_dy = make_xgb_model(random_state=random_state)
    model_dx.fit(x_all, y_dx)
    model_dy.fit(x_all, y_dy)
    return model_dx, model_dy


def load_week_pair(data_dir: Path, week: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    week_str = f"{week:02d}"
    input_path = data_dir / f"input_2023_w{week_str}_standardized.csv"
    output_path = data_dir / f"output_2023_w{week_str}_standardized.csv"

    if not input_path.exists() or not output_path.exists():
        raise FileNotFoundError(
            f"Missing standardized files for week {week_str}:\n"
            f"  {input_path}\n"
            f"  {output_path}"
        )

    input_df = pd.read_csv(input_path)
    output_df = pd.read_csv(output_path)
    check_columns(input_df, REQUIRED_INPUT_COLUMNS, f"input week {week_str}")
    check_columns(output_df, REQUIRED_OUTPUT_COLUMNS, f"output week {week_str}")

    input_df["source_week"] = week
    output_df["source_week"] = week
    return input_df, output_df


def load_all_weeks(data_dir: Path, weeks: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inputs = []
    outputs = []
    for week in weeks:
        input_df, output_df = load_week_pair(data_dir=data_dir, week=week)
        inputs.append(input_df)
        outputs.append(output_df)

    return pd.concat(inputs, ignore_index=True), pd.concat(outputs, ignore_index=True)


def save_feature_importance(model: XGBRegressor, feature_cols: Sequence[str], out_path: Path) -> None:
    importance_df = pd.DataFrame(
        {
            "feature": list(feature_cols),
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading standardized data from: {data_dir.resolve()}")
    print(f"Weeks: {args.weeks}")
    print(f"Validation weeks: {args.val_weeks}")
    print(f"Using last_k = {args.last_k}")

    input_df, output_df = load_all_weeks(data_dir=data_dir, weeks=args.weeks)
    print("Combined input shape:", input_df.shape)
    print("Combined output shape:", output_df.shape)

    train_df = build_training_table(input_df=input_df, output_df=output_df, last_k=args.last_k)
    feature_cols = get_feature_columns(train_df)

    print("Training table shape:", train_df.shape)
    print("Number of features:", len(feature_cols))

    eval_dx, eval_dy, metrics, val_predictions, val_by_week_df = train_and_evaluate(
        train_df=train_df,
        feature_cols=feature_cols,
        val_weeks=args.val_weeks,
        random_state=args.random_state,
    )

    print("Validation metrics:")
    print(json.dumps(metrics, indent=2))

    val_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "validation_metrics.csv", index=False)
    val_by_week_df.to_csv(output_dir / "validation_metrics_by_week.csv", index=False)

    additional_eval_files = run_additional_evaluation(
        val_predictions=val_predictions,
        output_dir=output_dir,
        top_n_player_table=args.top_n_player_table,
        max_horizon_plot=args.max_horizon_plot,
        min_horizon_count_ratio=args.min_horizon_count_ratio,
    )

    save_feature_importance(eval_dx, feature_cols, output_dir / "feature_importance_dx.csv")
    save_feature_importance(eval_dy, feature_cols, output_dir / "feature_importance_dy.csv")

    final_dx, final_dy = retrain_on_all_data(
        train_df=train_df,
        feature_cols=feature_cols,
        random_state=args.random_state,
    )

    joblib.dump(final_dx, output_dir / "final_model_dx.joblib")
    joblib.dump(final_dy, output_dir / "final_model_dy.joblib")

    metadata = {
        "weeks": list(args.weeks),
        "val_weeks": list(args.val_weeks),
        "last_k": int(args.last_k),
        "feature_columns": list(feature_cols),
        "metrics": metrics,
        "additional_evaluation_settings": {
            "top_n_player_table": int(args.top_n_player_table),
            "max_horizon_plot": int(args.max_horizon_plot),
            "min_horizon_count_ratio": float(args.min_horizon_count_ratio),
        },
        "additional_evaluation_files": additional_eval_files,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved results to: {output_dir.resolve()}")
    print("Saved files:")
    for name in [
        "validation_predictions.csv",
        "validation_predictions_with_errors.csv",
        "validation_metrics.csv",
        "validation_metrics_by_week.csv",
        "error_distribution_summary.csv",
        "error_by_prediction_horizon.csv",
        "error_distribution.png",
        "error_by_prediction_horizon.png",
        "feature_importance_dx.csv",
        "feature_importance_dy.csv",
        "final_model_dx.joblib",
        "final_model_dy.joblib",
        "metadata.json",
    ]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
