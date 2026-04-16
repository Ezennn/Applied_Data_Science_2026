from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained baseline model on standardized NFL Big Data Bowl files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="train_preprocessed",
        help="Directory containing standardized input/output CSV files.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="baseline_output",
        help="Directory containing final_model_dx.joblib, final_model_dy.joblib, metadata.json.",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="+",
        default=[14, 15, 16, 17, 18],
        help="Weeks to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_eval_14_18",
        help="Directory to save evaluation outputs.",
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


def _parse_height_to_inches(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    if "-" in s:
        parts = s.split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            feet = int(parts[0])
            inches = int(parts[1])
            return float(feet * 12 + inches)
    try:
        return float(s)
    except ValueError:
        return np.nan


def _birth_date_to_age_years(value) -> float:
    if pd.isna(value):
        return np.nan
    try:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return np.nan
        # fixed reference date keeps this deterministic
        ref = pd.Timestamp("2023-09-01")
        return float((ref - dt).days / 365.25)
    except Exception:
        return np.nan


def _pad_to_last_k(group: pd.DataFrame, last_k: int) -> pd.DataFrame:
    group = group.sort_values("frame_id").reset_index(drop=True)
    if len(group) >= last_k:
        return group.iloc[-last_k:].reset_index(drop=True)
    first_row = group.iloc[[0]].copy()
    pad_count = last_k - len(group)
    pad = pd.concat([first_row] * pad_count, ignore_index=True)
    return pd.concat([pad, group], ignore_index=True).reset_index(drop=True)


def _history_features(player_hist: pd.DataFrame, last_k: int):
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
        feat["player_weight"] = pd.to_numeric(pd.Series([last_row["player_weight"]]), errors="coerce").iloc[0]
    if "player_height" in hist.columns:
        feat["player_height"] = _parse_height_to_inches(last_row["player_height"])
    if "player_birth_date" in hist.columns:
        feat["player_age_years"] = _birth_date_to_age_years(last_row["player_birth_date"])

    # Optional categorical codes only if present in source data. If not used by model metadata,
    # they will be ignored later.
    for col in ["player_position", "player_side", "player_role"]:
        if col in hist.columns:
            feat[f"{col}_code"] = float(pd.Categorical([last_row[col]]).codes[0])

    for i, row in hist.iterrows():
        prefix = f"hist_{i+1}"
        feat[f"{prefix}_x_rel"] = float(row["x"] - last_row["x"])
        feat[f"{prefix}_y_rel"] = float(row["y"] - last_row["y"])
        feat[f"{prefix}_s"] = float(row["s"])
        feat[f"{prefix}_a"] = float(row["a"])
        feat[f"{prefix}_dir_sin"] = float(row["dir_sin"])
        feat[f"{prefix}_dir_cos"] = float(row["dir_cos"])
        feat[f"{prefix}_o_sin"] = float(row["o_sin"])
        feat[f"{prefix}_o_cos"] = float(row["o_cos"])

    return feat


def build_prediction_table(input_df: pd.DataFrame, last_k: int = 5) -> pd.DataFrame:
    input_df = add_angle_features(input_df)
    input_df = input_df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"]).copy()
    target_input = input_df[input_df["player_to_predict"] == 1].copy()

    rows = []
    grouped_target_input = target_input.groupby(["game_id", "play_id", "nfl_id"], sort=False)

    for key, player_hist in grouped_target_input:
        feat = _history_features(player_hist, last_k=last_k)
        game_id, play_id, nfl_id = key
        num_future = int(player_hist["num_frames_output"].iloc[-1])
        week = int(player_hist["week"].iloc[-1]) if "week" in player_hist.columns else -1

        for t in range(1, num_future + 1):
            row = {
                "week": week,
                "game_id": game_id,
                "play_id": play_id,
                "nfl_id": nfl_id,
                "future_frame_id": t,
                "horizon_t": t,
                "horizon_frac": float(t) / max(float(feat["num_frames_output"]), 1.0),
            }
            row.update(feat)
            rows.append(row)

    if not rows:
        raise ValueError("No prediction rows were built. Check player_to_predict in input data.")

    return pd.DataFrame(rows)


def load_week_files(data_dir: Path, weeks: List[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_frames = []
    output_frames = []

    for week in weeks:
        input_path = data_dir / f"input_2023_w{week:02d}_standardized.csv"
        output_path = data_dir / f"output_2023_w{week:02d}_standardized.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file: {input_path}")
        if not output_path.exists():
            raise FileNotFoundError(f"Missing output file: {output_path}")

        input_df = pd.read_csv(input_path)
        output_df = pd.read_csv(output_path)

        check_columns(input_df, REQUIRED_INPUT_COLUMNS, f"input week {week}")
        check_columns(output_df, REQUIRED_OUTPUT_COLUMNS, f"output week {week}")

        input_df["week"] = week
        output_df["week"] = week

        input_frames.append(input_df)
        output_frames.append(output_df)

    combined_input = pd.concat(input_frames, ignore_index=True)
    combined_output = pd.concat(output_frames, ignore_index=True)
    return combined_input, combined_output


def load_artifacts(model_dir: Path):
    model_dx_path = model_dir / "final_model_dx.joblib"
    model_dy_path = model_dir / "final_model_dy.joblib"
    metadata_path = model_dir / "metadata.json"

    if not model_dx_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_dx_path}")
    if not model_dy_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_dy_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    model_dx = joblib.load(model_dx_path)
    model_dy = joblib.load(model_dy_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_columns"]
    last_k = int(metadata["last_k"])
    return model_dx, model_dy, feature_cols, last_k, metadata


def make_predictions(input_df: pd.DataFrame, model_dx, model_dy, feature_cols: List[str], last_k: int) -> pd.DataFrame:
    pred_table = build_prediction_table(input_df=input_df, last_k=last_k)

    for col in feature_cols:
        if col not in pred_table.columns:
            pred_table[col] = 0.0

    x_pred = pred_table[feature_cols].copy()
    for col in feature_cols:
        x_pred[col] = pd.to_numeric(x_pred[col], errors="coerce")
    x_pred = x_pred.fillna(0.0)

    pred_dx = model_dx.predict(x_pred)
    pred_dy = model_dy.predict(x_pred)

    pred_table["pred_x"] = pred_table["last_x"] + pred_dx
    pred_table["pred_y"] = pred_table["last_y"] + pred_dy

    return pred_table[["week", "game_id", "play_id", "nfl_id", "future_frame_id", "pred_x", "pred_y"]].rename(
        columns={"future_frame_id": "frame_id"}
    )


def compute_metrics(merged: pd.DataFrame) -> dict:
    dx2 = (merged["pred_x"] - merged["true_x"]) ** 2
    dy2 = (merged["pred_y"] - merged["true_y"]) ** 2

    rmse_x = float(np.sqrt(np.mean(dx2)))
    rmse_y = float(np.sqrt(np.mean(dy2)))
    kaggle_score = float(np.sqrt(np.mean((dx2 + dy2) / 2.0)))

    return {
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "kaggle_score": kaggle_score,
        "n_rows": int(len(merged)),
    }


def evaluate(predictions: pd.DataFrame, output_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    truth = output_df[["week", "game_id", "play_id", "nfl_id", "frame_id", "x", "y"]].copy()
    truth = truth.rename(columns={"x": "true_x", "y": "true_y"})

    merged = truth.merge(
        predictions,
        on=["week", "game_id", "play_id", "nfl_id", "frame_id"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(truth):
        missing = len(truth) - len(merged)
        raise ValueError(f"Prediction/truth merge lost {missing} rows. Check keys and preprocessing.")

    overall = compute_metrics(merged)
    overall["weeks"] = sorted(merged["week"].unique().tolist())

    by_week_rows = []
    for week, group in merged.groupby("week", sort=True):
        row = {"week": int(week)}
        row.update(compute_metrics(group))
        by_week_rows.append(row)

    by_week_df = pd.DataFrame(by_week_rows)
    return merged, by_week_df, overall


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading standardized data from: {data_dir.resolve()}")
    print(f"Evaluating weeks: {args.weeks}")
    print(f"Loading model artifacts from: {model_dir.resolve()}")

    input_df, output_df = load_week_files(data_dir=data_dir, weeks=args.weeks)
    print("Combined input shape:", input_df.shape)
    print("Combined output shape:", output_df.shape)

    model_dx, model_dy, feature_cols, last_k, metadata = load_artifacts(model_dir=model_dir)
    print(f"Using last_k = {last_k}")
    print(f"Number of features expected by model: {len(feature_cols)}")

    predictions = make_predictions(
        input_df=input_df,
        model_dx=model_dx,
        model_dy=model_dy,
        feature_cols=feature_cols,
        last_k=last_k,
    )
    print("Prediction table shape:", predictions.shape)

    merged, by_week_df, overall = evaluate(predictions=predictions, output_df=output_df)

    merged.to_csv(output_dir / "evaluation_predictions.csv", index=False)
    pd.DataFrame([overall]).to_csv(output_dir / "evaluation_metrics_overall.csv", index=False)
    by_week_df.to_csv(output_dir / "evaluation_metrics_by_week.csv", index=False)

    print("Overall metrics:")
    print(json.dumps(overall, indent=2))
    print("Saved results to:", output_dir.resolve())
    print("Saved files:")
    print("  - evaluation_predictions.csv")
    print("  - evaluation_metrics_overall.csv")
    print("  - evaluation_metrics_by_week.csv")


if __name__ == "__main__":
    main()
