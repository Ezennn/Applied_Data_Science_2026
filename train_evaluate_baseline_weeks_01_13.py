from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import joblib
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
    return [c for c in df.columns if c not in ignore]


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


def train_and_evaluate(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    val_weeks: Sequence[int],
    random_state: int,
) -> Tuple[XGBRegressor, XGBRegressor, dict, pd.DataFrame]:
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

    rmse_x = float(np.sqrt(mean_squared_error(va["target_x"], pred_x)))
    rmse_y = float(np.sqrt(mean_squared_error(va["target_y"], pred_y)))
    dx2 = (pred_x - va["target_x"].to_numpy()) ** 2
    dy2 = (pred_y - va["target_y"].to_numpy()) ** 2
    rmse_distance = float(np.sqrt(np.mean((dx2 + dy2) / 2.0)))

    metrics = {
        "train_weeks": sorted(tr["source_week"].unique().tolist()),
        "val_weeks": sorted(va["source_week"].unique().tolist()),
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_distance": rmse_distance,
        "n_train_rows": int(len(tr)),
        "n_val_rows": int(len(va)),
        "n_features": int(len(feature_cols)),
    }

    val_predictions = va[
        ["source_week", "game_id", "play_id", "nfl_id", "future_frame_id", "target_x", "target_y"]
    ].copy()
    val_predictions["pred_x"] = pred_x
    val_predictions["pred_y"] = pred_y

    return model_dx, model_dy, metrics, val_predictions


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

    eval_dx, eval_dy, metrics, val_predictions = train_and_evaluate(
        train_df=train_df,
        feature_cols=feature_cols,
        val_weeks=args.val_weeks,
        random_state=args.random_state,
    )

    print("Validation metrics:")
    print(json.dumps(metrics, indent=2))

    val_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "validation_metrics.csv", index=False)
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
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved results to: {output_dir.resolve()}")
    print("Saved files:")
    for name in [
        "validation_predictions.csv",
        "validation_metrics.csv",
        "feature_importance_dx.csv",
        "feature_importance_dy.csv",
        "final_model_dx.joblib",
        "final_model_dy.joblib",
        "metadata.json",
    ]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
