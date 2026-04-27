"""
Option C - Step 2: train recommended local model.

This trains on weeks 1-13 and evaluates on weeks 14-18.
Run preprocess_weekly_tracking_csvs_local.py first.
"""
import json
import os
import sys
import torch


def _compact_metrics_row(row: dict) -> dict:
    if not isinstance(row, dict):
        return {"_raw": str(row)}

    def _pick(name: str):
        v = row.get(name, None)
        if isinstance(v, (int, str)) or v is None:
            return v
        try:
            return float(v)
        except Exception:
            return str(v)

    out = {
        "epoch": _pick("epoch"),
        "val_monitor_name": _pick("val_monitor_name"),
        "val_monitor_score": _pick("val_monitor_score"),
        "train_loss": _pick("train_loss"),
        "val_loss": _pick("val_loss"),
        "val_kaggle_rmse": _pick("val_kaggle_rmse"),
        "train_track_loss_raw": _pick("train_track_loss_raw"),
        "train_vel_loss_raw": _pick("train_vel_loss_raw"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _write_training_run_log(artifact_dir: str, summary: dict) -> str:
    path = os.path.join(artifact_dir, "training_run_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    return path


def _extract_holdout_rmse(holdout_summary: dict):
    if not isinstance(holdout_summary, dict):
        return None
    for key in ("overall_kaggle_rmse", "kaggle_rmse", "rmse"):
        value = holdout_summary.get(key)
        if value is not None:
            return value
    return None


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    preproc_dir = os.path.join(project_dir, "preprocessed_csv_weekly")
    artifact_dir = os.path.join(project_dir, "artifacts_recommended_1_13_14_18")
    holdout_dir = os.path.join(project_dir, "holdout_predictions_recommended_14_18")

    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(holdout_dir, exist_ok=True)

    if project_dir not in sys.path:
        sys.path.append(project_dir)

    from train_recommended_holdout_split_local import (
        DEFAULT_TRAIN_WEEKS,
        DEFAULT_VAL_WEEKS,
        fit_recommended_1_13_eval_14_18,
        make_recommended_tuning_config,
    )

    config = make_recommended_tuning_config(
        device="cuda" if torch.cuda.is_available() else "cpu",
        d_model=32,
        edge_dim=8,
        static_dim=16,
        dropout=0.10,
        lr=2e-4,
        batch_size=2,
        weight_decay=1e-5,
        num_epochs=80,
        patience=12,
        horizon_weight_lambda=1.5,
        target_player_weight=2.5,
        num_workers=2,
    )
    config.min_epochs = 20

    print("========== Option C Step 2: Training ==========")
    print(f"Using device: {config.device}")
    print(f"Epochs: {config.num_epochs} (min_epochs={config.min_epochs}, patience={config.patience})")
    print(f"Batch size: {config.batch_size} | lr: {config.lr} | dropout: {config.dropout} | weight_decay: {config.weight_decay}")
    print(f"Training weeks: {DEFAULT_TRAIN_WEEKS}")
    print(f"Validation/holdout weeks: {DEFAULT_VAL_WEEKS}")
    print(f"Artifacts dir: {artifact_dir}")
    print(f"Holdout dir: {holdout_dir}")

    result = fit_recommended_1_13_eval_14_18(
        preprocessed_dir=preproc_dir,
        artifact_dir=artifact_dir,
        holdout_output_dir=holdout_dir,
        config=config,
        train_weeks=DEFAULT_TRAIN_WEEKS,
        val_weeks=DEFAULT_VAL_WEEKS,
        eval_every=2,
        resume=False,
    )

    history = result.get("history") or []
    last_row = history[-1] if isinstance(history, list) and len(history) > 0 else None

    summary = {
        "best_model_path": result.get("best_model_path"),
        "settings_path": result.get("settings_path"),
        "history_rows": len(history) if isinstance(history, list) else None,
        "history_last": _compact_metrics_row(last_row) if last_row is not None else None,
        "has_holdout_summary": result.get("holdout_summary") is not None,
        "holdout_summary": result.get("holdout_summary"),
        "artifacts_dir": artifact_dir,
        "holdout_dir": holdout_dir,
        "history_csv": os.path.join(artifact_dir, "training_history.csv"),
    }
    log_path = _write_training_run_log(artifact_dir=artifact_dir, summary=summary)
    holdout_rmse = _extract_holdout_rmse(result.get("holdout_summary"))

    print("\n========== Finished ==========")
    print(f"Best model: {summary['best_model_path']}")
    print(f"History CSV: {summary['history_csv']}")
    if holdout_rmse is not None:
        print(f"Holdout RMSE: {holdout_rmse:.6f}")
    print(f"Detailed log: {log_path}")


if __name__ == "__main__":
    # Required on Windows when DataLoader uses num_workers > 0.
    main()
