
import json
import os
from typing import Dict, Iterable, Optional

import torch

import nfl_trajectory_gnn_pipeline as base
import train_from_preprocessed_weeks_local as weektrain


DEFAULT_TRAIN_WEEKS = list(range(1, 14))
DEFAULT_VAL_WEEKS = list(range(14, 19))

# These are the parameters we intentionally expose for tuning.
TUNABLE_PARAM_NAMES = [
    "d_model",
    "edge_dim",
    "static_dim",
    "dropout",
    "lr",
    "batch_size",
    "weight_decay",
    "num_epochs",
    "patience",
    "horizon_weight_lambda",
    "target_player_weight",
]

# These are kept fixed on purpose.
FIXED_PARAM_NAMES = [
    "loss_weighting",
    "friction_accel_limit",
    "max_speed",
    "max_yaw_rate",
    "track_loss_beta",
    "velocity_loss_beta",
    "horizon_weight_power",
    "non_target_player_weight",
    "phys_loss_weight",
    "attn_loss_weight",
    "phys_warmup_frac",
    "attn_warmup_frac",
    "normalize_main_loss_terms",
    "normalize_regularizer_terms",
    "loss_ema_decay",
    "loss_norm_eps",
    "val_frac",
]

FIXED_PREPROCESS_CONSTANTS = {
    "DT": base.DT,
    "FIELD_X_MIN": base.FIELD_X_MIN,
    "FIELD_X_MAX": base.FIELD_X_MAX,
    "FIELD_Y_MIN": base.FIELD_Y_MIN,
    "FIELD_Y_MAX": base.FIELD_Y_MAX,
    "height_default_inches": 72.0,
    "weight_default_lbs": 210.0,
    "age_default_years": 27.0,
    "savgol_window": 7,
    "savgol_poly": 2,
    "lowpass_cutoff_hz": 2.5,
    "lowpass_order": 2,
    "despike_window_radius": 2,
    "despike_n_sigmas": 3.0,
}


def _sm_str(major_minor) -> str:
    major, minor = major_minor
    return f"sm_{int(major)}{int(minor)}"


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False

    try:
        cap = torch.cuda.get_device_capability(0)
        cap_str = _sm_str(cap)
        arch_list = torch.cuda.get_arch_list()
        if not any(str(arch).startswith(cap_str) for arch in arch_list):
            return False

        # Ensure we can actually launch a trivial kernel on this device.
        _ = torch.rand(1, device="cuda") * 2
        return True
    except Exception:
        return False


def _as_int_list(values: Optional[Iterable]) -> Optional[list]:
    if values is None:
        return None
    return [int(v) for v in values]


def apply_recommended_fixed_settings(config: base.ModelConfig) -> base.ModelConfig:
    """
    Keep the parameters in the 'fix first' bucket fixed.
    Also keep track-vs-velocity weighting learnable.
    """
    defaults = base.ModelConfig()

    # Keep these fixed / literature-or-prior style settings.
    config.loss_weighting = "uncertainty_main"
    config.friction_accel_limit = defaults.friction_accel_limit
    config.max_speed = defaults.max_speed
    config.max_yaw_rate = defaults.max_yaw_rate

    # Keep preprocessing-independent loss shape choices fixed first.
    config.track_loss_beta = defaults.track_loss_beta
    config.velocity_loss_beta = defaults.velocity_loss_beta
    config.horizon_weight_power = defaults.horizon_weight_power
    config.non_target_player_weight = defaults.non_target_player_weight

    # Keep regularizers fixed first.
    config.phys_loss_weight = defaults.phys_loss_weight
    config.attn_loss_weight = defaults.attn_loss_weight
    config.phys_warmup_frac = defaults.phys_warmup_frac
    config.attn_warmup_frac = defaults.attn_warmup_frac

    # Make explicit week split the source of truth.
    config.val_frac = 0.0

    # Preserve the current learned-loss behavior.
    config.normalize_main_loss_terms = defaults.normalize_main_loss_terms
    config.normalize_regularizer_terms = defaults.normalize_regularizer_terms
    config.loss_ema_decay = defaults.loss_ema_decay
    config.loss_norm_eps = defaults.loss_norm_eps
    return config


def make_recommended_tuning_config(
    device: Optional[str] = None,
    d_model: int = 32,
    edge_dim: int = 8,
    static_dim: int = 16,
    dropout: float = 0.10,
    lr: float = 2e-4,
    batch_size: int = 4,
    weight_decay: float = 1e-5,
    num_epochs: int = 8,
    patience: int = 3,
    horizon_weight_lambda: float = 1.5,
    target_player_weight: float = 2.5,
    num_workers: int = 2,
) -> base.ModelConfig:
    """
    Recommended split of responsibilities:
    - learn: all network weights + relative track/vel weighting
    - tune: capacity / optimization / task emphasis parameters
    - fix: physical bounds + preprocessing constants + data defaults
    """
    config = base.ModelConfig(
        d_model=d_model,
        edge_dim=edge_dim,
        static_dim=static_dim,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        patience=patience,
        horizon_weight_lambda=horizon_weight_lambda,
        target_player_weight=target_player_weight,
        num_workers=num_workers,
        min_epochs=max(2, min(4, num_epochs // 2 if num_epochs > 2 else 2)),
    )
    requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(requested).startswith("cuda") and not _cuda_usable():
        try:
            cap_str = _sm_str(torch.cuda.get_device_capability(0))
            arch_list = torch.cuda.get_arch_list()
            gpu_name = torch.cuda.get_device_name(0)
            print(
                f"[WARN] CUDA requested but not usable for {gpu_name} ({cap_str}). "
                f"PyTorch wheel supports: {arch_list}. Falling back to CPU."
            )
        except Exception:
            print("[WARN] CUDA requested but not usable. Falling back to CPU.")
        requested = "cpu"

    config.device = requested
    return apply_recommended_fixed_settings(config)


def _collect_experiment_settings(
    config: base.ModelConfig,
    train_weeks: Optional[Iterable],
    val_weeks: Optional[Iterable],
) -> Dict[str, object]:
    config_dict = vars(config).copy()
    tuned = {name: config_dict.get(name) for name in TUNABLE_PARAM_NAMES}
    fixed = {name: config_dict.get(name) for name in FIXED_PARAM_NAMES}
    return {
        "train_weeks": _as_int_list(train_weeks),
        "val_weeks": _as_int_list(val_weeks),
        "learned_by_training": [
            "all_main_network_weights",
            "relative_weight_between_track_loss_and_vel_loss_via_uncertainty_main",
        ],
        "tuned_manually": tuned,
        "fixed_in_code": fixed,
        "fixed_preprocessing_constants": FIXED_PREPROCESS_CONSTANTS,
    }


def save_experiment_settings(
    artifact_dir: str,
    config: base.ModelConfig,
    train_weeks: Optional[Iterable],
    val_weeks: Optional[Iterable],
) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    payload = _collect_experiment_settings(config, train_weeks=train_weeks, val_weeks=val_weeks)
    path = os.path.join(artifact_dir, "recommended_experiment_settings.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def fit_recommended_1_13_eval_14_18(
    preprocessed_dir: str,
    artifact_dir: str,
    holdout_output_dir: Optional[str] = None,
    config: Optional[base.ModelConfig] = None,
    train_weeks: Optional[Iterable] = None,
    val_weeks: Optional[Iterable] = None,
    eval_every: int = 1,
    resume: bool = True,
):
    config = config or make_recommended_tuning_config()
    config = apply_recommended_fixed_settings(config)

    train_weeks = list(DEFAULT_TRAIN_WEEKS if train_weeks is None else train_weeks)
    val_weeks = list(DEFAULT_VAL_WEEKS if val_weeks is None else val_weeks)

    settings_path = save_experiment_settings(
        artifact_dir=artifact_dir,
        config=config,
        train_weeks=train_weeks,
        val_weeks=val_weeks,
    )

    history, best_model_path = weektrain.fit_from_preprocessed_dir(
        preprocessed_dir=preprocessed_dir,
        artifact_dir=artifact_dir,
        config=config,
        eval_every=eval_every,
        resume=resume,
        train_weeks=train_weeks,
        val_weeks=val_weeks,
    )

    summary = None
    if holdout_output_dir is not None:
        summary = weektrain.evaluate_holdout_weeks(
            preprocessed_dir=preprocessed_dir,
            artifact_dir=artifact_dir,
            eval_weeks=val_weeks,
            output_dir=holdout_output_dir,
        )

    return {
        "history": history,
        "best_model_path": best_model_path,
        "settings_path": settings_path,
        "holdout_summary": summary,
    }


def _parse_cli_weeks(text: Optional[str]):
    if text is None or str(text).strip() == "":
        return None
    out = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Recommended NFL GNN trainer: learn main weights + learned track/vel weighting, "
                    "tune only the key model/optimization parameters, keep physical and preprocessing priors fixed."
    )
    parser.add_argument("--preprocessed_dir", required=True)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--holdout_output_dir", default=None)

    parser.add_argument("--train_weeks", default="1,2,3,4,5,6,7,8,9,10,11,12,13")
    parser.add_argument("--val_weeks", default="14,15,16,17,18")

    # Tunable parameters only.
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--edge_dim", type=int, default=8)
    parser.add_argument("--static_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--horizon_weight_lambda", type=float, default=1.5)
    parser.add_argument("--target_player_weight", type=float, default=2.5)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = make_recommended_tuning_config(
        d_model=args.d_model,
        edge_dim=args.edge_dim,
        static_dim=args.static_dim,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        patience=args.patience,
        horizon_weight_lambda=args.horizon_weight_lambda,
        target_player_weight=args.target_player_weight,
        num_workers=args.num_workers,
    )

    result = fit_recommended_1_13_eval_14_18(
        preprocessed_dir=args.preprocessed_dir,
        artifact_dir=args.artifact_dir,
        holdout_output_dir=args.holdout_output_dir,
        config=config,
        train_weeks=_parse_cli_weeks(args.train_weeks),
        val_weeks=_parse_cli_weeks(args.val_weeks),
        eval_every=args.eval_every,
        resume=args.resume,
    )
    print(json.dumps({
        "best_model_path": result["best_model_path"],
        "settings_path": result["settings_path"],
        "has_holdout_summary": result["holdout_summary"] is not None,
    }, ensure_ascii=False, indent=2))
