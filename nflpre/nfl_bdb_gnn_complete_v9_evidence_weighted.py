
import os
import math
import glob
import json
import pickle
import random
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter, butter, sosfiltfilt
except Exception:  # pragma: no cover
    savgol_filter = None
    butter = None
    sosfiltfilt = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


FIELD_X_MIN = 0.0
FIELD_X_MAX = 120.0
FIELD_Y_MIN = 0.0
FIELD_Y_MAX = 53.3
DT = 0.1
EPS = 1e-6

INPUT_REQUIRED_COLS = [
    "game_id", "play_id", "player_to_predict", "nfl_id", "frame_id",
    "play_direction", "absolute_yardline_number", "player_name",
    "player_height", "player_weight", "player_birth_date",
    "player_position", "player_side", "player_role",
    "x", "y", "s", "a", "dir", "o",
    "num_frames_output", "ball_land_x", "ball_land_y",
]

OUTPUT_REQUIRED_COLS = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y"]
TEMPLATE_REQUIRED_COLS = ["id", "game_id", "play_id", "nfl_id", "frame_id"]

OBS_FEATURE_NAMES = [
    "x", "y",
    "vx_own", "vy_own",
    "ax_own", "ay_own",
    "a_tan_own", "a_lat_own",
    "speed_own",
    "s_official",
    "a_official",
    "a_residual",
    "sin_dir", "cos_dir",
    "sin_o", "cos_o",
    "sin_gap", "cos_gap",
    "turn_rate_own",
    "ball_land_x", "ball_land_y",
    "absolute_yardline_number",
    "num_frames_output",
    "obs_mask",
]
OBS_FEATURE_INDEX = {name: i for i, name in enumerate(OBS_FEATURE_NAMES)}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_height_to_inches(h):
    if pd.isna(h):
        return 72.0
    if isinstance(h, (int, float)):
        return float(h)
    s = str(h).strip()
    if "-" in s:
        try:
            ft, inch = s.split("-")
            return float(ft) * 12.0 + float(inch)
        except Exception:
            return 72.0
    try:
        return float(s)
    except Exception:
        return 72.0


def coerce_bool_to_float(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(np.float32)
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0).astype(float) > 0).astype(np.float32)
    vals = series.astype(str).str.lower().str.strip()
    return vals.isin({"1", "true", "t", "yes", "y"}).astype(np.float32)


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi


def canonicalize_play_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "play_direction" not in df.columns:
        return df
    left_mask = df["play_direction"].astype(str).str.lower().eq("left")
    if "x" in df.columns:
        df.loc[left_mask, "x"] = FIELD_X_MAX - df.loc[left_mask, "x"]
    if "y" in df.columns:
        df.loc[left_mask, "y"] = FIELD_Y_MAX - df.loc[left_mask, "y"]
    for ang_col in ["dir", "o"]:
        if ang_col in df.columns:
            df.loc[left_mask, ang_col] = (df.loc[left_mask, ang_col] + 180.0) % 360.0
    for c in ["ball_land_x", "ball_land_y"]:
        if c in df.columns:
            if c == "ball_land_x":
                df.loc[left_mask, c] = FIELD_X_MAX - df.loc[left_mask, c]
            else:
                df.loc[left_mask, c] = FIELD_Y_MAX - df.loc[left_mask, c]
    df["play_direction_canonical"] = "right"
    return df


def _safe_savgol(values: np.ndarray, window: int = 7, poly: int = 2, deriv: int = 0, delta: float = DT) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    n = len(values)
    if savgol_filter is None or n < 5:
        if deriv == 0:
            return values
        return np.gradient(values, delta).astype(np.float32) if deriv == 1 else np.gradient(np.gradient(values, delta), delta).astype(np.float32)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 5:
        if deriv == 0:
            return values
        return np.gradient(values, delta).astype(np.float32) if deriv == 1 else np.gradient(np.gradient(values, delta), delta).astype(np.float32)
    p = min(poly, w - 1)
    try:
        return savgol_filter(values, window_length=w, polyorder=p, deriv=deriv, delta=delta, mode="interp").astype(np.float32)
    except Exception:
        if deriv == 0:
            return values
        return np.gradient(values, delta).astype(np.float32) if deriv == 1 else np.gradient(np.gradient(values, delta), delta).astype(np.float32)


def _hampel_like_despike(values: np.ndarray, window_radius: int = 2, n_sigmas: float = 3.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    n = len(values)
    if n < 5:
        return values
    out = values.copy()
    k = max(1, int(window_radius))
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        segment = values[lo:hi]
        med = np.median(segment)
        mad = np.median(np.abs(segment - med))
        sigma = 1.4826 * mad + 1e-6
        if abs(values[i] - med) > n_sigmas * sigma:
            out[i] = med
    return out.astype(np.float32)


def _safe_zero_phase_lowpass(values: np.ndarray, cutoff_hz: float = 2.5, order: int = 2, fs: float = 1.0 / DT) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    n = len(values)
    if butter is None or sosfiltfilt is None or n < 9:
        return values
    try:
        # zero-phase Butterworth low-pass, common in engineering motion processing
        sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
        padlen = min(n - 1, max(0, 3 * (2 * len(sos) + 1)))
        if n <= padlen + 1:
            return values
        return sosfiltfilt(sos, values, padtype="odd", padlen=padlen).astype(np.float32)
    except Exception:
        return values


def _estimate_kinematics_from_xy(
    x: np.ndarray,
    y: np.ndarray,
    smooth_window: int = 7,
    smooth_poly: int = 2,
    cutoff_hz: float = 2.5,
) -> Dict[str, np.ndarray]:
    # 1) despike raw tracking glitches
    x0 = _hampel_like_despike(x, window_radius=2, n_sigmas=3.0)
    y0 = _hampel_like_despike(y, window_radius=2, n_sigmas=3.0)

    # 2) zero-phase low-pass denoise to avoid lag
    x1 = _safe_zero_phase_lowpass(x0, cutoff_hz=cutoff_hz, order=2, fs=1.0 / DT)
    y1 = _safe_zero_phase_lowpass(y0, cutoff_hz=cutoff_hz, order=2, fs=1.0 / DT)

    # 3) Savitzky-Golay local polynomial fit; use direct derivatives instead of repeated np.gradient
    xs = _safe_savgol(x1, smooth_window, smooth_poly, deriv=0, delta=DT)
    ys = _safe_savgol(y1, smooth_window, smooth_poly, deriv=0, delta=DT)
    vx = _safe_savgol(x1, smooth_window, smooth_poly, deriv=1, delta=DT)
    vy = _safe_savgol(y1, smooth_window, smooth_poly, deriv=1, delta=DT)
    ax = _safe_savgol(x1, smooth_window, smooth_poly, deriv=2, delta=DT)
    ay = _safe_savgol(y1, smooth_window, smooth_poly, deriv=2, delta=DT)

    return {
        "x_smooth": xs.astype(np.float32),
        "y_smooth": ys.astype(np.float32),
        "vx_own": vx.astype(np.float32),
        "vy_own": vy.astype(np.float32),
        "ax_own": ax.astype(np.float32),
        "ay_own": ay.astype(np.float32),
    }


def add_derived_observed_features(df: pd.DataFrame, smooth_window: int = 7, smooth_poly: int = 2, cutoff_hz: float = 2.5) -> pd.DataFrame:
    df = df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"]).copy()

    derived_parts = []
    group_cols = ["game_id", "play_id", "nfl_id"]

    for _, g in df.groupby(group_cols, sort=False):
        g = g.copy()
        g = g.sort_values("frame_id").reset_index(drop=True)

        x = g["x"].to_numpy(dtype=np.float32)
        y = g["y"].to_numpy(dtype=np.float32)
        dir_deg = g["dir"].to_numpy(dtype=np.float32)
        o_deg = g["o"].to_numpy(dtype=np.float32)

        kin = _estimate_kinematics_from_xy(
            x=x,
            y=y,
            smooth_window=smooth_window,
            smooth_poly=smooth_poly,
            cutoff_hz=cutoff_hz,
        )
        xs = kin["x_smooth"]
        ys = kin["y_smooth"]
        vx = kin["vx_own"]
        vy = kin["vy_own"]
        ax = kin["ax_own"]
        ay = kin["ay_own"]

        speed_own = np.sqrt(vx ** 2 + vy ** 2).astype(np.float32)
        move_ang = np.arctan2(vy, vx + 1e-6).astype(np.float32)

        a_tan = (ax * np.cos(move_ang) + ay * np.sin(move_ang)).astype(np.float32)
        a_lat = (-ax * np.sin(move_ang) + ay * np.cos(move_ang)).astype(np.float32)
        a_mag = np.sqrt(ax ** 2 + ay ** 2).astype(np.float32)

        dir_rad = np.unwrap(np.deg2rad(dir_deg)).astype(np.float32)
        turn_rate = _safe_savgol(dir_rad, smooth_window, smooth_poly, deriv=1, delta=DT)

        gap_deg = wrap_deg(dir_deg - o_deg).astype(np.float32)

        g["x_smooth"] = xs
        g["y_smooth"] = ys
        g["vx_own"] = vx
        g["vy_own"] = vy
        g["ax_own"] = ax
        g["ay_own"] = ay
        g["speed_own"] = speed_own
        g["a_mag_own"] = a_mag
        g["a_tan_own"] = a_tan
        g["a_lat_own"] = a_lat
        g["turn_rate_own"] = turn_rate.astype(np.float32)
        g["body_motion_gap_deg"] = gap_deg

        derived_parts.append(g)

    df = pd.concat(derived_parts, ignore_index=True)

    df["height_inches"] = df["player_height"].map(parse_height_to_inches).astype(np.float32)
    if "player_weight" in df.columns:
        med_weight = pd.to_numeric(df["player_weight"], errors="coerce").median()
        df["player_weight"] = pd.to_numeric(df["player_weight"], errors="coerce").fillna(med_weight if pd.notna(med_weight) else 210.0).astype(np.float32)
    else:
        df["player_weight"] = np.float32(210.0)

    birth = pd.to_datetime(df.get("player_birth_date"), errors="coerce")
    ref_date = pd.Timestamp("2025-09-01")
    age_years = ((ref_date - birth).dt.days / 365.25).astype("float32")
    df["age_years"] = age_years.fillna(27.0).astype(np.float32)

    if "player_to_predict" in df.columns:
        df["is_target"] = coerce_bool_to_float(df["player_to_predict"]).astype(np.float32)
    else:
        df["is_target"] = np.float32(0.0)

    df["s_official"] = pd.to_numeric(df["s"], errors="coerce").fillna(0.0).astype(np.float32)
    df["a_official"] = pd.to_numeric(df["a"], errors="coerce").fillna(0.0).astype(np.float32)
    df["a_residual"] = (df["a_mag_own"] - df["a_official"]).astype(np.float32)

    return df


def _recursive_glob(root: str, pattern: str) -> List[str]:
    return glob.glob(os.path.join(root, pattern), recursive=True)


def _normalize_search_roots(search_root) -> List[str]:
    if isinstance(search_root, (list, tuple, set)):
        roots = [os.path.abspath(str(r)) for r in search_root if r]
    else:
        roots = [os.path.abspath(str(search_root))]

    extra = [
        os.getcwd(),
        "/content",
        "/content/nfl_data",
        "/content/kaggle_download",
        "/content/drive/MyDrive",
        "/kaggle/input",
    ]
    roots.extend(extra)

    seen = set()
    out = []
    for r in roots:
        if r and os.path.exists(r) and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _candidate_data_dirs_from_csvs(search_root: str) -> List[str]:
    search_root = os.path.abspath(search_root)
    candidate_patterns = [
        "**/train_input*.csv",
        "**/input_*.csv",
    ]
    candidates = []
    for pat in candidate_patterns:
        for p in _recursive_glob(search_root, pat):
            if not os.path.isfile(p):
                continue
            parent = os.path.dirname(p)
            if os.path.basename(parent).lower() == "train":
                candidates.append(os.path.dirname(parent))
            else:
                candidates.append(parent)
    return sorted(set(candidates))


def _candidate_zip_paths(search_root) -> List[str]:
    roots = _normalize_search_roots(search_root)
    pats = [
        "**/nfl-big-data-bowl*.zip",
        "**/*big-data-bowl*.zip",
        "**/*.zip",
    ]
    zips = []
    for root in roots:
        for pat in pats:
            zips.extend(_recursive_glob(root, pat))
    zips = [p for p in sorted(set(zips)) if os.path.isfile(p)]
    zips.sort(key=lambda p: ("nfl-big-data-bowl" not in os.path.basename(p).lower(), len(p)))
    return zips


def _extract_zip_if_needed(zip_path: str, extract_root: str) -> str:
    zip_path = os.path.abspath(zip_path)
    extract_root = os.path.abspath(extract_root)
    os.makedirs(extract_root, exist_ok=True)
    stem = os.path.splitext(os.path.basename(zip_path))[0]
    target_dir = os.path.join(extract_root, stem)

    if os.path.isdir(target_dir):
        found = _candidate_data_dirs_from_csvs(target_dir)
        if found:
            return found[0]

    found_existing = _candidate_data_dirs_from_csvs(extract_root)
    if found_existing:
        return found_existing[0]

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)

    found = _candidate_data_dirs_from_csvs(target_dir)
    if found:
        return found[0]

    found = _candidate_data_dirs_from_csvs(extract_root)
    if found:
        return found[0]

    raise FileNotFoundError(
        f"Unzipped {zip_path} into {target_dir}, but still could not find input_*.csv/train_input*.csv."
    )


def running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def ensure_kaggle_json_colab(upload_if_missing: bool = True, target_dir: str = "/root/.kaggle") -> Optional[str]:
    candidates = [
        os.path.expanduser("~/.kaggle/kaggle.json"),
        "/content/kaggle.json",
        "/root/.kaggle/kaggle.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    if not upload_if_missing or not running_in_colab():
        return None

    try:
        from google.colab import files  # type: ignore
    except Exception:
        return None

    print("kaggle.json not found. Please upload kaggle.json now.")
    uploaded = files.upload()
    if "kaggle.json" not in uploaded:
        print("No kaggle.json uploaded.")
        return None

    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "kaggle.json")
    with open(target_path, "wb") as f:
        f.write(uploaded["kaggle.json"])
    os.chmod(target_path, 0o600)

    # keep a copy under /content for visibility
    try:
        shutil.copy2(target_path, "/content/kaggle.json")
    except Exception:
        pass

    return target_path


def _try_kaggle_competition_download(download_dir: str, competition_slug: str = "nfl-big-data-bowl-2026-prediction") -> Optional[str]:
    download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    kaggle_json_candidates = [
        os.path.expanduser("~/.kaggle/kaggle.json"),
        "/content/kaggle.json",
        "/root/.kaggle/kaggle.json",
    ]
    has_creds = any(os.path.exists(p) for p in kaggle_json_candidates)
    if not has_creds:
        ensure_kaggle_json_colab(upload_if_missing=True)
        has_creds = any(os.path.exists(p) for p in kaggle_json_candidates)
    if not has_creds:
        return None

    cmd = ["kaggle", "competitions", "download", "-c", competition_slug, "-p", download_dir]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return None

    zips = _candidate_zip_paths(download_dir)
    return zips[0] if zips else None


def auto_find_data_dir(
    search_root: str = "/content",
    extract_root: Optional[str] = None,
    auto_unzip: bool = True,
    try_kaggle_download: bool = True,
    competition_slug: str = "nfl-big-data-bowl-2026-prediction",
) -> str:
    """Find or prepare the competition data directory.

    Search order:
    1) already-extracted CSVs under known roots
    2) zip files under known roots, then auto-unzip
    3) Kaggle API download (if kaggle.json is available), then auto-unzip
    """
    roots = _normalize_search_roots(search_root)
    if extract_root is None:
        extract_root = os.path.join(roots[0] if roots else os.getcwd(), "nfl_data")
    extract_root = os.path.abspath(extract_root)

    for root in roots:
        candidates = _candidate_data_dirs_from_csvs(root)
        if candidates:
            print(f"Found extracted CSV data under: {candidates[0]}")
            return candidates[0]

    if auto_unzip:
        zip_paths = _candidate_zip_paths(roots)
        if zip_paths:
            print(f"No extracted training CSVs found. Trying zip: {zip_paths[0]}")
            return _extract_zip_if_needed(zip_paths[0], extract_root)

    if try_kaggle_download:
        download_dir = os.path.join(extract_root, "_download")
        zip_path = _try_kaggle_competition_download(download_dir, competition_slug=competition_slug)
        if zip_path is not None:
            print(f"Downloaded Kaggle zip to: {zip_path}")
            return _extract_zip_if_needed(zip_path, extract_root)

    searched = "\n  - " + "\n  - ".join(roots)
    raise FileNotFoundError(
        "Could not find extracted training CSVs or a competition zip.\n"
        f"Searched roots:{searched}\n"
        f"Expected either extracted files like train/input_2023_w01.csv or a zip for {competition_slug}.\n"
        "If you already uploaded the zip, pass its parent folder as search_root. "
        "If not, upload kaggle.json and let the notebook download it automatically, or unzip manually."
    )


def discover_competition_files(data_dir: str) -> Dict[str, List[str]]:
    data_dir = os.path.abspath(data_dir)
    patterns = {
        "train_input": [
            "**/train_input*.csv",
            "**/train/train_input*.csv",
            "**/input_*.csv",
            "**/train/input_*.csv",
        ],
        "train_output": [
            "**/train_output*.csv",
            "**/train/train_output*.csv",
            "**/output_*.csv",
            "**/train/output_*.csv",
        ],
        "test_input": [
            "**/test_input*.csv",
            "**/test/test_input*.csv",
            "**/test_input.csv",
            "**/test/input_*.csv",
        ],
        "sample_submission": [
            "**/sample_submission*.csv",
            "**/*sample*submission*.csv",
            "**/test.csv",
            "**/test/test.csv",
        ],
    }

    found: Dict[str, List[str]] = {}
    for key, pats in patterns.items():
        files = []
        for pat in pats:
            files.extend(_recursive_glob(data_dir, pat))
        found[key] = sorted(set([p for p in files if os.path.isfile(p)]))
    return found


def concat_csvs(paths: List[str], usecols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    if not paths:
        return None
    dfs = []
    for p in paths:
        df = pd.read_csv(p, usecols=usecols)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def ensure_required_columns(df: pd.DataFrame, required_cols: List[str], df_name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


@dataclass
class ModelConfig:
    d_model: int = 64
    edge_dim: int = 16
    static_dim: int = 24
    obs_feat_dim: int = len(OBS_FEATURE_NAMES)
    num_graph_layers: int = 1
    dropout: float = 0.10
    friction_accel_limit: float = 6.5
    max_speed: float = 11.5
    max_yaw_rate: float = 6.0
    non_target_blend: float = 0.35
    lr: float = 2e-4
    batch_size: int = 2
    num_epochs: int = 14
    weight_decay: float = 1e-5
    val_frac: float = 0.15
    grad_clip: float = 1.0
    min_epochs: int = 4
    patience: int = 4
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss design: horizon-weighted tracking + velocity consistency + soft regularizers.
    track_loss_beta: float = 1.0
    velocity_loss_beta: float = 0.5
    horizon_weight_lambda: float = 1.5
    horizon_weight_power: float = 2.0
    target_player_weight: float = 2.5
    non_target_player_weight: float = 0.35

    # Fixed regularizer strengths.
    phys_loss_weight: float = 0.02
    attn_loss_weight: float = 0.001
    phys_warmup_frac: float = 0.20
    attn_warmup_frac: float = 0.20

    # Main supervised-loss weighting.
    # fixed: track + vel use manual weights.
    # fixed_normalized: manual weights after EMA normalization.
    # uncertainty_main: learn weights only for track/vel; keep regularizers fixed.
    loss_weighting: str = "uncertainty_main"
    vel_loss_weight: float = 0.20  # used in fixed / fixed_normalized modes
    normalize_main_loss_terms: bool = False
    normalize_regularizer_terms: bool = True
    loss_ema_decay: float = 0.95
    loss_norm_eps: float = 1e-6


class CategoryVocab:
    def __init__(self, values: List[str]):
        unique = sorted({str(v) for v in values if pd.notna(v)})
        self.itos = ["<UNK>"] + unique
        self.stoi = {v: i for i, v in enumerate(self.itos)}

    def encode(self, x) -> int:
        return self.stoi.get(str(x), 0)

    def __len__(self):
        return len(self.itos)


class NFLTrajectoryDataset(Dataset):
    def __init__(
        self,
        input_df: pd.DataFrame,
        output_df: Optional[pd.DataFrame],
        pos_vocab: CategoryVocab,
        side_vocab: CategoryVocab,
        role_vocab: CategoryVocab,
        play_keys: Optional[List[Tuple[int, int]]] = None,
    ):
        self.input_df = input_df
        self.output_df = output_df
        self.pos_vocab = pos_vocab
        self.side_vocab = side_vocab
        self.role_vocab = role_vocab

        self.input_groups = input_df.groupby(["game_id", "play_id"], sort=False).indices
        self.output_groups = output_df.groupby(["game_id", "play_id"], sort=False).indices if output_df is not None else {}
        all_keys = list(self.input_groups.keys())
        self.play_keys = play_keys if play_keys is not None else all_keys

    def __len__(self):
        return len(self.play_keys)

    def __getitem__(self, idx):
        key = self.play_keys[idx]
        inp = self.input_df.loc[self.input_groups[key]].copy()
        inp = inp.sort_values(["frame_id", "nfl_id"]).reset_index(drop=True)

        players_meta = (
            inp[[
                "nfl_id", "player_name", "player_position", "player_side", "player_role",
                "height_inches", "player_weight", "age_years", "is_target"
            ]]
            .drop_duplicates("nfl_id")
            .sort_values(["player_side", "nfl_id"])
            .reset_index(drop=True)
        )

        player_ids = players_meta["nfl_id"].tolist()
        player_to_idx = {pid: i for i, pid in enumerate(player_ids)}
        N = len(player_ids)

        frame_values = sorted(inp["frame_id"].unique().tolist())
        frame_to_t = {fid: t for t, fid in enumerate(frame_values)}
        T = len(frame_values)

        num_frames_output = int(inp["num_frames_output"].max()) if "num_frames_output" in inp.columns else 1
        H = num_frames_output

        x_seq = np.zeros((T, N), dtype=np.float32)
        y_seq = np.zeros((T, N), dtype=np.float32)
        vx_seq = np.zeros((T, N), dtype=np.float32)
        vy_seq = np.zeros((T, N), dtype=np.float32)
        ax_seq = np.zeros((T, N), dtype=np.float32)
        ay_seq = np.zeros((T, N), dtype=np.float32)
        atan_seq = np.zeros((T, N), dtype=np.float32)
        alat_seq = np.zeros((T, N), dtype=np.float32)
        speed_own_seq = np.zeros((T, N), dtype=np.float32)
        s_official_seq = np.zeros((T, N), dtype=np.float32)
        a_official_seq = np.zeros((T, N), dtype=np.float32)
        a_residual_seq = np.zeros((T, N), dtype=np.float32)
        dir_seq = np.zeros((T, N), dtype=np.float32)
        o_seq = np.zeros((T, N), dtype=np.float32)
        gap_seq = np.zeros((T, N), dtype=np.float32)
        turn_rate_seq = np.zeros((T, N), dtype=np.float32)
        obs_mask = np.zeros((T, N), dtype=np.float32)

        for row in inp.itertuples(index=False):
            t = frame_to_t[int(row.frame_id)]
            j = player_to_idx[row.nfl_id]
            x_seq[t, j] = float(row.x)
            y_seq[t, j] = float(row.y)
            vx_seq[t, j] = float(row.vx_own)
            vy_seq[t, j] = float(row.vy_own)
            ax_seq[t, j] = float(row.ax_own)
            ay_seq[t, j] = float(row.ay_own)
            atan_seq[t, j] = float(row.a_tan_own)
            alat_seq[t, j] = float(row.a_lat_own)
            speed_own_seq[t, j] = float(row.speed_own)
            s_official_seq[t, j] = float(row.s_official)
            a_official_seq[t, j] = float(row.a_official)
            a_residual_seq[t, j] = float(row.a_residual)
            dir_seq[t, j] = math.radians(float(row.dir))
            o_seq[t, j] = math.radians(float(row.o))
            gap_seq[t, j] = math.radians(float(row.body_motion_gap_deg))
            turn_rate_seq[t, j] = float(row.turn_rate_own)
            obs_mask[t, j] = 1.0

        fill_arrays = [
            x_seq, y_seq, vx_seq, vy_seq, ax_seq, ay_seq,
            atan_seq, alat_seq, speed_own_seq,
            s_official_seq, a_official_seq, a_residual_seq,
            dir_seq, o_seq, gap_seq, turn_rate_seq
        ]
        for arr in fill_arrays:
            for j in range(N):
                last = 0.0
                for t in range(T):
                    if obs_mask[t, j] == 1.0:
                        last = arr[t, j]
                    else:
                        arr[t, j] = last

        static_cont = np.stack([
            players_meta["height_inches"].astype(np.float32).to_numpy(),
            players_meta["player_weight"].astype(np.float32).to_numpy(),
            players_meta["age_years"].astype(np.float32).to_numpy(),
            players_meta["is_target"].astype(np.float32).to_numpy(),
        ], axis=-1)

        pos_ids = np.array([self.pos_vocab.encode(v) for v in players_meta["player_position"]], dtype=np.int64)
        side_ids = np.array([self.side_vocab.encode(v) for v in players_meta["player_side"]], dtype=np.int64)
        role_ids = np.array([self.role_vocab.encode(v) for v in players_meta["player_role"]], dtype=np.int64)

        play_globals = {
            "ball_land_x": float(inp["ball_land_x"].iloc[0]) if "ball_land_x" in inp.columns else 60.0,
            "ball_land_y": float(inp["ball_land_y"].iloc[0]) if "ball_land_y" in inp.columns else FIELD_Y_MAX / 2.0,
            "absolute_yardline_number": float(inp["absolute_yardline_number"].iloc[0]) if "absolute_yardline_number" in inp.columns else 60.0,
            "num_frames_output": float(H),
        }

        obs_feats = np.stack([
            x_seq, y_seq,
            vx_seq, vy_seq,
            ax_seq, ay_seq,
            atan_seq, alat_seq,
            speed_own_seq,
            s_official_seq,
            a_official_seq,
            a_residual_seq,
            np.sin(dir_seq), np.cos(dir_seq),
            np.sin(o_seq), np.cos(o_seq),
            np.sin(gap_seq), np.cos(gap_seq),
            turn_rate_seq,
            np.full_like(x_seq, play_globals["ball_land_x"]),
            np.full_like(x_seq, play_globals["ball_land_y"]),
            np.full_like(x_seq, play_globals["absolute_yardline_number"]),
            np.full_like(x_seq, play_globals["num_frames_output"]),
            obs_mask,
        ], axis=-1).astype(np.float32)

        target_xy = np.full((H, N, 2), np.nan, dtype=np.float32)
        target_mask = np.zeros((H, N), dtype=np.float32)

        if self.output_df is not None and key in self.output_groups:
            out = self.output_df.loc[self.output_groups[key]].copy()
            out = out.sort_values(["frame_id", "nfl_id"])
            if "x" in out.columns and "y" in out.columns:
                for row in out.itertuples(index=False):
                    if row.nfl_id in player_to_idx:
                        h = int(row.frame_id) - 1
                        if 0 <= h < H:
                            j = player_to_idx[row.nfl_id]
                            target_xy[h, j, 0] = float(row.x)
                            target_xy[h, j, 1] = float(row.y)
                            target_mask[h, j] = 1.0

        return {
            "key": key,
            "game_id": key[0],
            "play_id": key[1],
            "player_ids": np.array(player_ids, dtype=np.int64),
            "obs_feats": obs_feats,
            "obs_mask": obs_mask.astype(np.float32),
            "static_cont": static_cont.astype(np.float32),
            "pos_ids": pos_ids,
            "side_ids": side_ids,
            "role_ids": role_ids,
            "target_xy": target_xy,
            "target_mask": target_mask,
            "num_frames_output": H,
            "ball_land_xy": np.array([play_globals["ball_land_x"], play_globals["ball_land_y"]], dtype=np.float32),
        }


def collate_plays(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    T_max = max(item["obs_feats"].shape[0] for item in batch)
    N_max = max(item["obs_feats"].shape[1] for item in batch)
    H_max = max(item["target_xy"].shape[0] for item in batch)
    F = batch[0]["obs_feats"].shape[-1]
    S = batch[0]["static_cont"].shape[-1]

    obs_feats = torch.zeros(B, T_max, N_max, F, dtype=torch.float32)
    obs_mask = torch.zeros(B, T_max, N_max, dtype=torch.float32)
    static_cont = torch.zeros(B, N_max, S, dtype=torch.float32)
    pos_ids = torch.zeros(B, N_max, dtype=torch.long)
    side_ids = torch.zeros(B, N_max, dtype=torch.long)
    role_ids = torch.zeros(B, N_max, dtype=torch.long)
    node_mask = torch.zeros(B, N_max, dtype=torch.float32)
    target_xy = torch.full((B, H_max, N_max, 2), float("nan"), dtype=torch.float32)
    target_mask = torch.zeros(B, H_max, N_max, dtype=torch.float32)
    ball_land_xy = torch.zeros(B, 2, dtype=torch.float32)
    player_ids = torch.full((B, N_max), -1, dtype=torch.long)
    obs_lengths = torch.zeros(B, dtype=torch.long)
    hor_lengths = torch.zeros(B, dtype=torch.long)

    game_ids = []
    play_ids = []

    for i, item in enumerate(batch):
        T, N, _ = item["obs_feats"].shape
        H = item["target_xy"].shape[0]
        obs_feats[i, :T, :N] = torch.from_numpy(item["obs_feats"])
        obs_mask[i, :T, :N] = torch.from_numpy(item["obs_mask"])
        static_cont[i, :N] = torch.from_numpy(item["static_cont"])
        pos_ids[i, :N] = torch.from_numpy(item["pos_ids"])
        side_ids[i, :N] = torch.from_numpy(item["side_ids"])
        role_ids[i, :N] = torch.from_numpy(item["role_ids"])
        node_mask[i, :N] = 1.0
        target_xy[i, :H, :N] = torch.from_numpy(item["target_xy"])
        target_mask[i, :H, :N] = torch.from_numpy(item["target_mask"])
        ball_land_xy[i] = torch.from_numpy(item["ball_land_xy"])
        player_ids[i, :N] = torch.from_numpy(item["player_ids"])
        obs_lengths[i] = T
        hor_lengths[i] = H
        game_ids.append(item["game_id"])
        play_ids.append(item["play_id"])

    return {
        "obs_feats": obs_feats,
        "obs_mask": obs_mask,
        "static_cont": static_cont,
        "pos_ids": pos_ids,
        "side_ids": side_ids,
        "role_ids": role_ids,
        "node_mask": node_mask,
        "target_xy": target_xy,
        "target_mask": target_mask,
        "ball_land_xy": ball_land_xy,
        "player_ids": player_ids,
        "obs_lengths": obs_lengths,
        "hor_lengths": hor_lengths,
        "game_ids": game_ids,
        "play_ids": play_ids,
    }


class GraphInteractionBlock(nn.Module):
    def __init__(self, d_model: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.edge_msg = nn.Sequential(
            nn.Linear(edge_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.out = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_feat, node_mask):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn_logits = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(x.shape[-1])
        attn_logits = attn_logits + self.edge_bias(edge_feat).squeeze(-1)

        valid = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        attn_logits = attn_logits.masked_fill(valid == 0, -1e9)
        attn = torch.softmax(attn_logits, dim=-1)

        edge_msg = self.edge_msg(edge_feat)
        msg = torch.einsum("bij,bjd->bid", attn, v) + torch.einsum("bij,bijd->bid", attn, edge_msg)

        out = self.norm(x + self.out(torch.cat([x, msg], dim=-1)))
        return out, attn


def pairwise_edge_features(pos, vel, side_ids, role_ids, target_flag):
    rel_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
    rel_vel = vel.unsqueeze(2) - vel.unsqueeze(1)

    dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True).clamp_min(1e-3)
    unit = rel_pos / dist
    closing = (rel_vel * unit).sum(dim=-1, keepdim=True)
    dot_rv = (rel_pos * rel_vel).sum(dim=-1, keepdim=True)
    speed_sq = (rel_vel ** 2).sum(dim=-1, keepdim=True) + 1e-3
    ttc = (-dot_rv / speed_sq).clamp(-10.0, 10.0)
    los_rate = (
        rel_pos[..., 0:1] * rel_vel[..., 1:2] - rel_pos[..., 1:2] * rel_vel[..., 0:1]
    ) / (dist ** 2 + 1e-3)

    same_side = (side_ids.unsqueeze(2) == side_ids.unsqueeze(1)).float().unsqueeze(-1)
    same_role = (role_ids.unsqueeze(2) == role_ids.unsqueeze(1)).float().unsqueeze(-1)
    tgt_pair = (target_flag.unsqueeze(2) * target_flag.unsqueeze(1)).unsqueeze(-1)

    feat = torch.cat([
        rel_pos,
        rel_vel,
        dist,
        unit,
        closing,
        ttc,
        los_rate,
        same_side,
        same_role,
        tgt_pair,
    ], dim=-1)
    return feat


class PhysicsAwareDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.ctrl_head = nn.Sequential(
            nn.Linear(config.d_model + 17, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 3),
        )

    def forward(self, h, pos, vel, acc_prev, official_a_aux, body_ang, move_ang, ball_ctx, target_flag):
        ball_rel = ball_ctx.unsqueeze(1) - pos
        speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
        gap = wrap_to_pi(move_ang - body_ang)

        decoder_in = torch.cat([
            h,
            pos,
            vel,
            acc_prev,
            ball_rel,
            official_a_aux,
            target_flag.unsqueeze(-1),
            torch.sin(body_ang), torch.cos(body_ang),
            torch.sin(move_ang), torch.cos(move_ang),
            torch.sin(gap), torch.cos(gap),
            speed,
        ], dim=-1)

        raw = self.ctrl_head(decoder_in)
        a_forward = raw[..., 0:1]
        a_lat = raw[..., 1:2]
        yaw_ctrl = torch.tanh(raw[..., 2:3]) * self.config.max_yaw_rate

        orient_gate = 0.25 + 0.75 * torch.sigmoid(2.0 * torch.cos(gap))
        a_lat_eff = a_lat * orient_gate

        a_local = torch.cat([a_forward, a_lat_eff], dim=-1)
        a_norm = torch.linalg.norm(a_local, dim=-1, keepdim=True).clamp_min(1e-6)
        a_max = self.config.friction_accel_limit
        scale = torch.clamp(a_max / a_norm, max=1.0)
        overflow = F.relu(a_norm - a_max)
        a_local_clip = a_local * scale

        c = torch.cos(body_ang)
        s = torch.sin(body_ang)
        ax = a_local_clip[..., 0:1] * c - a_local_clip[..., 1:2] * s
        ay = a_local_clip[..., 0:1] * s + a_local_clip[..., 1:2] * c
        acc_next = torch.cat([ax, ay], dim=-1)

        vel_next = vel + DT * acc_next
        speed_next = torch.linalg.norm(vel_next, dim=-1, keepdim=True).clamp_min(1e-6)
        speed_scale = torch.clamp(self.config.max_speed / speed_next, max=1.0)
        vel_next = vel_next * speed_scale

        pos_next_raw = pos + DT * vel_next
        pos_next = torch.stack([
            pos_next_raw[..., 0].clamp(FIELD_X_MIN, FIELD_X_MAX),
            pos_next_raw[..., 1].clamp(FIELD_Y_MIN, FIELD_Y_MAX),
        ], dim=-1)

        desired_move_ang = torch.atan2(vel_next[..., 1:2], vel_next[..., 0:1] + 1e-6)
        body_next = body_ang + DT * yaw_ctrl + 0.15 * wrap_to_pi(desired_move_ang - body_ang)
        move_next = desired_move_ang

        pos_cv = pos + DT * vel
        vel_cv = vel
        non_target = (1.0 - target_flag.unsqueeze(-1))
        alpha = self.config.non_target_blend
        pos_next = pos_next * (1.0 - non_target * alpha) + pos_cv * (non_target * alpha)
        vel_next = vel_next * (1.0 - non_target * alpha) + vel_cv * (non_target * alpha)

        return pos_next, vel_next, acc_next, body_next, move_next, overflow.squeeze(-1)


class TrajectoryGNN(nn.Module):
    def __init__(self, config: ModelConfig, n_pos: int, n_side: int, n_role: int):
        super().__init__()
        self.config = config
        self.pos_emb = nn.Embedding(n_pos, 16)
        self.side_emb = nn.Embedding(n_side, 4)
        self.role_emb = nn.Embedding(n_role, 16)

        self.static_proj = nn.Sequential(
            nn.Linear(4 + 16 + 4 + 16, config.static_dim),
            nn.SiLU(),
            nn.Linear(config.static_dim, config.static_dim),
        )
        self.obs_proj = nn.Sequential(
            nn.Linear(config.obs_feat_dim + config.static_dim, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.obs_graph = nn.ModuleList([
            GraphInteractionBlock(config.d_model, config.edge_dim, config.dropout)
            for _ in range(config.num_graph_layers)
        ])
        self.obs_edge_proj = nn.Sequential(
            nn.Linear(13, config.edge_dim),
            nn.SiLU(),
            nn.Linear(config.edge_dim, config.edge_dim),
        )
        self.obs_gru = nn.GRU(config.d_model, config.d_model, batch_first=True)

        self.dec_graph = nn.ModuleList([
            GraphInteractionBlock(config.d_model, config.edge_dim, config.dropout)
            for _ in range(config.num_graph_layers)
        ])
        self.dec_edge_proj = nn.Sequential(
            nn.Linear(13, config.edge_dim),
            nn.SiLU(),
            nn.Linear(config.edge_dim, config.edge_dim),
        )
        self.decoder = PhysicsAwareDecoder(config)

    def encode_static(self, static_cont, pos_ids, side_ids, role_ids):
        x = torch.cat([
            static_cont,
            self.pos_emb(pos_ids),
            self.side_emb(side_ids),
            self.role_emb(role_ids),
        ], dim=-1)
        return self.static_proj(x)

    def forward(self, batch):
        obs_feats = batch["obs_feats"]
        static_cont = batch["static_cont"]
        pos_ids = batch["pos_ids"]
        side_ids = batch["side_ids"]
        role_ids = batch["role_ids"]
        node_mask = batch["node_mask"]
        hor_lengths = batch["hor_lengths"]

        B, T, N, Fdim = obs_feats.shape
        H = int(hor_lengths.max().item())

        static_ctx = self.encode_static(static_cont, pos_ids, side_ids, role_ids)
        static_seq = static_ctx.unsqueeze(1).expand(B, T, N, static_ctx.shape[-1])
        obs_in = self.obs_proj(torch.cat([obs_feats, static_seq], dim=-1))

        flat = obs_in.permute(0, 2, 1, 3).reshape(B * N, T, self.config.d_model)
        _, h_last = self.obs_gru(flat)
        h = h_last.squeeze(0).reshape(B, N, self.config.d_model)

        last_idx = (batch["obs_lengths"] - 1).view(B, 1, 1, 1).expand(B, 1, N, Fdim)
        last_frame = torch.gather(obs_feats, 1, last_idx).squeeze(1)

        pos = last_frame[..., [OBS_FEATURE_INDEX["x"], OBS_FEATURE_INDEX["y"]]]
        vel = last_frame[..., [OBS_FEATURE_INDEX["vx_own"], OBS_FEATURE_INDEX["vy_own"]]]
        acc_prev = last_frame[..., [OBS_FEATURE_INDEX["ax_own"], OBS_FEATURE_INDEX["ay_own"]]]
        official_a_aux = last_frame[..., OBS_FEATURE_INDEX["a_official"]:OBS_FEATURE_INDEX["a_official"] + 1]

        body_ang = torch.atan2(
            last_frame[..., OBS_FEATURE_INDEX["sin_o"]:OBS_FEATURE_INDEX["sin_o"] + 1],
            last_frame[..., OBS_FEATURE_INDEX["cos_o"]:OBS_FEATURE_INDEX["cos_o"] + 1] + 1e-6,
        )
        move_ang = torch.atan2(vel[..., 1:2], vel[..., 0:1] + 1e-6)
        target_flag = static_cont[..., 3]

        edge_raw = pairwise_edge_features(pos, vel, side_ids, role_ids, target_flag)
        edge_feat = self.obs_edge_proj(edge_raw)
        for layer in self.obs_graph:
            h, _ = layer(h, edge_feat, node_mask)

        preds = []
        overflows = []
        attn_seq = []

        for _ in range(H):
            edge_raw = pairwise_edge_features(pos, vel, side_ids, role_ids, target_flag)
            edge_feat = self.dec_edge_proj(edge_raw)
            for layer in self.dec_graph:
                h, attn = layer(h, edge_feat, node_mask)
                attn_seq.append(attn)

            pos, vel, acc_prev, body_ang, move_ang, overflow = self.decoder(
                h=h,
                pos=pos,
                vel=vel,
                acc_prev=acc_prev,
                official_a_aux=official_a_aux,
                body_ang=body_ang,
                move_ang=move_ang,
                ball_ctx=batch["ball_land_xy"],
                target_flag=target_flag,
            )
            preds.append(pos)
            overflows.append(overflow)

        pred_xy = torch.stack(preds, dim=1)
        overflow = torch.stack(overflows, dim=1)
        return {
            "pred_xy": pred_xy,
            "overflow": overflow,
            "attn_seq": attn_seq,
        }


def masked_smooth_l1(pred, target, mask, beta=1.0):
    valid = mask > 0
    if valid.sum() == 0:
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[valid], target[valid], beta=beta)


class LossWeightController(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.main_loss_names = ["track_loss", "vel_loss"]
        self.regularizer_loss_names = ["phys_loss", "attn_loss"]
        self.loss_names = self.main_loss_names + self.regularizer_loss_names

        self.fixed_weights = {
            "track_loss": 1.0,
            "vel_loss": config.vel_loss_weight,
            "phys_loss": config.phys_loss_weight,
            "attn_loss": config.attn_loss_weight,
        }

        for name in self.loss_names:
            self.register_buffer(f"ema_{name}", torch.tensor(1.0, dtype=torch.float32))

        if config.loss_weighting == "uncertainty_main":
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1, dtype=torch.float32))
                for name in self.main_loss_names
            })
        else:
            self.log_vars = None

    def current_warmup_scale(self, loss_name: str, epoch_idx: int) -> float:
        if loss_name == "phys_loss":
            frac = self.config.phys_warmup_frac
        elif loss_name == "attn_loss":
            frac = self.config.attn_warmup_frac
        else:
            return 1.0
        if frac <= 0:
            return 1.0
        warmup_epochs = max(1, int(math.ceil(self.config.num_epochs * frac)))
        return min(1.0, float(epoch_idx + 1) / float(warmup_epochs))

    def should_normalize(self, loss_name: str) -> bool:
        if self.config.loss_weighting == "fixed_normalized":
            return True
        if loss_name in self.main_loss_names:
            return self.config.normalize_main_loss_terms
        return self.config.normalize_regularizer_terms

    def maybe_normalize(self, loss_name: str, loss_value: torch.Tensor, update_stats: bool) -> torch.Tensor:
        if not self.should_normalize(loss_name):
            return loss_value
        ema_buf = getattr(self, f"ema_{loss_name}")
        if update_stats:
            ema_buf.mul_(self.config.loss_ema_decay).add_(loss_value.detach().to(ema_buf.device) * (1.0 - self.config.loss_ema_decay))
        denom = ema_buf.detach().clamp_min(self.config.loss_norm_eps).to(loss_value.device)
        return loss_value / denom

    def _combine_main_losses(self, normalized_losses: Dict[str, torch.Tensor], metrics: Dict[str, float]) -> torch.Tensor:
        if self.config.loss_weighting == "uncertainty_main":
            total_main = None
            for name in self.main_loss_names:
                loss_value = normalized_losses[name]
                log_var = self.log_vars[name].to(loss_value.device)
                weighted = 0.5 * torch.exp(-log_var) * loss_value + 0.5 * log_var
                total_main = weighted if total_main is None else total_main + weighted
                metrics[f"{name}_weight"] = float((0.5 * torch.exp(-log_var.detach())).item())
                metrics[f"{name}_log_var"] = float(log_var.detach().item())
            return total_main

        total_main = None
        for name in self.main_loss_names:
            weight = self.fixed_weights[name]
            weighted = weight * normalized_losses[name]
            total_main = weighted if total_main is None else total_main + weighted
            metrics[f"{name}_weight"] = float(weight)
        return total_main

    def combine(self, raw_losses: Dict[str, torch.Tensor], epoch_idx: int = 0, update_stats: bool = False):
        metrics: Dict[str, float] = {}
        normalized_losses: Dict[str, torch.Tensor] = {}

        for name, raw_loss in raw_losses.items():
            normalized = self.maybe_normalize(name, raw_loss, update_stats=update_stats)
            normalized_losses[name] = normalized
            metrics[f"{name}_raw"] = raw_loss.detach().item()
            metrics[f"{name}_norm"] = normalized.detach().item()

        total = self._combine_main_losses(normalized_losses, metrics)

        for name in self.regularizer_loss_names:
            base_weight = self.fixed_weights[name] * self.current_warmup_scale(name, epoch_idx)
            total = total + base_weight * normalized_losses[name]
            metrics[f"{name}_weight"] = float(base_weight)

        metrics["loss"] = total.detach().item()
        return total, metrics


def _safe_weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, beta: float) -> torch.Tensor:
    if weights.numel() == 0 or float(weights.sum().detach().item()) <= 0.0:
        return pred.sum() * 0.0
    safe_target = torch.where(weights > 0, target, pred.detach())
    per_elem = F.smooth_l1_loss(pred, safe_target, beta=beta, reduction="none")
    return (per_elem * weights).sum() / weights.sum().clamp_min(1e-6)


def _build_time_player_weights(batch, pred_xy: torch.Tensor, config: ModelConfig):
    B, H, N, _ = pred_xy.shape
    device = pred_xy.device
    hor_lengths = batch["hor_lengths"].to(device).float().view(B, 1, 1)
    time_index = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    denom = torch.clamp(hor_lengths - 1.0, min=1.0)
    time_frac = torch.minimum(time_index / denom, torch.ones_like(time_index))
    horizon_weights = 1.0 + config.horizon_weight_lambda * time_frac.pow(config.horizon_weight_power)
    valid_time = (time_index < hor_lengths).float()

    target_flag = batch["static_cont"][..., 3].to(device).unsqueeze(1)
    player_weights = torch.full((B, 1, N), config.non_target_player_weight, device=device, dtype=torch.float32)
    player_weights = player_weights + target_flag * (config.target_player_weight - config.non_target_player_weight)
    node_weights = batch["node_mask"].to(device).unsqueeze(1)
    return horizon_weights * valid_time * player_weights * node_weights


def _compute_track_loss(pred_xy: torch.Tensor, target_xy: torch.Tensor, target_mask: torch.Tensor, batch, config: ModelConfig):
    base_weights = _build_time_player_weights(batch, pred_xy, config)
    mask = target_mask.to(pred_xy.device) * base_weights
    coord_weights = mask.unsqueeze(-1).expand_as(pred_xy)
    return _safe_weighted_smooth_l1(pred_xy, target_xy.to(pred_xy.device), coord_weights, beta=config.track_loss_beta), mask


def _compute_velocity_consistency_loss(pred_xy: torch.Tensor, target_xy: torch.Tensor, target_mask: torch.Tensor, base_mask: torch.Tensor, config: ModelConfig):
    if pred_xy.shape[1] <= 1:
        return pred_xy.sum() * 0.0
    pred_delta = pred_xy[:, 1:] - pred_xy[:, :-1]
    true_delta = target_xy[:, 1:].to(pred_xy.device) - target_xy[:, :-1].to(pred_xy.device)
    delta_mask = (target_mask[:, 1:].to(pred_xy.device) * target_mask[:, :-1].to(pred_xy.device)) * base_mask[:, 1:]
    coord_weights = delta_mask.unsqueeze(-1).expand_as(pred_delta)
    return _safe_weighted_smooth_l1(pred_delta, true_delta, coord_weights, beta=config.velocity_loss_beta)


def _compute_physics_penalty(model_out, batch, config: ModelConfig):
    overflow = model_out["overflow"]
    base_weights = _build_time_player_weights(batch, model_out["pred_xy"], config)
    if overflow.numel() == 0 or float(base_weights.sum().detach().item()) <= 0.0:
        return overflow.sum() * 0.0
    return (overflow * base_weights).sum() / base_weights.sum().clamp_min(1e-6)


def _compute_attention_smoothness(model_out):
    attn_seq = model_out.get("attn_seq", [])
    pred_xy = model_out["pred_xy"]
    if len(attn_seq) <= 1:
        return pred_xy.sum() * 0.0
    diffs = [(attn_seq[i] - attn_seq[i - 1]).abs().mean() for i in range(1, len(attn_seq))]
    return torch.stack(diffs).mean()


def compute_loss(model_out, batch, config: Optional[ModelConfig] = None, loss_controller: Optional[LossWeightController] = None, epoch_idx: int = 0, update_controller: bool = False):
    config = config or ModelConfig()
    pred_xy = model_out["pred_xy"]
    target_xy = batch["target_xy"]
    target_mask = batch["target_mask"]

    track_loss, base_mask = _compute_track_loss(pred_xy, target_xy, target_mask, batch, config)
    vel_loss = _compute_velocity_consistency_loss(pred_xy, target_xy, target_mask, base_mask, config)
    phys_loss = _compute_physics_penalty(model_out, batch, config)
    attn_loss = _compute_attention_smoothness(model_out)

    raw_losses = {
        "track_loss": track_loss,
        "vel_loss": vel_loss,
        "phys_loss": phys_loss,
        "attn_loss": attn_loss,
    }

    if loss_controller is None:
        total = (
            raw_losses["track_loss"]
            + config.vel_loss_weight * raw_losses["vel_loss"]
            + config.phys_loss_weight * raw_losses["phys_loss"]
            + config.attn_loss_weight * raw_losses["attn_loss"]
        )
        metrics = {
            "loss": total.detach().item(),
            "track_loss_weight": 1.0,
            "vel_loss_weight": float(config.vel_loss_weight),
            "phys_loss_weight": float(config.phys_loss_weight),
            "attn_loss_weight": float(config.attn_loss_weight),
        }
        for name, loss_value in raw_losses.items():
            metrics[f"{name}_raw"] = loss_value.detach().item()
            metrics[f"{name}_norm"] = loss_value.detach().item()
        return total, metrics

    total, metrics = loss_controller.combine(raw_losses, epoch_idx=epoch_idx, update_stats=update_controller)
    return total, metrics


def split_play_keys(keys: List[Tuple[int, int]], val_frac: float = 0.15, seed: int = 42):
    rng = random.Random(seed)
    keys = list(keys)
    rng.shuffle(keys)
    if len(keys) <= 1:
        return keys, keys
    n_val = max(1, int(len(keys) * val_frac))
    n_val = min(n_val, len(keys) - 1)
    val_keys = keys[:n_val]
    train_keys = keys[n_val:]
    if len(train_keys) == 0:
        train_keys = val_keys
    if len(val_keys) == 0:
        val_keys = train_keys[:1]
    return train_keys, val_keys


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}
    keys = sorted({k for metrics in metric_list for k in metrics.keys()})
    out = {}
    for key in keys:
        vals = [metrics[key] for metrics in metric_list if key in metrics]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def evaluate(model, loader, device, config: ModelConfig, loss_controller: LossWeightController):
    model.eval()
    loss_controller.eval()
    metric_list = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            _, metrics = compute_loss(out, batch, config=config, loss_controller=loss_controller, update_controller=False)
            metric_list.append(metrics)
    return aggregate_metrics(metric_list)


def train_model(model, train_loader, val_loader, config: ModelConfig, save_path: str):
    device = config.device
    model.to(device)
    loss_controller = LossWeightController(config).to(device)
    optim_params = list(model.parameters()) + list(loss_controller.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=config.lr, weight_decay=config.weight_decay)
    best_val = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(config.num_epochs):
        model.train()
        loss_controller.train()
        train_metric_list = []

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss, metrics = compute_loss(
                out,
                batch,
                config=config,
                loss_controller=loss_controller,
                epoch_idx=epoch,
                update_controller=True,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_metric_list.append(metrics)

        train_metrics = aggregate_metrics(train_metric_list)
        val_metrics = evaluate(model, val_loader, device, config=config, loss_controller=loss_controller)
        train_loss = train_metrics.get("loss", float("nan"))
        val_loss = val_metrics.get("loss", float("nan"))
        history_row = {"epoch": epoch + 1}
        history_row.update({f"train_{k}": v for k, v in train_metrics.items()})
        history_row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(history_row)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train {train_loss:.5f} | val {val_loss:.5f} | "
            f"track {train_metrics.get('track_loss_raw', float('nan')):.5f} "
            f"(w={train_metrics.get('track_loss_weight', float('nan')):.4f}) | "
            f"vel {train_metrics.get('vel_loss_raw', float('nan')):.5f} "
            f"(w={train_metrics.get('vel_loss_weight', float('nan')):.4f}) | "
            f"phys {train_metrics.get('phys_loss_raw', float('nan')):.5f}"
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        if epoch + 1 >= config.min_epochs and epoch - best_epoch >= config.patience:
            print("Early stopping.")
            break

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    return history


def build_vocabs(input_df: pd.DataFrame):
    return (
        CategoryVocab(input_df["player_position"].astype(str).tolist()),
        CategoryVocab(input_df["player_side"].astype(str).tolist()),
        CategoryVocab(input_df["player_role"].astype(str).tolist()),
    )


def preprocess_input_df(df: pd.DataFrame) -> pd.DataFrame:
    ensure_required_columns(df, INPUT_REQUIRED_COLS, "train_input/test_input")
    df = canonicalize_play_direction(df)
    df = add_derived_observed_features(df)
    return df


def preprocess_output_df(df: pd.DataFrame, input_ref_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    ensure_required_columns(df, OUTPUT_REQUIRED_COLS, "train_output")
    out = df.copy()
    if input_ref_df is not None and "play_direction" in input_ref_df.columns:
        play_dir = (
            input_ref_df[["game_id", "play_id", "play_direction"]]
            .drop_duplicates(["game_id", "play_id"])
        )
        out = out.merge(play_dir, on=["game_id", "play_id"], how="left")
        out = canonicalize_play_direction(out)
        out = out.drop(columns=["play_direction_canonical"], errors="ignore")
    return out


def fit_pipeline(data_dir: str, artifact_dir: str = "./artifacts", config: Optional[ModelConfig] = None):
    os.makedirs(artifact_dir, exist_ok=True)
    config = config or ModelConfig()
    seed_everything(config.seed)

    data_dir = os.path.abspath(data_dir)
    if os.path.isfile(data_dir) and data_dir.lower().endswith(".zip"):
        print(f"data_dir points to a zip file. Extracting automatically: {data_dir}")
        data_dir = _extract_zip_if_needed(data_dir, os.path.join(os.path.dirname(data_dir), "extracted_competition_data"))
    elif not os.path.exists(data_dir):
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    files = discover_competition_files(data_dir)
    train_input_paths = files["train_input"]
    train_output_paths = files["train_output"]

    if not train_input_paths or not train_output_paths:
        zip_paths = _candidate_zip_paths(data_dir)
        if zip_paths:
            print(f"No training CSVs found directly under {data_dir}. Trying zip: {zip_paths[0]}")
            data_dir = _extract_zip_if_needed(zip_paths[0], os.path.join(data_dir, "auto_unzipped"))
            files = discover_competition_files(data_dir)
            train_input_paths = files["train_input"]
            train_output_paths = files["train_output"]

    if not train_input_paths or not train_output_paths:
        raise FileNotFoundError(
            "Could not find training CSVs recursively under data_dir. "
            "Accepted input names include train_input*.csv or input_*.csv; "
            "accepted output names include train_output*.csv or output_*.csv. "
            f"Scanned root: {os.path.abspath(data_dir)} | "
            f"found train_input={train_input_paths[:5]} | found train_output={train_output_paths[:5]}. "
            "For the Kaggle layout in your screenshot, data_dir should be the folder above train/, "
            "or you can pass the zip path directly and this version will unzip it."
        )

    print("Resolved DATA_DIR:", data_dir)
    print("Discovered train_input files:", json.dumps(train_input_paths[:5], ensure_ascii=False, indent=2))
    print("Discovered train_output files:", json.dumps(train_output_paths[:5], ensure_ascii=False, indent=2))

    train_input = concat_csvs(train_input_paths, usecols=INPUT_REQUIRED_COLS)
    train_output = concat_csvs(train_output_paths, usecols=OUTPUT_REQUIRED_COLS)

    print("Preprocessing train_input...")
    train_input = preprocess_input_df(train_input)
    print("Preprocessing train_output...")
    train_output = preprocess_output_df(train_output, train_input)

    pos_vocab, side_vocab, role_vocab = build_vocabs(train_input)

    all_keys = sorted(train_input.groupby(["game_id", "play_id"]).size().index.tolist())
    train_keys, val_keys = split_play_keys(all_keys, config.val_frac, config.seed)

    train_ds = NFLTrajectoryDataset(train_input, train_output, pos_vocab, side_vocab, role_vocab, train_keys)
    val_ds = NFLTrajectoryDataset(train_input, train_output, pos_vocab, side_vocab, role_vocab, val_keys)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_plays,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_plays,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = TrajectoryGNN(config, len(pos_vocab), len(side_vocab), len(role_vocab))
    save_path = os.path.join(artifact_dir, "best_model.pt")
    history = train_model(model, train_loader, val_loader, config, save_path)

    bundle = {
        "config": asdict(config),
        "pos_vocab": pos_vocab.itos,
        "side_vocab": side_vocab.itos,
        "role_vocab": role_vocab.itos,
        "history": history,
        "obs_feature_names": OBS_FEATURE_NAMES,
    }
    with open(os.path.join(artifact_dir, "bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    pd.DataFrame(history).to_csv(os.path.join(artifact_dir, "training_history.csv"), index=False)
    print(f"Saved artifacts to {artifact_dir}")
    return model, bundle

def load_bundle(artifact_dir: str):
    with open(os.path.join(artifact_dir, "bundle.pkl"), "rb") as f:
        bundle = pickle.load(f)

    config = ModelConfig(**bundle["config"])

    pos_vocab = CategoryVocab([])
    pos_vocab.itos = bundle["pos_vocab"]
    pos_vocab.stoi = {v: i for i, v in enumerate(pos_vocab.itos)}

    side_vocab = CategoryVocab([])
    side_vocab.itos = bundle["side_vocab"]
    side_vocab.stoi = {v: i for i, v in enumerate(side_vocab.itos)}

    role_vocab = CategoryVocab([])
    role_vocab.itos = bundle["role_vocab"]
    role_vocab.stoi = {v: i for i, v in enumerate(role_vocab.itos)}

    model = TrajectoryGNN(config, len(pos_vocab), len(side_vocab), len(role_vocab))
    model.load_state_dict(torch.load(os.path.join(artifact_dir, "best_model.pt"), map_location=config.device))
    model.to(config.device)
    model.eval()
    return model, config, pos_vocab, side_vocab, role_vocab, bundle


def predict_from_test_input(
    test_input_path: str,
    artifact_dir: str,
    template_path: Optional[str] = None,
    output_path: str = "submission.csv",
    full_output_path: str = "predictions_full.csv",
):
    model, config, pos_vocab, side_vocab, role_vocab, _ = load_bundle(artifact_dir)

    test_input = pd.read_csv(test_input_path, usecols=INPUT_REQUIRED_COLS)
    test_input = preprocess_input_df(test_input)

    ds = NFLTrajectoryDataset(test_input, None, pos_vocab, side_vocab, role_vocab, None)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_plays, num_workers=0)

    rows = []
    device = config.device
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            pred = out["pred_xy"].cpu().numpy()

            B = pred.shape[0]
            for b in range(B):
                game_id = batch["game_ids"][b]
                play_id = batch["play_ids"][b]
                H = int(batch["hor_lengths"][b].item())
                player_ids = batch["player_ids"][b].cpu().numpy()
                target_flags = batch["static_cont"][b, :, 3].cpu().numpy()
                for j, nfl_id in enumerate(player_ids):
                    if nfl_id < 0 or target_flags[j] < 0.5:
                        continue
                    for h in range(H):
                        rows.append({
                            "game_id": int(game_id),
                            "play_id": int(play_id),
                            "nfl_id": int(nfl_id),
                            "frame_id": int(h + 1),
                            "x": float(pred[b, h, j, 0]),
                            "y": float(pred[b, h, j, 1]),
                        })

    pred_df = pd.DataFrame(rows).sort_values(["game_id", "play_id", "nfl_id", "frame_id"]).reset_index(drop=True)

    if template_path is not None and os.path.exists(template_path):
        template = pd.read_csv(template_path, usecols=TEMPLATE_REQUIRED_COLS)
        merge_cols = ["game_id", "play_id", "nfl_id", "frame_id"]
        full_df = template.merge(pred_df, on=merge_cols, how="left")
        full_df.to_csv(full_output_path, index=False)
        if "id" in full_df.columns:
            submission = full_df[["id", "x", "y"]].copy()
            submission.to_csv(output_path, index=False)
            return submission
        return full_df

    pred_df.to_csv(full_output_path, index=False)
    return pred_df


def validate_on_file(test_input_path: str, truth_template_path: str, artifact_dir: str):
    pred = predict_from_test_input(
        test_input_path=test_input_path,
        artifact_dir=artifact_dir,
        template_path=truth_template_path,
        output_path="tmp_submission.csv",
        full_output_path="tmp_predictions_full.csv",
    )
    print("Prediction file saved.")
    return pred
