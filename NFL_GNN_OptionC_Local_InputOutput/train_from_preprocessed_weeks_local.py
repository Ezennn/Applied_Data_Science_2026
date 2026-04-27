import contextlib
import gc
import glob
import importlib.util
import json
import math
import os
import pickle
import random
import re
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BASE_FILENAME = "nfl_trajectory_gnn_pipeline.py"


def _resolve_base_path() -> str:
    candidates = []
    env_path = os.environ.get("NFL_BDB_BASE_PATH")
    if env_path:
        candidates.append(env_path)

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, BASE_FILENAME))
        candidates.extend(sorted(glob.glob(os.path.join(here, "nfl_trajectory_gnn_pipeline*.py"))))
    except Exception:
        pass

    candidates.extend([
        os.path.join(os.getcwd(), BASE_FILENAME),
        *sorted(glob.glob(os.path.join(os.getcwd(), "nfl_trajectory_gnn_pipeline*.py"))),
        f"/content/{BASE_FILENAME}",
        *sorted(glob.glob("/content/nfl_trajectory_gnn_pipeline*.py")),
        f"/mnt/data/{BASE_FILENAME}",
        *sorted(glob.glob("/mnt/data/nfl_trajectory_gnn_pipeline*.py")),
    ])

    seen = set()
    for path in candidates:
        if not path:
            continue
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find {BASE_FILENAME}. Put it in the same folder as this script, "
        f"or set NFL_BDB_BASE_PATH to the full file path."
    )


BASE_PATH = _resolve_base_path()
spec = importlib.util.spec_from_file_location("nfl_bdb_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def _set_runtime_speed_flags() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


_set_runtime_speed_flags()


def _load_group_df(df: pd.DataFrame, group_indices, key: Tuple[int, int]) -> pd.DataFrame:
    part = df.loc[group_indices[key]]
    return part.sort_values(["frame_id", "nfl_id"]).reset_index(drop=True)


def build_play_sample(
    input_df: pd.DataFrame,
    input_groups,
    output_df: Optional[pd.DataFrame],
    output_groups,
    key: Tuple[int, int],
    pos_vocab,
    side_vocab,
    role_vocab,
) -> Dict:
    inp = _load_group_df(input_df, input_groups, key)

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
    frame_values = sorted(inp["frame_id"].unique().tolist())
    frame_to_t = {fid: t for t, fid in enumerate(frame_values)}

    T = len(frame_values)
    N = len(player_ids)
    H = int(inp["num_frames_output"].max()) if "num_frames_output" in inp.columns else 1

    feature_names = [
        "x", "y",
        "vx_own", "vy_own",
        "ax_own", "ay_own",
        "a_tan_own", "a_lat_own",
        "speed_own",
        "s_official",
        "a_official",
        "a_residual",
        "dir_rad", "o_rad", "gap_rad",
        "turn_rate_own",
        "obs_mask",
    ]
    arrs = {name: torch.zeros((T, N), dtype=torch.float32) for name in feature_names}

    for row in inp.itertuples(index=False):
        t = frame_to_t[int(row.frame_id)]
        j = player_to_idx[row.nfl_id]
        arrs["x"][t, j] = float(row.x)
        arrs["y"][t, j] = float(row.y)
        arrs["vx_own"][t, j] = float(row.vx_own)
        arrs["vy_own"][t, j] = float(row.vy_own)
        arrs["ax_own"][t, j] = float(row.ax_own)
        arrs["ay_own"][t, j] = float(row.ay_own)
        arrs["a_tan_own"][t, j] = float(row.a_tan_own)
        arrs["a_lat_own"][t, j] = float(row.a_lat_own)
        arrs["speed_own"][t, j] = float(row.speed_own)
        arrs["s_official"][t, j] = float(row.s_official)
        arrs["a_official"][t, j] = float(row.a_official)
        arrs["a_residual"][t, j] = float(row.a_residual)
        arrs["dir_rad"][t, j] = math.radians(float(row.dir))
        arrs["o_rad"][t, j] = math.radians(float(row.o))
        arrs["gap_rad"][t, j] = math.radians(float(row.body_motion_gap_deg))
        arrs["turn_rate_own"][t, j] = float(row.turn_rate_own)
        arrs["obs_mask"][t, j] = 1.0

    obs_mask = arrs["obs_mask"]
    fill_names = [
        "x", "y", "vx_own", "vy_own", "ax_own", "ay_own",
        "a_tan_own", "a_lat_own", "speed_own",
        "s_official", "a_official", "a_residual",
        "dir_rad", "o_rad", "gap_rad", "turn_rate_own",
    ]
    time_index = torch.arange(T, dtype=torch.long)
    zeros_idx = torch.zeros(T, dtype=torch.long)
    for name in fill_names:
        arr = arrs[name]
        for j in range(N):
            col_mask = obs_mask[:, j] > 0
            if not torch.any(col_mask):
                continue
            idx = torch.where(col_mask, time_index, zeros_idx)
            idx = torch.cummax(idx, dim=0).values
            arr[:, j] = arr[idx, j]

    static_cont = torch.stack([
        torch.tensor(players_meta["height_inches"].astype("float32").to_numpy()),
        torch.tensor(players_meta["player_weight"].astype("float32").to_numpy()),
        torch.tensor(players_meta["age_years"].astype("float32").to_numpy()),
        torch.tensor(players_meta["is_target"].astype("float32").to_numpy()),
    ], dim=-1)

    pos_ids = torch.tensor([pos_vocab.encode(v) for v in players_meta["player_position"]], dtype=torch.long)
    side_ids = torch.tensor([side_vocab.encode(v) for v in players_meta["player_side"]], dtype=torch.long)
    role_ids = torch.tensor([role_vocab.encode(v) for v in players_meta["player_role"]], dtype=torch.long)

    ball_land_x = float(inp["ball_land_x"].iloc[0]) if "ball_land_x" in inp.columns else 60.0
    ball_land_y = float(inp["ball_land_y"].iloc[0]) if "ball_land_y" in inp.columns else base.FIELD_Y_MAX / 2.0
    absolute_yardline_number = float(inp["absolute_yardline_number"].iloc[0]) if "absolute_yardline_number" in inp.columns else 60.0

    obs_feats = torch.stack([
        arrs["x"], arrs["y"],
        arrs["vx_own"], arrs["vy_own"],
        arrs["ax_own"], arrs["ay_own"],
        arrs["a_tan_own"], arrs["a_lat_own"],
        arrs["speed_own"],
        arrs["s_official"],
        arrs["a_official"],
        arrs["a_residual"],
        torch.sin(arrs["dir_rad"]), torch.cos(arrs["dir_rad"]),
        torch.sin(arrs["o_rad"]), torch.cos(arrs["o_rad"]),
        torch.sin(arrs["gap_rad"]), torch.cos(arrs["gap_rad"]),
        arrs["turn_rate_own"],
        torch.full_like(arrs["x"], ball_land_x),
        torch.full_like(arrs["x"], ball_land_y),
        torch.full_like(arrs["x"], absolute_yardline_number),
        torch.full_like(arrs["x"], float(H)),
        obs_mask,
    ], dim=-1).contiguous()

    target_xy = torch.full((H, N, 2), float("nan"), dtype=torch.float32)
    target_mask = torch.zeros((H, N), dtype=torch.float32)
    if output_df is not None and key in output_groups:
        out = _load_group_df(output_df, output_groups, key)
        for row in out.itertuples(index=False):
            j = player_to_idx.get(row.nfl_id)
            if j is None:
                continue
            h = int(row.frame_id) - 1
            if 0 <= h < H:
                target_xy[h, j, 0] = float(row.x)
                target_xy[h, j, 1] = float(row.y)
                target_mask[h, j] = 1.0

    return {
        "key": key,
        "game_id": int(key[0]),
        "play_id": int(key[1]),
        "player_ids": torch.tensor(player_ids, dtype=torch.long),
        "obs_feats": obs_feats,
        "obs_mask": obs_mask.contiguous(),
        "static_cont": static_cont.contiguous(),
        "pos_ids": pos_ids,
        "side_ids": side_ids,
        "role_ids": role_ids,
        "target_xy": target_xy,
        "target_mask": target_mask,
        "num_frames_output": int(H),
        "ball_land_xy": torch.tensor([ball_land_x, ball_land_y], dtype=torch.float32),
    }


class InMemoryPlayDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _extract_preprocessed_week_key(path: str, role: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    m = re.search(r"(\d{4}_w\d{2})", stem)
    if m:
        return m.group(1)
    if role == "test":
        return "test"
    return "all"


def discover_preprocessed_train_pairs(preprocessed_dir: str) -> List[Tuple[str, str, str]]:
    preprocessed_dir = os.path.abspath(preprocessed_dir)
    input_paths = [
        p for p in sorted(glob.glob(os.path.join(preprocessed_dir, "*.csv")))
        if os.path.isfile(p) and "train_input" in os.path.basename(p).lower() and "test" not in os.path.basename(p).lower()
    ]
    output_paths = [
        p for p in sorted(glob.glob(os.path.join(preprocessed_dir, "*.csv")))
        if os.path.isfile(p) and "train_output" in os.path.basename(p).lower() and "test" not in os.path.basename(p).lower()
    ]

    if not input_paths or not output_paths:
        raise FileNotFoundError(
            f"Could not find preprocessed weekly train_input/train_output CSV files under: {preprocessed_dir}"
        )

    input_map = {_extract_preprocessed_week_key(p, "input"): p for p in input_paths}
    output_map = {_extract_preprocessed_week_key(p, "output"): p for p in output_paths}
    common_keys = sorted(set(input_map) & set(output_map))
    missing_inputs = sorted(set(output_map) - set(input_map))
    missing_outputs = sorted(set(input_map) - set(output_map))
    if missing_inputs or missing_outputs:
        raise FileNotFoundError(
            "Found unmatched preprocessed weekly CSV files. "
            f"missing_inputs={missing_inputs}, missing_outputs={missing_outputs}"
        )
    return [(k, input_map[k], output_map[k]) for k in common_keys]


def _week_num_from_key(week_key: str) -> Optional[int]:
    m = re.search(r"_w(\d{2})", str(week_key).lower())
    return int(m.group(1)) if m else None


def _normalize_week_selection(weeks: Optional[Iterable]) -> Optional[set]:
    if weeks is None:
        return None
    out = set()
    for w in weeks:
        if w is None:
            continue
        if isinstance(w, int):
            out.add(int(w))
            continue
        s = str(w).strip().lower()
        if not s:
            continue
        if s.isdigit():
            out.add(int(s))
            continue
        m = re.search(r"w?(\d{1,2})$", s)
        if m:
            out.add(int(m.group(1)))
            continue
        m = re.search(r"_w(\d{2})", s)
        if m:
            out.add(int(m.group(1)))
            continue
        raise ValueError(f"Could not parse week selector: {w}")
    return out


def _pair_in_week_selection(pair: Tuple[str, str, str], week_selection: Optional[set]) -> bool:
    if week_selection is None:
        return True
    week_num = _week_num_from_key(pair[0])
    return week_num in week_selection


def _filter_pairs_by_weeks(train_pairs: Sequence[Tuple[str, str, str]], week_selection: Optional[Iterable]) -> List[Tuple[str, str, str]]:
    selection = _normalize_week_selection(week_selection)
    return [pair for pair in train_pairs if _pair_in_week_selection(pair, selection)]


def _split_sample_ids(sample_ids: Sequence[Tuple[str, int, int]], val_frac: float, seed: int) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]]]:
    sample_ids = list(sample_ids)
    rng = random.Random(seed)
    rng.shuffle(sample_ids)

    if not sample_ids:
        return [], []

    n_val = int(round(len(sample_ids) * float(val_frac)))
    if val_frac > 0.0 and len(sample_ids) > 1:
        n_val = max(1, min(len(sample_ids) - 1, n_val))
    else:
        n_val = 0

    val_ids = sample_ids[:n_val]
    train_ids = sample_ids[n_val:]
    return train_ids, val_ids


def _build_combined_vocabs(train_pairs: Sequence[Tuple[str, str, str]]):
    pos_vals = set()
    side_vals = set()
    role_vals = set()
    for week_key, input_csv, _ in train_pairs:
        print(f"Scanning vocab columns from {week_key}: {input_csv}")
        df = pd.read_csv(input_csv, usecols=["player_position", "player_side", "player_role"])
        pos_vals.update(df["player_position"].astype(str).tolist())
        side_vals.update(df["player_side"].astype(str).tolist())
        role_vals.update(df["player_role"].astype(str).tolist())
        del df
        gc.collect()
    return (
        base.CategoryVocab(list(pos_vals)),
        base.CategoryVocab(list(side_vals)),
        base.CategoryVocab(list(role_vals)),
    )


def _build_samples_for_pairs(
    train_pairs: Sequence[Tuple[str, str, str]],
    pos_vocab,
    side_vocab,
    role_vocab,
    split_name: str,
    max_plays: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    samples: List[Dict] = []
    pair_summaries: List[Dict] = []

    for idx, (week_key, input_csv, output_csv) in enumerate(train_pairs, start=1):
        print(f"[{idx}/{len(train_pairs)}] Loading preprocessed weekly files for {week_key} ({split_name})")
        input_df = pd.read_csv(input_csv)
        output_df = pd.read_csv(output_csv)

        input_groups = input_df.groupby(["game_id", "play_id"], sort=False).indices
        output_groups = output_df.groupby(["game_id", "play_id"], sort=False).indices
        local_keys = sorted((int(game_id), int(play_id)) for game_id, play_id in input_groups.keys())
        if max_plays is not None:
            local_keys = local_keys[:max(0, int(max_plays) - len(samples))]

        total = len(local_keys)
        for j, key in enumerate(local_keys, start=1):
            if total and (j == 1 or j % 500 == 0 or j == total):
                print(f"Building {split_name} samples for {week_key}: {j}/{total}")
            sample = build_play_sample(
                input_df=input_df,
                input_groups=input_groups,
                output_df=output_df,
                output_groups=output_groups,
                key=key,
                pos_vocab=pos_vocab,
                side_vocab=side_vocab,
                role_vocab=role_vocab,
            )
            samples.append(sample)
            if max_plays is not None and len(samples) >= max_plays:
                break

        pair_summaries.append({
            "week_key": week_key,
            "train_input_csv": input_csv,
            "train_output_csv": output_csv,
            "num_local_plays": len(input_groups),
            f"num_{split_name}_plays": total,
        })

        del input_df, output_df
        gc.collect()

        if max_plays is not None and len(samples) >= max_plays:
            break

    return samples, pair_summaries


def build_samples_from_preprocessed_dir(
    preprocessed_dir: str,
    val_frac: float,
    seed: int,
    max_train_plays: Optional[int] = None,
    max_val_plays: Optional[int] = None,
    train_weeks: Optional[Iterable] = None,
    val_weeks: Optional[Iterable] = None,
) -> Tuple[List[Dict], List[Dict], Dict]:
    all_pairs = discover_preprocessed_train_pairs(preprocessed_dir)

    explicit_week_split = (train_weeks is not None) or (val_weeks is not None)
    if explicit_week_split:
        train_pairs = _filter_pairs_by_weeks(all_pairs, train_weeks)
        val_pairs = _filter_pairs_by_weeks(all_pairs, val_weeks)
        if not train_pairs:
            raise ValueError(f"No training weeks matched train_weeks={list(train_weeks) if train_weeks is not None else train_weeks}")
        if not val_pairs:
            raise ValueError(f"No validation weeks matched val_weeks={list(val_weeks) if val_weeks is not None else val_weeks}")

        pos_vocab, side_vocab, role_vocab = _build_combined_vocabs(train_pairs)
        train_samples, train_pair_summaries = _build_samples_for_pairs(
            train_pairs=train_pairs,
            pos_vocab=pos_vocab,
            side_vocab=side_vocab,
            role_vocab=role_vocab,
            split_name="train",
            max_plays=max_train_plays,
        )
        val_samples, val_pair_summaries = _build_samples_for_pairs(
            train_pairs=val_pairs,
            pos_vocab=pos_vocab,
            side_vocab=side_vocab,
            role_vocab=role_vocab,
            split_name="val",
            max_plays=max_val_plays,
        )

        meta = {
            "pos_vocab": pos_vocab.itos,
            "side_vocab": side_vocab.itos,
            "role_vocab": role_vocab.itos,
            "train_pairs": train_pair_summaries,
            "val_pairs": val_pair_summaries,
            "preprocessed_dir": os.path.abspath(preprocessed_dir),
            "obs_feature_names": list(base.OBS_FEATURE_NAMES),
            "train_ids": [(sample["key"][0], sample["key"][1]) for sample in train_samples],
            "val_ids": [(sample["key"][0], sample["key"][1]) for sample in val_samples],
            "train_weeks": sorted(_normalize_week_selection(train_weeks) or []),
            "val_weeks": sorted(_normalize_week_selection(val_weeks) or []),
            "split_mode": "explicit_week_holdout",
        }
        return train_samples, val_samples, meta

    train_pairs = all_pairs
    pos_vocab, side_vocab, role_vocab = _build_combined_vocabs(train_pairs)

    all_sample_ids: List[Tuple[str, int, int]] = []
    for week_key, input_csv, _ in train_pairs:
        key_df = pd.read_csv(input_csv, usecols=["game_id", "play_id"]).drop_duplicates()
        for row in key_df.itertuples(index=False):
            all_sample_ids.append((week_key, int(row.game_id), int(row.play_id)))
        del key_df
        gc.collect()

    train_ids, val_ids = _split_sample_ids(all_sample_ids, val_frac=val_frac, seed=seed)
    if max_train_plays is not None:
        train_ids = train_ids[:max_train_plays]
    if max_val_plays is not None:
        val_ids = val_ids[:max_val_plays]

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)

    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    pair_summaries: List[Dict] = []

    for idx, (week_key, input_csv, output_csv) in enumerate(train_pairs, start=1):
        print(f"[{idx}/{len(train_pairs)}] Loading preprocessed weekly files for {week_key}")
        input_df = pd.read_csv(input_csv)
        output_df = pd.read_csv(output_csv)

        input_groups = input_df.groupby(["game_id", "play_id"], sort=False).indices
        output_groups = output_df.groupby(["game_id", "play_id"], sort=False).indices
        local_keys = sorted((int(game_id), int(play_id)) for game_id, play_id in input_groups.keys())
        local_train_keys = [key for key in local_keys if (week_key, key[0], key[1]) in train_id_set]
        local_val_keys = [key for key in local_keys if (week_key, key[0], key[1]) in val_id_set]

        for split_name, keys, dest in (("train", local_train_keys, train_samples), ("val", local_val_keys, val_samples)):
            total = len(keys)
            for j, key in enumerate(keys, start=1):
                if total and (j == 1 or j % 500 == 0 or j == total):
                    print(f"Building {split_name} samples for {week_key}: {j}/{total}")
                sample = build_play_sample(
                    input_df=input_df,
                    input_groups=input_groups,
                    output_df=output_df,
                    output_groups=output_groups,
                    key=key,
                    pos_vocab=pos_vocab,
                    side_vocab=side_vocab,
                    role_vocab=role_vocab,
                )
                dest.append(sample)

        pair_summaries.append({
            "week_key": week_key,
            "train_input_csv": input_csv,
            "train_output_csv": output_csv,
            "num_local_plays": len(local_keys),
            "num_train_plays": len(local_train_keys),
            "num_val_plays": len(local_val_keys),
        })

        del input_df, output_df
        gc.collect()

    meta = {
        "pos_vocab": pos_vocab.itos,
        "side_vocab": side_vocab.itos,
        "role_vocab": role_vocab.itos,
        "train_pairs": pair_summaries,
        "preprocessed_dir": os.path.abspath(preprocessed_dir),
        "obs_feature_names": list(base.OBS_FEATURE_NAMES),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_weeks": None,
        "val_weeks": None,
        "split_mode": "random_play_split",
    }
    return train_samples, val_samples, meta


def build_samples_from_preprocessed_csv(
    train_input_csv: str,
    train_output_csv: Optional[str],
    val_frac: float,
    seed: int,
    max_train_plays: Optional[int] = None,
    max_val_plays: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict], Dict]:
    print(f"Loading preprocessed train_input from {train_input_csv}")
    train_input = pd.read_csv(train_input_csv)
    if train_output_csv is not None:
        print(f"Loading preprocessed train_output from {train_output_csv}")
        train_output = pd.read_csv(train_output_csv)
    else:
        train_output = None

    pos_vocab, side_vocab, role_vocab = base.build_vocabs(train_input)
    all_keys = sorted(train_input.groupby(["game_id", "play_id"]).size().index.tolist())
    train_keys, val_keys = base.split_play_keys(all_keys, val_frac, seed)
    if max_train_plays is not None:
        train_keys = train_keys[:max_train_plays]
    if max_val_plays is not None:
        val_keys = val_keys[:max_val_plays]

    input_groups = train_input.groupby(["game_id", "play_id"], sort=False).indices
    output_groups = train_output.groupby(["game_id", "play_id"], sort=False).indices if train_output is not None else {}

    def build_split(keys: List[Tuple[int, int]], split_name: str) -> List[Dict]:
        samples: List[Dict] = []
        total = len(keys)
        for idx, key in enumerate(keys, start=1):
            if idx == 1 or idx % 500 == 0 or idx == total:
                print(f"Building {split_name} samples: {idx}/{total}")
            sample = build_play_sample(
                input_df=train_input,
                input_groups=input_groups,
                output_df=train_output,
                output_groups=output_groups,
                key=key,
                pos_vocab=pos_vocab,
                side_vocab=side_vocab,
                role_vocab=role_vocab,
            )
            samples.append(sample)
        return samples

    train_samples = build_split(train_keys, "train")
    val_samples = build_split(val_keys, "val") if train_output is not None else []

    meta = {
        "pos_vocab": pos_vocab.itos,
        "side_vocab": side_vocab.itos,
        "role_vocab": role_vocab.itos,
        "train_keys": train_keys,
        "val_keys": val_keys,
        "train_input_csv": train_input_csv,
        "train_output_csv": train_output_csv,
        "obs_feature_names": list(base.OBS_FEATURE_NAMES),
    }

    del train_input, train_output
    gc.collect()
    return train_samples, val_samples, meta


def collate_cached_plays(batch: List[Dict]) -> Dict[str, torch.Tensor]:
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
    game_ids, play_ids = [], []

    for i, item in enumerate(batch):
        T, N, _ = item["obs_feats"].shape
        H = item["target_xy"].shape[0]
        obs_feats[i, :T, :N] = item["obs_feats"]
        obs_mask[i, :T, :N] = item["obs_mask"]
        static_cont[i, :N] = item["static_cont"]
        pos_ids[i, :N] = item["pos_ids"]
        side_ids[i, :N] = item["side_ids"]
        role_ids[i, :N] = item["role_ids"]
        node_mask[i, :N] = 1.0
        target_xy[i, :H, :N] = item["target_xy"]
        target_mask[i, :H, :N] = item["target_mask"]
        ball_land_xy[i] = item["ball_land_xy"]
        player_ids[i, :N] = item["player_ids"]
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


class TrajectoryGNNFast(base.TrajectoryGNN):
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

        pos = last_frame[..., [base.OBS_FEATURE_INDEX["x"], base.OBS_FEATURE_INDEX["y"]]]
        vel = last_frame[..., [base.OBS_FEATURE_INDEX["vx_own"], base.OBS_FEATURE_INDEX["vy_own"]]]
        acc_prev = last_frame[..., [base.OBS_FEATURE_INDEX["ax_own"], base.OBS_FEATURE_INDEX["ay_own"]]]
        official_a_aux = last_frame[..., base.OBS_FEATURE_INDEX["a_official"]:base.OBS_FEATURE_INDEX["a_official"] + 1]
        body_ang = torch.atan2(
            last_frame[..., base.OBS_FEATURE_INDEX["sin_o"]:base.OBS_FEATURE_INDEX["sin_o"] + 1],
            last_frame[..., base.OBS_FEATURE_INDEX["cos_o"]:base.OBS_FEATURE_INDEX["cos_o"] + 1] + 1e-6,
        )
        move_ang = torch.atan2(vel[..., 1:2], vel[..., 0:1] + 1e-6)
        target_flag = static_cont[..., 3]

        edge_raw = base.pairwise_edge_features(pos, vel, side_ids, role_ids, target_flag)
        edge_feat = self.obs_edge_proj(edge_raw)
        for layer in self.obs_graph:
            h, _ = layer(h, edge_feat, node_mask)

        preds = []
        overflows = []
        collect_attn = getattr(self.config, "attn_loss_weight", 0.0) > 0
        attn_seq = [] if collect_attn else None

        for _ in range(H):
            edge_raw = base.pairwise_edge_features(pos, vel, side_ids, role_ids, target_flag)
            edge_feat = self.dec_edge_proj(edge_raw)
            for layer in self.dec_graph:
                h, attn = layer(h, edge_feat, node_mask)
                if collect_attn:
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

        return {
            "pred_xy": torch.stack(preds, dim=1),
            "overflow": torch.stack(overflows, dim=1),
            "attn_seq": attn_seq if collect_attn else [],
        }


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _make_adamw(params, lr: float, weight_decay: float):
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
        except Exception:
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _kaggle_rmse_accumulator_from_batch(pred_xy: torch.Tensor, batch: Dict) -> Tuple[float, int]:
    target_xy = batch["target_xy"]
    target_mask = batch["target_mask"] > 0.5
    finite_mask = torch.isfinite(target_xy[..., 0]) & torch.isfinite(target_xy[..., 1])
    valid = target_mask & finite_mask
    if not torch.any(valid):
        return 0.0, 0
    sq = (pred_xy[..., 0] - target_xy[..., 0]).pow(2) + (pred_xy[..., 1] - target_xy[..., 1]).pow(2)
    return float(sq[valid].sum().item()), int(valid.sum().item())


def kaggle_rmse_from_sum_count(sum_sq: float, n_rows: int) -> float:
    if n_rows <= 0:
        return float("nan")
    return math.sqrt(sum_sq / (2.0 * float(n_rows)))


def kaggle_rmse_from_dataframes(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    key_cols = ["game_id", "play_id", "nfl_id", "frame_id"]
    pred_part = pred_df[key_cols + ["x", "y"]].rename(columns={"x": "x_pred", "y": "y_pred"})
    truth_part = truth_df[key_cols + ["x", "y"]].rename(columns={"x": "x_true", "y": "y_true"})
    merged = truth_part.merge(pred_part, on=key_cols, how="left")
    valid = merged[["x_true", "y_true", "x_pred", "y_pred"]].notna().all(axis=1)
    if not valid.any():
        return float("nan"), merged
    dx2 = (merged.loc[valid, "x_true"] - merged.loc[valid, "x_pred"]).pow(2)
    dy2 = (merged.loc[valid, "y_true"] - merged.loc[valid, "y_pred"]).pow(2)
    rmse = math.sqrt(float((dx2 + dy2).sum()) / (2.0 * int(valid.sum())))
    return rmse, merged


def evaluate_fast(model, loader, device, config, loss_controller, amp_enabled: bool):
    model.eval()
    loss_controller.eval()
    metric_list = []
    total_sum_sq = 0.0
    total_rows = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                out = model(batch)
                _, metrics = base.compute_loss(out, batch, config=config, loss_controller=loss_controller, update_controller=False)
            batch_sum_sq, batch_rows = _kaggle_rmse_accumulator_from_batch(out["pred_xy"].float(), batch)
            total_sum_sq += batch_sum_sq
            total_rows += batch_rows
            metric_list.append(metrics)
    aggregated = base.aggregate_metrics(metric_list)
    aggregated["kaggle_rmse"] = kaggle_rmse_from_sum_count(total_sum_sq, total_rows)
    aggregated["kaggle_rows"] = float(total_rows)
    return aggregated


def _save_checkpoint(
    artifact_dir: str,
    epoch: int,
    model,
    optimizer,
    scaler,
    loss_controller,
    best_val: float,
    best_epoch: int,
    history: List[Dict],
    config,
    bundle_payload: Dict,
) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    checkpoint_path = os.path.join(artifact_dir, "last_checkpoint.pt")
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "loss_controller_state": loss_controller.state_dict(),
        "best_val": best_val,
        "best_epoch": best_epoch,
        "history": history,
        "config": asdict(config),
        "bundle_payload": bundle_payload,
    }
    torch.save(ckpt, checkpoint_path)
    pd.DataFrame(history).to_csv(os.path.join(artifact_dir, "training_history_partial.csv"), index=False)
    with open(os.path.join(artifact_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)
    return checkpoint_path


def train_model_fast_resume(
    model,
    train_loader,
    val_loader,
    config,
    artifact_dir: str,
    bundle_payload: Dict,
    eval_every: int = 2,
    resume: bool = True,
):
    os.makedirs(artifact_dir, exist_ok=True)
    device = config.device
    model.to(device)
    loss_controller = base.LossWeightController(config).to(device)
    optimizer = _make_adamw(list(model.parameters()) + list(loss_controller.parameters()), config.lr, config.weight_decay)
    amp_enabled = torch.cuda.is_available() and str(device).startswith("cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    save_path = os.path.join(artifact_dir, "best_model.pt")
    checkpoint_path = os.path.join(artifact_dir, "last_checkpoint.pt")

    start_epoch = 0
    best_val = float("inf")
    best_epoch = -1
    history: List[Dict] = []

    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        loss_controller.load_state_dict(ckpt["loss_controller_state"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", -1))
        history = list(ckpt.get("history", []))
        print(f"Resume start epoch: {start_epoch + 1}")

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        loss_controller.train()
        train_metric_list = []

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                out = model(batch)
                loss, metrics = base.compute_loss(
                    out,
                    batch,
                    config=config,
                    loss_controller=loss_controller,
                    epoch_idx=epoch,
                    update_controller=True,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_metric_list.append(metrics)

        train_metrics = base.aggregate_metrics(train_metric_list)
        run_val = ((epoch + 1) % eval_every == 0) or (epoch == config.num_epochs - 1)
        val_metrics = (
            evaluate_fast(model, val_loader, device, config=config, loss_controller=loss_controller, amp_enabled=amp_enabled)
            if run_val and val_loader is not None and len(val_loader.dataset) > 0 else {}
        )
        val_loss = val_metrics.get("loss", float("inf"))
        monitor_name = "kaggle_rmse" if "kaggle_rmse" in val_metrics and not math.isnan(val_metrics.get("kaggle_rmse", float("nan"))) else "loss"
        val_score = val_metrics.get(monitor_name, float("inf"))

        history_row = {"epoch": epoch + 1, "val_monitor_name": monitor_name if run_val else None, "val_monitor_score": val_score if run_val else float("nan")}
        history_row.update({f"train_{k}": v for k, v in train_metrics.items()})
        history_row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(history_row)

        print(
            f"Epoch {epoch + 1:02d} | train_loss {train_metrics.get('loss', float('nan')):.5f} | "
            f"val_loss {val_loss if run_val else float('nan'):.5f} | "
            f"val_kaggle {val_metrics.get('kaggle_rmse', float('nan')) if run_val else float('nan'):.5f} | "
            f"track {train_metrics.get('track_loss_raw', float('nan')):.5f} | "
            f"vel {train_metrics.get('vel_loss_raw', float('nan')):.5f}"
        )

        if run_val and val_score < best_val - 1e-4:
            best_val = val_score
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        _save_checkpoint(
            artifact_dir=artifact_dir,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loss_controller=loss_controller,
            best_val=best_val,
            best_epoch=best_epoch,
            history=history,
            config=config,
            bundle_payload=bundle_payload,
        )

        if run_val and len(val_loader.dataset) > 0 and epoch + 1 >= config.min_epochs and epoch - best_epoch >= config.patience:
            print("Early stopping.")
            break

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    pd.DataFrame(history).to_csv(os.path.join(artifact_dir, "training_history.csv"), index=False)
    with open(os.path.join(artifact_dir, "bundle.pkl"), "wb") as f:
        pickle.dump(bundle_payload | {"config": asdict(config), "history": history}, f)
    return history, save_path


def make_colab_fast_config(device: Optional[str] = None):
    config = base.ModelConfig(
        d_model=32,
        edge_dim=8,
        static_dim=16,
        batch_size=4,
        num_epochs=6,
        min_epochs=2,
        patience=2,
        val_frac=0.08,
        num_workers=2,
        attn_loss_weight=0.0,
    )
    config.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return config


def _make_loaders(train_samples: List[Dict], val_samples: List[Dict], config):
    train_ds = InMemoryPlayDataset(train_samples)
    val_ds = InMemoryPlayDataset(val_samples)

    pin = torch.cuda.is_available() and str(config.device).startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_cached_plays,
        num_workers=config.num_workers,
        pin_memory=pin,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_cached_plays,
        num_workers=config.num_workers,
        pin_memory=pin,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader


def _vocabs_from_meta(meta: Dict):
    pos_vocab = base.CategoryVocab([])
    pos_vocab.itos = meta["pos_vocab"]
    pos_vocab.stoi = {v: i for i, v in enumerate(pos_vocab.itos)}

    side_vocab = base.CategoryVocab([])
    side_vocab.itos = meta["side_vocab"]
    side_vocab.stoi = {v: i for i, v in enumerate(side_vocab.itos)}

    role_vocab = base.CategoryVocab([])
    role_vocab.itos = meta["role_vocab"]
    role_vocab.stoi = {v: i for i, v in enumerate(role_vocab.itos)}
    return pos_vocab, side_vocab, role_vocab


def fit_from_preprocessed_dir(
    preprocessed_dir: str,
    artifact_dir: str = "./artifacts_preprocessed_csv_weekly",
    config=None,
    eval_every: int = 2,
    max_train_plays: Optional[int] = None,
    max_val_plays: Optional[int] = None,
    resume: bool = True,
    train_weeks: Optional[Iterable] = None,
    val_weeks: Optional[Iterable] = None,
):
    os.makedirs(artifact_dir, exist_ok=True)
    config = config or make_colab_fast_config()
    base.seed_everything(config.seed)

    train_samples, val_samples, meta = build_samples_from_preprocessed_dir(
        preprocessed_dir=preprocessed_dir,
        val_frac=config.val_frac,
        seed=config.seed,
        max_train_plays=max_train_plays,
        max_val_plays=max_val_plays,
        train_weeks=train_weeks,
        val_weeks=val_weeks,
    )

    train_loader, val_loader = _make_loaders(train_samples, val_samples, config)
    pos_vocab, side_vocab, role_vocab = _vocabs_from_meta(meta)

    model = TrajectoryGNNFast(config, len(pos_vocab), len(side_vocab), len(role_vocab))
    bundle_payload = {
        "pos_vocab": meta["pos_vocab"],
        "side_vocab": meta["side_vocab"],
        "role_vocab": meta["role_vocab"],
        "obs_feature_names": meta["obs_feature_names"],
        "preprocessed_dir": meta["preprocessed_dir"],
        "train_pairs": meta["train_pairs"],
        "val_pairs": meta.get("val_pairs", []),
        "train_ids": meta["train_ids"],
        "val_ids": meta["val_ids"],
        "train_weeks": meta.get("train_weeks"),
        "val_weeks": meta.get("val_weeks"),
        "split_mode": meta.get("split_mode"),
        "mode": "weekly_preprocessed_csvs",
    }

    history, save_path = train_model_fast_resume(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        artifact_dir=artifact_dir,
        bundle_payload=bundle_payload,
        eval_every=eval_every,
        resume=resume,
    )
    return history, save_path


def fit_from_preprocessed_csv(
    train_input_csv: str,
    train_output_csv: str,
    artifact_dir: str = "./artifacts_preprocessed_csv",
    config=None,
    eval_every: int = 2,
    max_train_plays: Optional[int] = None,
    max_val_plays: Optional[int] = None,
    resume: bool = True,
):
    os.makedirs(artifact_dir, exist_ok=True)
    config = config or make_colab_fast_config()
    base.seed_everything(config.seed)

    train_samples, val_samples, meta = build_samples_from_preprocessed_csv(
        train_input_csv=train_input_csv,
        train_output_csv=train_output_csv,
        val_frac=config.val_frac,
        seed=config.seed,
        max_train_plays=max_train_plays,
        max_val_plays=max_val_plays,
    )

    train_loader, val_loader = _make_loaders(train_samples, val_samples, config)
    pos_vocab, side_vocab, role_vocab = _vocabs_from_meta(meta)

    model = TrajectoryGNNFast(config, len(pos_vocab), len(side_vocab), len(role_vocab))
    bundle_payload = {
        "pos_vocab": meta["pos_vocab"],
        "side_vocab": meta["side_vocab"],
        "role_vocab": meta["role_vocab"],
        "obs_feature_names": meta["obs_feature_names"],
        "train_input_csv": train_input_csv,
        "train_output_csv": train_output_csv,
        "train_keys": meta["train_keys"],
        "val_keys": meta["val_keys"],
        "mode": "single_preprocessed_csv_pair",
    }

    history, save_path = train_model_fast_resume(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        artifact_dir=artifact_dir,
        bundle_payload=bundle_payload,
        eval_every=eval_every,
        resume=resume,
    )
    return history, save_path


def load_bundle(artifact_dir: str):
    with open(os.path.join(artifact_dir, "bundle.pkl"), "rb") as f:
        bundle = pickle.load(f)

    config = base.ModelConfig(**bundle["config"])
    pos_vocab = base.CategoryVocab([])
    pos_vocab.itos = bundle["pos_vocab"]
    pos_vocab.stoi = {v: i for i, v in enumerate(pos_vocab.itos)}

    side_vocab = base.CategoryVocab([])
    side_vocab.itos = bundle["side_vocab"]
    side_vocab.stoi = {v: i for i, v in enumerate(side_vocab.itos)}

    role_vocab = base.CategoryVocab([])
    role_vocab.itos = bundle["role_vocab"]
    role_vocab.stoi = {v: i for i, v in enumerate(role_vocab.itos)}

    model = TrajectoryGNNFast(config, len(pos_vocab), len(side_vocab), len(role_vocab))
    model.load_state_dict(torch.load(os.path.join(artifact_dir, "best_model.pt"), map_location=config.device))
    model.to(config.device)
    model.eval()
    return model, config, pos_vocab, side_vocab, role_vocab, bundle


def predict_from_preprocessed_input_csv(
    input_csv: str,
    artifact_dir: str,
    output_path: str = "submission.csv",
    full_output_path: str = "predictions_full.csv",
    template_path: Optional[str] = None,
    truth_output_csv: Optional[str] = None,
    evaluation_output_path: Optional[str] = None,
):
    model, config, pos_vocab, side_vocab, role_vocab, bundle = load_bundle(artifact_dir)
    input_df = pd.read_csv(input_csv)

    input_groups = input_df.groupby(["game_id", "play_id"], sort=False).indices
    keys = list(input_groups.keys())
    samples = [
        build_play_sample(
            input_df=input_df,
            input_groups=input_groups,
            output_df=None,
            output_groups={},
            key=key,
            pos_vocab=pos_vocab,
            side_vocab=side_vocab,
            role_vocab=role_vocab,
        )
        for key in keys
    ]

    ds = InMemoryPlayDataset(samples)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_cached_plays, num_workers=0)

    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, config.device)
            out = model(batch)
            pred = out["pred_xy"].cpu().numpy()
            B = pred.shape[0]
            for i in range(B):
                game_id = batch["game_ids"][i]
                play_id = batch["play_ids"][i]
                N = int((batch["player_ids"][i] >= 0).sum().item())
                H = int(batch["hor_lengths"][i].item())
                for j in range(N):
                    nfl_id = int(batch["player_ids"][i, j].item())
                    for h in range(H):
                        rows.append({
                            "game_id": int(game_id),
                            "play_id": int(play_id),
                            "nfl_id": nfl_id,
                            "frame_id": h + 1,
                            "x": float(pred[i, h, j, 0]),
                            "y": float(pred[i, h, j, 1]),
                        })

    full_df = pd.DataFrame(rows).sort_values(["game_id", "play_id", "nfl_id", "frame_id"]).reset_index(drop=True)
    full_df.to_csv(full_output_path, index=False)

    submission_or_full = full_df
    if template_path is not None:
        template = pd.read_csv(template_path)
        required = ["game_id", "play_id", "nfl_id", "frame_id"]
        submission_or_full = template.merge(full_df, on=required, how="left")
        submission_or_full.to_csv(output_path, index=False)

    eval_result = None
    if truth_output_csv is not None:
        truth_df = pd.read_csv(truth_output_csv)
        rmse, merged_eval = kaggle_rmse_from_dataframes(full_df, truth_df)
        eval_result = {
            "kaggle_rmse": rmse,
            "n_rows": int(merged_eval[["x_true", "y_true", "x_pred", "y_pred"]].notna().all(axis=1).sum()),
        }
        if evaluation_output_path is not None:
            merged_eval.to_csv(evaluation_output_path, index=False)

    return submission_or_full, full_df, bundle, eval_result


def predict_from_preprocessed_test_csv(
    test_input_csv: str,
    artifact_dir: str,
    output_path: str = "submission.csv",
    full_output_path: str = "predictions_full.csv",
    template_path: Optional[str] = None,
):
    return predict_from_preprocessed_input_csv(
        input_csv=test_input_csv,
        artifact_dir=artifact_dir,
        output_path=output_path,
        full_output_path=full_output_path,
        template_path=template_path,
        truth_output_csv=None,
        evaluation_output_path=None,
    )


def evaluate_holdout_weeks(
    preprocessed_dir: str,
    artifact_dir: str,
    eval_weeks: Iterable,
    output_dir: str,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    eval_pairs = _filter_pairs_by_weeks(discover_preprocessed_train_pairs(preprocessed_dir), eval_weeks)
    if not eval_pairs:
        raise ValueError(f"No evaluation weeks matched eval_weeks={list(eval_weeks)}")

    summary_rows = []
    total_sum_sq = 0.0
    total_rows = 0

    for week_key, input_csv, truth_output_csv in eval_pairs:
        pred_path = os.path.join(output_dir, f"predictions_{week_key}.csv")
        eval_path = os.path.join(output_dir, f"evaluation_{week_key}.csv")
        _, full_df, _, eval_result = predict_from_preprocessed_input_csv(
            input_csv=input_csv,
            artifact_dir=artifact_dir,
            full_output_path=pred_path,
            truth_output_csv=truth_output_csv,
            evaluation_output_path=eval_path,
        )
        if eval_result is None:
            raise RuntimeError(f"Evaluation failed for {week_key}")
        merged_truth = pd.read_csv(truth_output_csv).rename(columns={"x": "x_true", "y": "y_true"})
        merged_pred = full_df.rename(columns={"x": "x_pred", "y": "y_pred"})
        merged = merged_truth.merge(merged_pred, on=["game_id", "play_id", "nfl_id", "frame_id"], how="left")
        valid = merged[["x_true", "y_true", "x_pred", "y_pred"]].notna().all(axis=1)
        sq_sum = float(((merged.loc[valid, "x_true"] - merged.loc[valid, "x_pred"]).pow(2) + (merged.loc[valid, "y_true"] - merged.loc[valid, "y_pred"]).pow(2)).sum())
        n_rows = int(valid.sum())
        total_sum_sq += sq_sum
        total_rows += n_rows
        summary_rows.append({
            "week_key": week_key,
            "input_csv": input_csv,
            "truth_output_csv": truth_output_csv,
            "prediction_csv": pred_path,
            "evaluation_csv": eval_path,
            "kaggle_rmse": eval_result["kaggle_rmse"],
            "n_rows": n_rows,
        })

    overall_rmse = kaggle_rmse_from_sum_count(total_sum_sq, total_rows)
    summary = {
        "preprocessed_dir": os.path.abspath(preprocessed_dir),
        "artifact_dir": os.path.abspath(artifact_dir),
        "eval_weeks": sorted(_normalize_week_selection(eval_weeks) or []),
        "overall_kaggle_rmse": overall_rmse,
        "overall_n_rows": total_rows,
        "weeks": summary_rows,
    }
    with open(os.path.join(output_dir, "holdout_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "holdout_summary.csv"), index=False)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the NFL GNN model from preprocessed CSV files.")
    parser.add_argument("--preprocessed_dir", default=None, help="Directory containing weekly preprocessed CSV files.")
    parser.add_argument("--train_input_csv", default=None)
    parser.add_argument("--train_output_csv", default=None)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train_weeks", default=None, help="Comma-separated week numbers, for example: 1,2,3,4,5,6,7,8,9,10,11,12,13")
    parser.add_argument("--val_weeks", default=None, help="Comma-separated week numbers, for example: 14,15,16,17,18")
    parser.add_argument("--holdout_output_dir", default=None, help="If set together with --preprocessed_dir and --val_weeks, save holdout predictions and Kaggle-style RMSE here.")
    args = parser.parse_args()

    def _parse_cli_weeks(text):
        if text is None:
            return None
        return [part.strip() for part in str(text).split(",") if part.strip()]

    config = make_colab_fast_config()
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.num_epochs = args.num_epochs

    if args.preprocessed_dir:
        fit_from_preprocessed_dir(
            preprocessed_dir=args.preprocessed_dir,
            artifact_dir=args.artifact_dir,
            config=config,
            eval_every=args.eval_every,
            resume=args.resume,
            train_weeks=_parse_cli_weeks(args.train_weeks),
            val_weeks=_parse_cli_weeks(args.val_weeks),
        )
        if args.holdout_output_dir and args.val_weeks:
            summary = evaluate_holdout_weeks(
                preprocessed_dir=args.preprocessed_dir,
                artifact_dir=args.artifact_dir,
                eval_weeks=_parse_cli_weeks(args.val_weeks),
                output_dir=args.holdout_output_dir,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        if not args.train_input_csv or not args.train_output_csv:
            raise ValueError("Either provide --preprocessed_dir, or provide both --train_input_csv and --train_output_csv.")
        fit_from_preprocessed_csv(
            train_input_csv=args.train_input_csv,
            train_output_csv=args.train_output_csv,
            artifact_dir=args.artifact_dir,
            config=config,
            eval_every=args.eval_every,
            resume=args.resume,
        )
