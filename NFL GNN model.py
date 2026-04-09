from google.colab import files
import os, shutil, zipfile
import os
import math
import json
import glob
import random
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

uploaded = files.upload()  # 上传 kaggle.json
os.makedirs('/root/.kaggle', exist_ok=True)
shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 0o600)

!kaggle competitions download -c nfl-big-data-bowl-2026-prediction -p ./nfl_data_zip
os.makedirs('./nfl_data', exist_ok=True)
for zf in os.listdir('./nfl_data_zip'):
    if zf.endswith('.zip'):
        with zipfile.ZipFile(os.path.join('./nfl_data_zip', zf), 'r') as z:
            z.extractall('./nfl_data')
print("Downloaded files:", os.listdir('./nfl_data')[:20])

# GNN MODEL
FIELD_X_MIN = 0.0
FIELD_X_MAX = 120.0
FIELD_Y_MIN = 0.0
FIELD_Y_MAX = 53.3
DT = 0.1
EPS = 1e-6


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def add_derived_observed_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"]).copy()
    g = df.groupby(["game_id", "play_id", "nfl_id"], sort=False)

    df["dx_obs"] = g["x"].diff().fillna(0.0)
    df["dy_obs"] = g["y"].diff().fillna(0.0)
    df["vx_obs"] = df["dx_obs"] / DT
    df["vy_obs"] = df["dy_obs"] / DT
    df["ax_obs"] = g["vx_obs"].diff().fillna(0.0) / DT
    df["ay_obs"] = g["vy_obs"].diff().fillna(0.0) / DT
    df["speed_obs"] = np.sqrt(df["vx_obs"] ** 2 + df["vy_obs"] ** 2)
    df["turn_rate_obs"] = g["dir"].diff().fillna(0.0).clip(-180.0, 180.0) / DT
    df["body_motion_gap_deg"] = ((df["dir"] - df["o"] + 180.0) % 360.0) - 180.0
    df["height_inches"] = df["player_height"].map(parse_height_to_inches).astype(np.float32)
    df["player_weight"] = df["player_weight"].fillna(df["player_weight"].median() if "player_weight" in df.columns else 210)
    df["age_proxy_year"] = pd.to_datetime(df["player_birth_date"], errors="coerce").dt.year.fillna(1998).astype(np.float32)
    df["is_target"] = df["player_to_predict"].astype(np.float32) if "player_to_predict" in df.columns else 0.0
    return df


def discover_competition_files(data_dir: str) -> Dict[str, List[str]]:
    paths = {
        "train_input": sorted(glob.glob(os.path.join(data_dir, "train_input*.csv"))),
        "train_output": sorted(glob.glob(os.path.join(data_dir, "train_output*.csv"))),
        "test_input": sorted(glob.glob(os.path.join(data_dir, "test_input*.csv"))),
        "sample_submission": sorted(glob.glob(os.path.join(data_dir, "*submission*.csv"))) + sorted(glob.glob(os.path.join(data_dir, "test.csv"))),
    }
    return paths


def concat_csvs(paths: List[str]) -> Optional[pd.DataFrame]:
    if not paths:
        return None
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


@dataclass
class ModelConfig:
    d_model: int = 32
    edge_dim: int = 8
    static_dim: int = 16
    obs_feat_dim: int = 20
    num_graph_layers: int = 1
    dropout: float = 0.1
    friction_accel_limit: float = 6.5
    max_speed: float = 11.5
    max_yaw_rate: float = 6.0
    non_target_blend: float = 0.35
    lr: float = 2e-4
    batch_size: int = 4
    num_epochs: int = 14
    weight_decay: float = 1e-5
    val_frac: float = 0.15
    grad_clip: float = 1.0
    min_epochs: int = 4
    patience: int = 4
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
                "height_inches", "player_weight", "age_proxy_year", "is_target"
            ]]
            .drop_duplicates("nfl_id")
            .sort_values(["player_side", "nfl_id"])
            .reset_index(drop=True)
        )

        player_ids = players_meta["nfl_id"].tolist()
        player_to_idx = {pid: i for i, pid in enumerate(player_ids)}
        N = len(player_ids)
        T = int(inp["frame_id"].max())

        num_frames_output = int(inp["num_frames_output"].max()) if "num_frames_output" in inp.columns else 1
        H = num_frames_output

        x_seq = np.zeros((T, N), dtype=np.float32)
        y_seq = np.zeros((T, N), dtype=np.float32)
        s_seq = np.zeros((T, N), dtype=np.float32)
        a_seq = np.zeros((T, N), dtype=np.float32)
        dir_seq = np.zeros((T, N), dtype=np.float32)
        o_seq = np.zeros((T, N), dtype=np.float32)
        vx_seq = np.zeros((T, N), dtype=np.float32)
        vy_seq = np.zeros((T, N), dtype=np.float32)
        ax_seq = np.zeros((T, N), dtype=np.float32)
        ay_seq = np.zeros((T, N), dtype=np.float32)
        gap_seq = np.zeros((T, N), dtype=np.float32)
        turn_rate_seq = np.zeros((T, N), dtype=np.float32)
        obs_mask = np.zeros((T, N), dtype=np.float32)

        for row in inp.itertuples(index=False):
            t = int(row.frame_id) - 1
            j = player_to_idx[row.nfl_id]
            x_seq[t, j] = float(row.x)
            y_seq[t, j] = float(row.y)
            s_seq[t, j] = float(row.s)
            a_seq[t, j] = float(row.a)
            dir_seq[t, j] = math.radians(float(row.dir))
            o_seq[t, j] = math.radians(float(row.o))
            vx_seq[t, j] = float(row.vx_obs)
            vy_seq[t, j] = float(row.vy_obs)
            ax_seq[t, j] = float(row.ax_obs)
            ay_seq[t, j] = float(row.ay_obs)
            gap_seq[t, j] = math.radians(float(row.body_motion_gap_deg))
            turn_rate_seq[t, j] = math.radians(float(row.turn_rate_obs))
            obs_mask[t, j] = 1.0

        # forward fill missing values player-wise
        for arr in [x_seq, y_seq, s_seq, a_seq, dir_seq, o_seq, vx_seq, vy_seq, ax_seq, ay_seq, gap_seq, turn_rate_seq]:
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
            players_meta["age_proxy_year"].astype(np.float32).to_numpy(),
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
            s_seq, a_seq,
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
            "obs_feats": obs_feats,       # [T, N, F]
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
        # x: [B, N, D], edge_feat: [B, N, N, E], node_mask: [B, N]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn_logits = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(x.shape[-1])
        attn_logits = attn_logits + self.edge_bias(edge_feat).squeeze(-1)

        valid = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)  # [B, N, N]
        attn_logits = attn_logits.masked_fill(valid == 0, -1e9)
        attn = torch.softmax(attn_logits, dim=-1)

        edge_msg = self.edge_msg(edge_feat)
        msg = torch.einsum("bij,bjd->bid", attn, v) + torch.einsum("bij,bijd->bid", attn, edge_msg)

        out = self.norm(x + self.out(torch.cat([x, msg], dim=-1)))
        return out, attn