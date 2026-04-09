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
def pairwise_edge_features(pos, vel, side_ids, role_ids, target_flag):
    # pos: [B,N,2], vel: [B,N,2]
    rel_pos = pos.unsqueeze(2) - pos.unsqueeze(1)  # i - j
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
            nn.Linear(config.d_model + 10, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 3),
        )
        self.overflow_head = nn.Sequential(
            nn.Linear(config.d_model + 10, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(self, h, pos, vel, body_ang, move_ang, static_ctx, ball_ctx, target_flag):
        # h: [B,N,D], pos/vel [B,N,2], angles [B,N,1]
        gap = wrap_to_pi(move_ang - body_ang)
        decoder_in = torch.cat([
            h,
            pos,
            vel,
            torch.sin(body_ang), torch.cos(body_ang),
            torch.sin(move_ang), torch.cos(move_ang),
            torch.sin(gap), torch.cos(gap),
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
        acc = torch.cat([ax, ay], dim=-1)

        vel_next = vel + DT * acc
        speed_next = torch.linalg.norm(vel_next, dim=-1, keepdim=True).clamp_min(1e-6)
        speed_scale = torch.clamp(self.config.max_speed / speed_next, max=1.0)
        vel_next = vel_next * speed_scale

        pos_next = pos + DT * vel_next
        pos_next[..., 0] = pos_next[..., 0].clamp(FIELD_X_MIN, FIELD_X_MAX)
        pos_next[..., 1] = pos_next[..., 1].clamp(FIELD_Y_MIN, FIELD_Y_MAX)

        desired_move_ang = torch.atan2(vel_next[..., 1:2], vel_next[..., 0:1] + 1e-6)
        body_next = body_ang + DT * yaw_ctrl + 0.15 * wrap_to_pi(desired_move_ang - body_ang)
        move_next = desired_move_ang

        # Non-target nodes are softly pulled toward constant-velocity prior
        pos_cv = pos + DT * vel
        vel_cv = vel
        non_target = (1.0 - target_flag.unsqueeze(-1))
        alpha = self.config.non_target_blend
        pos_next = pos_next * (1.0 - non_target * alpha) + pos_cv * (non_target * alpha)
        vel_next = vel_next * (1.0 - non_target * alpha) + vel_cv * (non_target * alpha)

        return pos_next, vel_next, body_next, move_next, overflow.squeeze(-1)


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
        obs_feats = batch["obs_feats"]          # [B,T,N,F]
        obs_mask = batch["obs_mask"]            # [B,T,N]
        static_cont = batch["static_cont"]      # [B,N,4]
        pos_ids = batch["pos_ids"]
        side_ids = batch["side_ids"]
        role_ids = batch["role_ids"]
        node_mask = batch["node_mask"]
        hor_lengths = batch["hor_lengths"]

        B, T, N, Fdim = obs_feats.shape
        H = int(hor_lengths.max().item())

        static_ctx = self.encode_static(static_cont, pos_ids, side_ids, role_ids)  # [B,N,S]
        static_seq = static_ctx.unsqueeze(1).expand(B, T, N, static_ctx.shape[-1])
        obs_in = self.obs_proj(torch.cat([obs_feats, static_seq], dim=-1))          # [B,T,N,D]

        flat = obs_in.permute(0, 2, 1, 3).reshape(B * N, T, self.config.d_model)
        _, h_last = self.obs_gru(flat)
        h = h_last.squeeze(0).reshape(B, N, self.config.d_model)

        last_idx = (batch["obs_lengths"] - 1).view(B, 1, 1, 1).expand(B, 1, N, Fdim)
        last_frame = torch.gather(obs_feats, 1, last_idx).squeeze(1)  # [B,N,F]
        pos = last_frame[..., 0:2]
        vel = last_frame[..., 2:4]
        body_ang = torch.atan2(last_frame[..., 10:11], last_frame[..., 11:12] + 1e-6)
        move_ang = torch.atan2(vel[..., 1:2], vel[..., 0:1] + 1e-6)
        target_flag = static_cont[..., 3]

        # One social-context graph update at the last observed frame
        edge_raw = pairwise_edge_features(pos, vel, side_ids, role_ids, target_flag)
        edge_feat = self.obs_edge_proj(edge_raw)
        for layer in self.obs_graph:
            h, _ = layer(h, edge_feat, node_mask)

        preds = []
        overflows = []
        attn_seq = []
        for step in range(H):
            edge_raw = pairwise_edge_features(pos, vel, side_ids, role_ids, target_flag)
            edge_feat = self.dec_edge_proj(edge_raw)
            for layer in self.dec_graph:
                h, attn = layer(h, edge_feat, node_mask)
                attn_seq.append(attn)
            pos, vel, body_ang, move_ang, overflow = self.decoder(
                h=h,
                pos=pos,
                vel=vel,
                body_ang=body_ang,
                move_ang=move_ang,
                static_ctx=static_ctx,
                ball_ctx=batch["ball_land_xy"],
                target_flag=target_flag,
            )
            preds.append(pos)
            overflows.append(overflow)

        pred_xy = torch.stack(preds, dim=1)          # [B,H,N,2]
        overflow = torch.stack(overflows, dim=1)     # [B,H,N]
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


def compute_loss(model_out, batch):
    pred_xy = model_out["pred_xy"]
    target_xy = batch["target_xy"]
    target_mask = batch["target_mask"]

    valid_mask = target_mask.unsqueeze(-1).expand_as(pred_xy)
    pos_loss = masked_smooth_l1(pred_xy, target_xy, valid_mask)

    # Future step smoothness
    if pred_xy.shape[1] > 1:
        pred_d = pred_xy[:, 1:] - pred_xy[:, :-1]
        step_loss = pred_d.pow(2).mean()
    else:
        step_loss = pred_xy.sum() * 0.0

    # Endpoint emphasis on last available target frame
    endpoint_losses = []
    for b in range(pred_xy.shape[0]):
        h = int(batch["hor_lengths"][b].item())
        if h > 0:
            pm = target_mask[b, h - 1] > 0
            if pm.any():
                endpoint_losses.append(F.smooth_l1_loss(pred_xy[b, h - 1, pm], target_xy[b, h - 1, pm]))
    endpoint_loss = torch.stack(endpoint_losses).mean() if endpoint_losses else pred_xy.sum() * 0.0

    friction_penalty = model_out["overflow"].mean()

    # Attention smoothness
    attn_seq = model_out["attn_seq"]
    if len(attn_seq) > 1:
        diffs = [(attn_seq[i] - attn_seq[i - 1]).abs().mean() for i in range(1, len(attn_seq))]
        attn_smooth = torch.stack(diffs).mean()
    else:
        attn_smooth = pred_xy.sum() * 0.0

    total = pos_loss + 0.10 * step_loss + 0.35 * endpoint_loss + 0.05 * friction_penalty + 0.01 * attn_smooth
    metrics = {
        "loss": total.detach().item(),
        "pos_loss": pos_loss.detach().item(),
        "endpoint_loss": endpoint_loss.detach().item(),
        "friction_penalty": friction_penalty.detach().item(),
    }
    return total, metrics


def split_play_keys(keys: List[Tuple[int, int]], val_frac: float = 0.15, seed: int = 42):
    rng = random.Random(seed)
    keys = list(keys)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac))
    val_keys = keys[:n_val]
    train_keys = keys[n_val:]
    return train_keys, val_keys


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def evaluate(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            loss, metrics = compute_loss(out, batch)
            losses.append(metrics["loss"])
    return float(np.mean(losses)) if losses else np.nan


def train_model(model, train_loader, val_loader, config: ModelConfig, save_path: str):
    device = config.device
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_val = float("inf")
    best_epoch = -1

    history = []
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss, metrics = compute_loss(out, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_losses.append(metrics["loss"])

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch+1:02d} | train {train_loss:.5f} | val {val_loss:.5f}")

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
    df = canonicalize_play_direction(df)
    df = add_derived_observed_features(df)
    return df


def preprocess_output_df(df: pd.DataFrame, input_ref_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    out = df.copy()
    # Output uses relative frame ids and x,y only. Canonicalization for x/y must match input play direction.
    if input_ref_df is not None and "play_direction" in input_ref_df.columns:
        play_dir = (
            input_ref_df[["game_id", "play_id", "play_direction"]]
            .drop_duplicates(["game_id", "play_id"])
        )
        out = out.merge(play_dir, on=["game_id", "play_id"], how="left")
        out = canonicalize_play_direction(out)
        if "play_direction" in out.columns:
            out = out.drop(columns=["play_direction_canonical"], errors="ignore")
    return out


def fit_pipeline(data_dir: str, artifact_dir: str = "./artifacts", config: Optional[ModelConfig] = None):
    os.makedirs(artifact_dir, exist_ok=True)
    config = config or ModelConfig()
    seed_everything(config.seed)

    files = discover_competition_files(data_dir)
    train_input = concat_csvs(files["train_input"])
    train_output = concat_csvs(files["train_output"])

    if train_input is None or train_output is None:
        raise FileNotFoundError("Could not find train_input*.csv and train_output*.csv in the provided data_dir.")

    print("Preprocessing train_input...")
    train_input = preprocess_input_df(train_input)
    print("Preprocessing train_output...")
    train_output = preprocess_output_df(train_output, train_input)

    pos_vocab, side_vocab, role_vocab = build_vocabs(train_input)

    all_keys = sorted(train_input.groupby(["game_id", "play_id"]).size().index.tolist())
    train_keys, val_keys = split_play_keys(all_keys, config.val_frac, config.seed)

    train_ds = NFLTrajectoryDataset(train_input, train_output, pos_vocab, side_vocab, role_vocab, train_keys)
    val_ds = NFLTrajectoryDataset(train_input, train_output, pos_vocab, side_vocab, role_vocab, val_keys)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_plays, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_plays, num_workers=config.num_workers)

    model = TrajectoryGNN(config, len(pos_vocab), len(side_vocab), len(role_vocab))
    save_path = os.path.join(artifact_dir, "best_model.pt")
    history = train_model(model, train_loader, val_loader, config, save_path)

    bundle = {
        "config": asdict(config),
        "pos_vocab": pos_vocab.itos,
        "side_vocab": side_vocab.itos,
        "role_vocab": role_vocab.itos,
        "history": history,
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
    model, config, pos_vocab, side_vocab, role_vocab, bundle = load_bundle(artifact_dir)

    test_input = pd.read_csv(test_input_path)
    test_input = preprocess_input_df(test_input)
    ds = NFLTrajectoryDataset(test_input, None, pos_vocab, side_vocab, role_vocab, None)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_plays, num_workers=0)

    rows = []
    device = config.device
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            pred = out["pred_xy"].cpu().numpy()  # [B,H,N,2]

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
        template = pd.read_csv(template_path)
        merge_cols = ["game_id", "play_id", "nfl_id", "frame_id"]
        full_df = template.merge(pred_df, on=merge_cols, how="left")
        if "id" in full_df.columns:
            submission = full_df[["id", "x", "y"]].copy()
            submission.to_csv(output_path, index=False)
        full_df.to_csv(full_output_path, index=False)
        return submission if "id" in full_df.columns else full_df

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
        
