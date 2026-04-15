from google.colab import drive

drive.mount('/content/drive')

import os
import sys
import torch

PROJECT_DIR = "/content/drive/MyDrive/NFL_GNN_Project"
PREPROC_DIR = "/content/drive/MyDrive/NFL_GNN_Project/preprocessed_csv_weekly"
ARTIFACT_DIR = "/content/drive/MyDrive/NFL_GNN_Project/artifacts_preprocessed_csv_weekly"

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from nfl_bdb_train_from_preprocessed_csv import fit_from_preprocessed_dir, make_colab_fast_config

config = make_colab_fast_config()
config.batch_size = 4
config.num_workers = 2
config.num_epochs = 6
config.min_epochs = 2
config.patience = 2
config.attn_loss_weight = 0.0
config.device = "cuda" if torch.cuda.is_available() else "cpu"

fit_from_preprocessed_dir(
    preprocessed_dir=PREPROC_DIR,
    artifact_dir=ARTIFACT_DIR,
    config=config,
    eval_every=2,
    resume=True,
)
