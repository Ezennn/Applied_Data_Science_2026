from google.colab import drive

drive.mount('/content/drive')

import os
import sys

PROJECT_DIR = "/content/drive/MyDrive/NFL_GNN_Project"
DATA_DIR = "/content/drive/MyDrive/NFL_GNN_Project/data"
PREPROC_DIR = "/content/drive/MyDrive/NFL_GNN_Project/preprocessed_csv_weekly"

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(PREPROC_DIR, exist_ok=True)

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from nfl_bdb_preprocess_to_csv import export_preprocessed_csvs

manifest = export_preprocessed_csvs(
    data_dir=DATA_DIR,
    output_dir=PREPROC_DIR,
    include_test=True,
    output_prefix="preprocessed",
    save_manifest=True,
)

print("Preprocessing done.")
print(manifest)
