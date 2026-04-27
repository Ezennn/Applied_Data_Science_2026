"""
Option C - Step 1: preprocess local NFL tracking CSVs.

Supported raw filenames:
  input_2023_w01.csv  + output_2023_w01.csv
  train_input_2023_w01.csv + train_output_2023_w01.csv

Put raw CSVs under ./data, then run this file in PyCharm.
"""
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PREPROC_DIR = os.path.join(PROJECT_DIR, "preprocessed_csv_weekly")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREPROC_DIR, exist_ok=True)

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from export_preprocessed_weekly_csvs import export_preprocessed_csvs

manifest = export_preprocessed_csvs(
    data_dir=DATA_DIR,
    output_dir=PREPROC_DIR,
    include_test=False,
    output_prefix="preprocessed",
    save_manifest=True,
)

print("\nPreprocessing finished.")
print(f"Preprocessed files saved to: {PREPROC_DIR}")
print(f"Manifest: {manifest.get('manifest_path')}")
