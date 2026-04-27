import glob
import importlib.util
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

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


def _resolve_data_dir(data_dir: str) -> str:
    data_dir = os.path.abspath(data_dir)
    if os.path.isfile(data_dir) and data_dir.lower().endswith(".zip"):
        print(f"data_dir is a zip file. Extracting: {data_dir}")
        return base._extract_zip_if_needed(
            data_dir,
            os.path.join(os.path.dirname(data_dir), "extracted_competition_data"),
        )
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
    return data_dir


def _extract_week_key(path: str, kind: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    m = re.search(r"(\d{4}_w\d{2})", stem)
    if m:
        return m.group(1)
    if kind == "test":
        return "test"
    return "all"


def _pair_paths(input_paths: List[str], output_paths: List[str]) -> List[Tuple[str, str, str]]:
    input_map: Dict[str, str] = {}
    for p in input_paths:
        key = _extract_week_key(p, "input")
        input_map[key] = p

    output_map: Dict[str, str] = {}
    for p in output_paths:
        key = _extract_week_key(p, "output")
        output_map[key] = p

    common_keys = sorted(set(input_map) & set(output_map))
    missing_inputs = sorted(set(output_map) - set(input_map))
    missing_outputs = sorted(set(input_map) - set(output_map))
    if missing_inputs or missing_outputs:
        raise FileNotFoundError(
            "Found unmatched weekly raw CSV files. "
            f"missing_inputs={missing_inputs}, missing_outputs={missing_outputs}"
        )
    return [(k, input_map[k], output_map[k]) for k in common_keys]


def _weekly_output_names(output_dir: str, output_prefix: str, week_key: str) -> Tuple[str, str]:
    if week_key == "all":
        return (
            os.path.join(output_dir, f"{output_prefix}_train_input.csv"),
            os.path.join(output_dir, f"{output_prefix}_train_output.csv"),
        )
    return (
        os.path.join(output_dir, f"{output_prefix}_train_input_{week_key}.csv"),
        os.path.join(output_dir, f"{output_prefix}_train_output_{week_key}.csv"),
    )


def export_preprocessed_csvs(
    data_dir: str,
    output_dir: str,
    include_test: bool = True,
    output_prefix: str = "preprocessed",
    save_manifest: bool = True,
) -> Dict[str, object]:
    """
    Read the original CSV files, preprocess them week by week,
    and write standalone weekly preprocessed CSV files. Supports both
    input_2023_wXX.csv/output_2023_wXX.csv and
    train_input_2023_wXX.csv/train_output_2023_wXX.csv naming.
    """
    os.makedirs(output_dir, exist_ok=True)
    data_dir = _resolve_data_dir(data_dir)
    files = base.discover_competition_files(data_dir)

    train_input_paths = files.get("train_input", [])
    train_output_paths = files.get("train_output", [])
    test_input_paths = files.get("test_input", [])
    sample_submission_paths = files.get("sample_submission", [])

    if not train_input_paths or not train_output_paths:
        raise FileNotFoundError(
            "Could not find input/train_input and output/train_output CSV files under data_dir. Supported raw names include input_2023_w01.csv + output_2023_w01.csv, or train_input_2023_w01.csv + train_output_2023_w01.csv."
        )

    train_pairs = _pair_paths(train_input_paths, train_output_paths)
    if not train_pairs:
        raise FileNotFoundError("Could not pair any weekly input/output CSV files. Use matching weeks, for example input_2023_w01.csv with output_2023_w01.csv.")

    manifest: Dict[str, object] = {
        "data_dir": data_dir,
        "output_dir": os.path.abspath(output_dir),
        "mode": "weekly_preprocessed_csvs",
        "train_pairs": [],
        "test_input_csv": None,
        "sample_submission_csv": None,
        "obs_feature_names": list(base.OBS_FEATURE_NAMES),
        "source_train_input_files": train_input_paths,
        "source_train_output_files": train_output_paths,
        "source_test_input_files": test_input_paths,
    }

    for idx, (week_key, train_input_path, train_output_path) in enumerate(train_pairs, start=1):
        print(f"[{idx}/{len(train_pairs)}] Loading raw weekly training CSV files for {week_key}...")
        train_input = pd.read_csv(train_input_path, usecols=base.INPUT_REQUIRED_COLS)
        train_output = pd.read_csv(train_output_path, usecols=base.OUTPUT_REQUIRED_COLS)

        print(f"Preprocessing train_input for {week_key}...")
        train_input_pp = base.preprocess_input_df(train_input)
        print(f"Preprocessing train_output for {week_key}...")
        train_output_pp = base.preprocess_output_df(train_output, train_input_pp)

        train_input_csv, train_output_csv = _weekly_output_names(output_dir, output_prefix, week_key)
        print(f"Saving {train_input_csv}")
        train_input_pp.to_csv(train_input_csv, index=False)
        print(f"Saving {train_output_csv}")
        train_output_pp.to_csv(train_output_csv, index=False)

        manifest["train_pairs"].append({
            "week_key": week_key,
            "source_train_input_csv": train_input_path,
            "source_train_output_csv": train_output_path,
            "preprocessed_train_input_csv": train_input_csv,
            "preprocessed_train_output_csv": train_output_csv,
            "train_input_rows": int(len(train_input_pp)),
            "train_output_rows": int(len(train_output_pp)),
            "train_input_columns": list(train_input_pp.columns),
            "train_output_columns": list(train_output_pp.columns),
        })

    if include_test and test_input_paths:
        print("Loading raw test CSV files...")
        test_input = base.concat_csvs(test_input_paths, usecols=base.INPUT_REQUIRED_COLS)
        if test_input is None:
            raise RuntimeError("Failed to load raw test CSV files.")
        print("Preprocessing test_input...")
        test_input_pp = base.preprocess_input_df(test_input)
        test_input_csv = os.path.join(output_dir, f"{output_prefix}_test_input.csv")
        print(f"Saving {test_input_csv}")
        test_input_pp.to_csv(test_input_csv, index=False)
        manifest["test_input_csv"] = test_input_csv
        manifest["test_input_rows"] = int(len(test_input_pp))
        manifest["test_input_columns"] = list(test_input_pp.columns)

    if sample_submission_paths:
        sample_submission = pd.read_csv(sample_submission_paths[0])
        sample_submission_copy = os.path.join(output_dir, "sample_submission.csv")
        sample_submission.to_csv(sample_submission_copy, index=False)
        manifest["sample_submission_csv"] = sample_submission_copy

    if save_manifest:
        manifest_path = os.path.join(output_dir, "preprocessed_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        manifest["manifest_path"] = manifest_path

    print("Done.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:4000])
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export weekly preprocessed train/test CSV files for the NFL GNN pipeline.")
    parser.add_argument("--data_dir", required=True, help="Folder containing the original competition CSVs, or a zip file path.")
    parser.add_argument("--output_dir", required=True, help="Folder to save the preprocessed CSV files.")
    parser.add_argument("--output_prefix", default="preprocessed", help="Filename prefix for exported CSV files.")
    parser.add_argument("--skip_test", action="store_true", help="Do not export the preprocessed test_input CSV.")
    args = parser.parse_args()

    export_preprocessed_csvs(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        include_test=not args.skip_test,
        output_prefix=args.output_prefix,
        save_manifest=True,
    )
