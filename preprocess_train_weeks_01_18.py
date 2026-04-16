from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FIELD_LENGTH = 120.0
FIELD_WIDTH = 160.0 / 3.0  # 53.3333333333


def _rotate_180_for_left(values: pd.Series, is_left: pd.Series, field_limit: float) -> pd.Series:
    values = values.astype(float)
    return pd.Series(np.where(is_left, field_limit - values, values), index=values.index)


def _rotate_angle_180_for_left(values: pd.Series, is_left: pd.Series) -> pd.Series:
    values = values.astype(float)
    return pd.Series(np.where(is_left, (values + 180.0) % 360.0, values), index=values.index)


def standardize_offense_to_right(
    df: pd.DataFrame,
    play_direction_col: str = "play_direction",
    copy: bool = True,
    preserve_original_direction: bool = True,
    add_was_flipped: bool = True,
) -> pd.DataFrame:
    """
    Standardize all plays so offense moves to the right.

    For rows where play_direction == 'left', apply a 180-degree field rotation:
      x  -> 120 - x
      y  -> 160/3 - y
      dir -> (dir + 180) % 360
      o   -> (o + 180) % 360

    If present, the same rotation is also applied to:
      ball_land_x, ball_land_y, absolute_yardline_number
    """
    if play_direction_col not in df.columns:
        raise ValueError(
            f"Column '{play_direction_col}' not found. "
            "For tables without play_direction, merge in play-level direction first."
        )

    out = df.copy() if copy else df

    direction = out[play_direction_col].astype(str).str.lower().str.strip()
    is_left = direction.eq("left")

    if preserve_original_direction and "original_play_direction" not in out.columns:
        out["original_play_direction"] = out[play_direction_col]

    if add_was_flipped:
        out["was_flipped"] = is_left

    if "x" in out.columns:
        out["x"] = _rotate_180_for_left(out["x"], is_left, FIELD_LENGTH)

    if "y" in out.columns:
        out["y"] = _rotate_180_for_left(out["y"], is_left, FIELD_WIDTH)

    if "ball_land_x" in out.columns:
        out["ball_land_x"] = _rotate_180_for_left(out["ball_land_x"], is_left, FIELD_LENGTH)

    if "ball_land_y" in out.columns:
        out["ball_land_y"] = _rotate_180_for_left(out["ball_land_y"], is_left, FIELD_WIDTH)

    if "absolute_yardline_number" in out.columns:
        out["absolute_yardline_number"] = _rotate_180_for_left(
            out["absolute_yardline_number"], is_left, FIELD_LENGTH
        )

    for angle_col in ["dir", "o"]:
        if angle_col in out.columns:
            out[angle_col] = _rotate_angle_180_for_left(out[angle_col], is_left)

    out[play_direction_col] = "right"
    return out


def build_play_direction_map(
    input_df: pd.DataFrame,
    game_col: str = "game_id",
    play_col: str = "play_id",
    play_direction_col: str = "play_direction",
) -> pd.DataFrame:
    needed = [game_col, play_col, play_direction_col]
    missing = [c for c in needed if c not in input_df.columns]
    if missing:
        raise ValueError(f"Missing columns in input_df: {missing}")

    play_map = input_df[[game_col, play_col, play_direction_col]].drop_duplicates()

    dup_check = play_map.groupby([game_col, play_col])[play_direction_col].nunique()
    bad = dup_check[dup_check > 1]
    if len(bad) > 0:
        raise ValueError(
            "Found plays with multiple play_direction values. "
            "Expected exactly one direction per play."
        )

    return play_map


def standardize_output_with_input_reference(
    output_df: pd.DataFrame,
    input_df: pd.DataFrame,
    game_col: str = "game_id",
    play_col: str = "play_id",
    play_direction_col: str = "play_direction",
    copy: bool = True,
) -> pd.DataFrame:
    play_map = build_play_direction_map(
        input_df=input_df,
        game_col=game_col,
        play_col=play_col,
        play_direction_col=play_direction_col,
    )

    out = output_df.copy() if copy else output_df
    out = out.merge(play_map, on=[game_col, play_col], how="left", validate="many_to_one")

    if out[play_direction_col].isna().any():
        missing_pairs = out.loc[out[play_direction_col].isna(), [game_col, play_col]].drop_duplicates()
        raise ValueError(
            "Some output plays could not find play_direction in input_df. "
            f"Missing play pairs example:\n{missing_pairs.head()}"
        )

    out = standardize_offense_to_right(
        out,
        play_direction_col=play_direction_col,
        copy=False,
        preserve_original_direction=True,
        add_was_flipped=True,
    )

    return out


def week_strings(start_week: int, end_week: int) -> Iterable[str]:
    for week in range(start_week, end_week + 1):
        yield f"{week:02d}"


def process_one_week(train_dir: Path, output_dir: Path, season: int, week: str) -> None:
    input_path = train_dir / f"input_{season}_w{week}.csv"
    output_path = train_dir / f"output_{season}_w{week}.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")
    if not output_path.exists():
        raise FileNotFoundError(f"Missing output file: {output_path}")

    print(f"\n[Week {week}] Reading files...")
    input_df = pd.read_csv(input_path)
    output_df = pd.read_csv(output_path)

    print(f"[Week {week}] Standardizing input...")
    input_std = standardize_offense_to_right(input_df)

    print(f"[Week {week}] Standardizing output using input play_direction...")
    output_std = standardize_output_with_input_reference(output_df, input_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    input_out_path = output_dir / f"input_{season}_w{week}_standardized.csv"
    output_out_path = output_dir / f"output_{season}_w{week}_standardized.csv"

    input_std.to_csv(input_out_path, index=False)
    output_std.to_csv(output_out_path, index=False)

    print(f"[Week {week}] Saved: {input_out_path.name}")
    print(f"[Week {week}] Saved: {output_out_path.name}")
    print(f"[Week {week}] Done. Input rows: {len(input_std):,} | Output rows: {len(output_std):,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess NFL Big Data Bowl training files from week 01 to week 14 by "
            "standardizing all plays so offense moves to the right."
        )
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="train",
        help="Folder containing input_2023_wXX.csv and output_2023_wXX.csv. Default: ./train",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="train_preprocessed",
        help="Folder to save standardized CSV files. Default: ./train_preprocessed",
    )
    parser.add_argument("--season", type=int, default=2023, help="Season year in file names. Default: 2023")
    parser.add_argument("--start-week", type=int, default=1, help="Start week. Default: 1")
    parser.add_argument("--end-week", type=int, default=18, help="End week. Default: 18")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    train_dir = (base_dir / args.train_dir).resolve()
    output_dir = (base_dir / args.output_dir).resolve()

    print("=" * 72)
    print("NFL Big Data Bowl preprocessing")
    print(f"Script location : {base_dir}")
    print(f"Input folder    : {train_dir}")
    print(f"Output folder   : {output_dir}")
    print(f"Weeks           : {args.start_week:02d} to {args.end_week:02d}")
    print("=" * 72)

    for week in week_strings(args.start_week, args.end_week):
        process_one_week(train_dir=train_dir, output_dir=output_dir, season=args.season, week=week)

    print("\nAll preprocessing finished successfully.")


if __name__ == "__main__":
    main()
