from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import nasa_cmapss_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NASA CMAPSS data for sequence-based modeling.")
    parser.add_argument("--subset", default="FD001", help="CMAPSS subset to prepare. Default: FD001")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Data split to preprocess and transform into sequences. Default: train",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=nasa_cmapss_preprocessing.DEFAULT_SEQUENCE_LENGTH,
        help="Sliding-window length for exported sequences.",
    )
    parser.add_argument(
        "--target-position",
        default="last",
        choices=["last", "next"],
        help="Sequence target alignment. Default uses the last timestep in each window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = nasa_cmapss_preprocessing.run_preprocessing(
        subset=args.subset,
        split=args.split,
        sequence_length=args.sequence_length,
        target_position=args.target_position,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
