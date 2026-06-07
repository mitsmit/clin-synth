"""
repair_synthetic.py
===================
One-time post-processing repair for synthetic_output.csv.

Applies all statistical corrections to the current (deduplicated) synthetic output:
  1. Categorical proportions  — age, race, metformin, insulin resampled to seed distribution
  2. Numeric distributions    — quantile normalization to match seed marginals
  3. Correlation structure    — Iman-Conover method to impose seed Pearson matrix

Usage:
    python scripts/utils/repair_synthetic.py
    python scripts/utils/repair_synthetic.py --input processed_data/synthetic_output.csv --seed data/diab_seed.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from clin_synth.config import load_config, get_root
from clin_synth.utils.postprocess_utils import (
    _enforce_numeric_distributions,
    _enforce_correlations,
)


def repair(input_csv: str, seed_csv: str) -> None:
    input_path = Path(input_csv)
    seed_path = Path(seed_csv)

    if not input_path.exists():
        logger.error("Input not found: %s", input_path.resolve())
        sys.exit(1)
    if not seed_path.exists():
        logger.error("Seed not found: %s", seed_path.resolve())
        sys.exit(1)

    logger.info("Reading: %s", input_path.resolve())
    rows: list[list[str]] = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)
    logger.info("  Rows loaded : %s", f"{len(rows):,}")

    col_indices = {col: i for i, col in enumerate(header)}

    logger.info("Reading seed: %s", seed_path.resolve())
    seed_df = pd.read_csv(seed_path)
    logger.info("  Seed rows   : %s\n", f"{len(seed_df):,}")

    logger.info("Step 1/3 — Enforcing categorical proportions (age, race, metformin, insulin)...")
    rows = _enforce_categorical_proportions(rows, col_indices, _CATEGORICAL_PROPORTIONS)

    logger.info("\nStep 2/3 — Quantile-normalizing numeric distributions...")
    rows = _enforce_numeric_distributions(rows, col_indices, seed_df)

    logger.info("\nStep 3/3 — Applying Iman-Conover correlation enforcement...")
    rows = _enforce_correlations(rows, col_indices, seed_df)

    logger.info("\nWriting repaired output: %s", input_path.resolve())
    with open(input_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info("Done. %s rows written.", f"{len(rows):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair synthetic_output.csv in-place.")
    parser.add_argument(
        "--input",
        default=str(_PROJECT_ROOT / "processed_data" / "synthetic_output.csv"),
        help="Path to the synthetic CSV to repair",
    )
    parser.add_argument(
        "--seed",
        default=str(_PROJECT_ROOT / "data" / "diab_seed.csv"),
        help="Path to the seed CSV",
    )
    args = parser.parse_args()
    repair(args.input, args.seed)
