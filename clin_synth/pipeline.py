"""
run_pipeline.py
===============
End-to-end synthetic data pipeline.

Usage:
    python scripts/run_pipeline.py --seed data/diab_seed.csv --condition "diabetes"
    python scripts/run_pipeline.py --seed data/ckd.csv --condition "ckd stage 3" --epsilon 0.5 --rows 5000
    python scripts/run_pipeline.py --seed data/ckd.csv --condition "ckd stage 3" --skip-tstr

Stages:
    1. Extract seed statistics
    2. Apply differential privacy to statistics
    3. Generate synthetic data
    4. Validate synthetic data (statistical + LLM + privacy)
    5. TSTR predictive model evaluation (train-on-synthetic / test-on-real)

All output files are written to <output-dir>/<condition_slug>_<artifact>.ext
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

from clin_synth.seed_statistics import extract_seed_stats
from clin_synth.dp_statistics import apply_dp
from clin_synth.generate import generate_synthetic_data
from clin_synth.validate import validate_synthetic_data, _print_report
from clin_synth.config import load_config, get_root


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

_W = 72

def _banner(msg: str) -> None:
    logger.info("\n%s", '═' * _W)
    logger.info("  %s", msg)
    logger.info("%s\n", '═' * _W)

def _ok(msg: str) -> None:
    logger.info("  ✓ %s", msg)

def _warn(msg: str) -> None:
    logger.warning("  ⚠  %s", msg)

def _elapsed(t0: float) -> str:
    secs = time.time() - t0
    return f"{secs:.1f}s" if secs < 60 else f"{secs/60:.1f}m"


# ─────────────────────────────────────────────────────────────────────────────
# TSTR helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_tstr(
    seed_file: str,
    synthetic_csv: str,
    output_dir: Path,
    slug: str,
) -> dict | None:
    """
    Run Train-on-Synthetic / Test-on-Real evaluation.

    Requires the dataset to contain the columns expected by train_xgboost.py
    (the diabetes schema: race, gender, age, …, readmitted).  Returns None and
    emits a warning if the schema does not match.
    """
    import clin_synth.tstr as tstr_module
    from clin_synth.tstr import (
        load_and_prepare,
        run_seed_seed,
        run_synth_synth,
        run_synth_seed,
        EXPECTED_COLS,
        TARGET,
    )
    from sklearn.model_selection import train_test_split

    # Schema guard — check both files before loading
    seed_cols  = set(pd.read_csv(seed_file,      nrows=0).columns)
    synth_cols = set(pd.read_csv(synthetic_csv,  nrows=0).columns)
    required   = EXPECTED_COLS | {TARGET}
    missing_seed  = required - seed_cols
    missing_synth = required - synth_cols

    if missing_seed or missing_synth:
        _warn("Skipping TSTR — dataset columns do not match the expected schema.")
        if missing_seed:
            _warn(f"  Seed  missing : {sorted(missing_seed)}")
        if missing_synth:
            _warn(f"  Synth missing : {sorted(missing_synth)}")
        return None

    # Redirect all tstr output files to the condition-specific output dir
    tstr_module.RESULTS_DIR = output_dir
    # Also patch the module-level save helper's path lookup
    tstr_module.SEED_FILE  = Path(seed_file)
    tstr_module.SYNTH_FILE = Path(synthetic_csv)

    seed_df  = load_and_prepare(Path(seed_file))
    synth_df = load_and_prepare(Path(synthetic_csv))
    feature_cols = [c for c in seed_df.columns if c != TARGET]
    logger.info("  Seed : %s rows  |  Synthetic : %s rows", f"{len(seed_df):,}", f"{len(synth_df):,}")
    logger.info("  Features : %s", feature_cols)

    seed_train_df, seed_holdout_df = train_test_split(
        seed_df, test_size=0.20, stratify=seed_df[TARGET], random_state=42
    )
    logger.info("  Seed hold-out split: %s train / %s test\n",
                f"{len(seed_train_df):,}", f"{len(seed_holdout_df):,}")

    r1 = run_seed_seed(seed_df,   feature_cols, cv_folds=5)
    r2 = run_synth_synth(synth_df, feature_cols)
    r3 = run_synth_seed(synth_df, seed_holdout_df, feature_cols)

    fidelity = r3.get("f1_macro", 0) / max(r1.get("f1_macro", 1e-9), 1e-9)
    tstr_summary = {
        "mode_1_seed_seed":   r1.get("f1_macro"),
        "mode_2_synth_synth": r2.get("f1_macro"),
        "mode_3_synth_seed":  r3.get("f1_macro"),
        "fidelity_ratio":     round(fidelity, 4),
    }

    tstr_path = output_dir / f"{slug}_tstr_comparison.json"
    tstr_path.write_text(json.dumps(tstr_summary, indent=2), encoding="utf-8")
    _ok(f"TSTR summary saved → {tstr_path}")

    bar_width = 30
    logger.info("\n  %-42s  %8s  Fidelity bar", "Mode", "F1-macro")
    logger.info("  %s  %s  %s", '─'*42, '─'*8, '─'*bar_width)
    for label, score in [
        ("seed → seed (hold-out)",           r1.get("f1_macro", 0)),
        ("synthetic → synthetic (hold-out)", r2.get("f1_macro", 0)),
        ("synthetic → seed  ← TSTR",         r3.get("f1_macro", 0)),
    ]:
        bar = "█" * int(score * bar_width)
        logger.info("  %-42s  %8.4f  %s", label, score, bar)
    logger.info("\n  Fidelity ratio (TSTR / seed-seed): %.2%%", fidelity)

    return tstr_summary


# ─────────────────────────────────────────────────────────────────────────────
# Rule mining (clin_synth.ruleex)
# ─────────────────────────────────────────────────────────────────────────────

def _mine_rules_if_supported(seed_file: str, condition: str, slug: str) -> None:
    """
    Mine hard/soft association rules for the seed dataset, alongside profiling.

    Writes domains/<slug>/hard_rules.csv (read by validate's association-rule
    coverage check) and domains/<slug>/soft_rules.csv (read by build-prompt's
    CLINICAL BEHAVIOR NARRATIVES section) — both are picked up automatically
    once present, no further wiring required.

    This is best-effort enrichment, not a required stage, and fully domain-driven:
    mining only runs if the active domain config has an `arm_rules.binning`
    section (see domains/heart_failure.yaml). Conditions without one are skipped
    with a warning rather than failing the pipeline.
    """
    from clin_synth.config import load_config
    from clin_synth.ruleex import mine_rules

    arm_cfg = load_config().get("arm_rules")
    if not arm_cfg or not arm_cfg.get("binning"):
        _warn(f"Rule mining skipped — domain config for '{condition}' has no 'arm_rules.binning' section.")
        return

    t0 = time.time()
    try:
        hard_df, soft_df, _ = mine_rules(seed_csv=seed_file, condition=condition)
    except (ValueError, RuntimeError) as exc:
        _warn(f"Rule mining failed — {exc}")
        return

    _ok(f"Mined {len(hard_df)} hard rules, {len(soft_df)} soft rules  [{_elapsed(t0)}]")
    _ok(f"Hard rules saved   → domains/{slug}/hard_rules.csv  (used by validate)")
    _ok(f"Soft rules saved   → domains/{slug}/soft_rules.csv  (used by build-prompt)")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    seed_file: str,
    condition: str,
    output_dir: Path,
    epsilon: float,
    total_rows: int,
    skip_tstr: bool,
    skip_llm: bool,
) -> None:
    import shutil

    slug = condition.lower().replace(" ", "_").replace("-", "_")

    # Directory layout:
    #   processed_data/<condition>/synthetic_output.csv
    #   processed_data/<condition>/stat_profile/{slug}_stats.json
    #   processed_data/<condition>/stat_profile/{slug}_dp_stats.json
    #   processed_data/<condition>/seed_data/<original-seed-filename>.csv
    condition_dir    = output_dir / slug
    stat_profile_dir = condition_dir / "stat_profile"
    seed_data_dir    = condition_dir / "seed_data"
    for d in (condition_dir, stat_profile_dir, seed_data_dir):
        d.mkdir(parents=True, exist_ok=True)

    pipeline_t0 = time.time()

    logger.info("\n  Condition   : %s", condition)
    logger.info("  Seed file   : %s", seed_file)
    logger.info("  Output dir  : %s", condition_dir)
    logger.info("  Rows to gen : %s", f"{total_rows:,}")
    logger.info("  DP epsilon  : %s", epsilon)
    logger.info("  TSTR        : %s", 'disabled' if skip_tstr else 'enabled')

    # ── Stage 1: Extract seed statistics ─────────────────────────────────────
    _banner(f"[1/5] Extracting seed statistics")
    t0 = time.time()
    stats_path = stat_profile_dir / f"{slug}_stats.json"
    seed_stats = extract_seed_stats(csv_path=seed_file, output_json=str(stats_path))
    n_rows = seed_stats["meta"]["n_rows"]
    n_cols = seed_stats["meta"]["n_columns"]
    _ok(f"Profile extracted  ({n_rows:,} rows, {n_cols} columns)  [{_elapsed(t0)}]")
    _ok(f"Stats saved        → {stats_path}")

    _mine_rules_if_supported(seed_file, condition, slug)

    # ── Stage 2: Apply differential privacy ──────────────────────────────────
    _banner(f"[2/5] Applying differential privacy  (ε = {epsilon})")
    t0 = time.time()
    dp_stats = apply_dp(seed_stats, epsilon=epsilon, seed=42)
    dp_stats_path = stat_profile_dir / f"{slug}_dp_stats.json"
    dp_stats_path.write_text(
        json.dumps(dp_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _ok(f"DP stats saved     → {dp_stats_path}  [{_elapsed(t0)}]")

    # ── Stage 3: Generate synthetic data ─────────────────────────────────────
    _banner(f"[3/5] Generating synthetic data  ({total_rows:,} rows)")
    t0 = time.time()
    synthetic_csv = str(condition_dir / "synthetic_output.csv")
    generate_synthetic_data(
        total_rows=total_rows,
        stats_file=str(dp_stats_path),
        output_csv=synthetic_csv,
        seed_csv=seed_file,
    )
    synth_row_count = len(pd.read_csv(synthetic_csv))
    _ok(f"Generated {synth_row_count:,} rows  [{_elapsed(t0)}]")
    _ok(f"Synthetic CSV      → {synthetic_csv}")

    # ── Copy seed file into seed_data/ for provenance ─────────────────────────
    seed_dest = seed_data_dir / Path(seed_file).name
    shutil.copy2(seed_file, seed_dest)
    _ok(f"Seed data archived → {seed_dest}")

    # ── Stage 4: Validate synthetic data ─────────────────────────────────────
    _banner(f"[4/5] Validating synthetic data")
    t0 = time.time()
    validation_report_path = str(condition_dir / f"{slug}_validation_report.json")
    report = validate_synthetic_data(
        synthetic_csv=synthetic_csv,
        seed_stats_json=str(stats_path),
        seed_csv=seed_file,
        output_report=validation_report_path,
        run_llm=not skip_llm,
    )
    _ok(f"Validation complete  [{_elapsed(t0)}]")
    _ok(f"Report saved       → {validation_report_path}")
    _print_report(report)

    # ── Stage 5: TSTR evaluation ──────────────────────────────────────────────
    if skip_tstr:
        _banner("[5/5] TSTR evaluation  (skipped)")
    else:
        _banner(f"[5/5] TSTR predictive model evaluation")
        t0 = time.time()
        tstr_result = _run_tstr(seed_file, synthetic_csv, condition_dir, slug)
        if tstr_result is not None:
            _ok(f"TSTR complete  [{_elapsed(t0)}]")

    # ── Pipeline complete ─────────────────────────────────────────────────────
    _banner(f"Pipeline complete  [{_elapsed(pipeline_t0)}]")
    logger.info("  Outputs written to: %s\n", condition_dir)
    for f in sorted(condition_dir.rglob("*")):
        if f.is_file():
            logger.info("    %s", f.relative_to(condition_dir))
    logger.info("")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end synthetic data pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed", required=True,
        help="Path to the seed CSV file.",
    )
    parser.add_argument(
        "--condition", required=True,
        help="Condition name (e.g. 'ckd stage 3'). Used as output file prefix.",
    )
    parser.add_argument(
        "--output-dir", default=str(get_root() / "processed_data"),
        help="Directory to write all output files.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0,
        help="Differential privacy budget ε (lower = more private, more distortion).",
    )
    parser.add_argument(
        "--rows", type=int, default=None,
        help="Total synthetic rows to generate. Defaults to row count in config.yaml.",
    )
    parser.add_argument(
        "--skip-tstr", action="store_true",
        help="Skip TSTR predictive model evaluation.",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM deep-analysis step in validation (faster, no OpenAI call).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()

    # Resolve rows: CLI arg > config.yaml > 1 000 fallback
    total_rows = args.rows
    if total_rows is None:
        try:
            total_rows = load_config().get("generation", {}).get("total_rows", 1_000)
        except FileNotFoundError:
            total_rows = 1_000

    seed_path = Path(args.seed)
    if not seed_path.exists():
        logger.error("seed file not found: %s", seed_path)
        sys.exit(1)

    run_pipeline(
        seed_file=str(seed_path),
        condition=args.condition,
        output_dir=Path(args.output_dir),
        epsilon=args.epsilon,
        total_rows=total_rows,
        skip_tstr=args.skip_tstr,
        skip_llm=args.skip_llm,
    )
