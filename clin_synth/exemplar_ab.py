"""
exemplar_ab.py
==============
A/B comparison: system prompt WITH vs WITHOUT the EXEMPLAR ROWS section.

Runs two small generation jobs, then scores each output against the seed on
four axes:

  1. Class distribution fidelity   — mean |p_synth − p_seed| for each readmitted class
  2. Numeric marginal fidelity      — mean KS statistic across numeric columns
  3. Rare category coverage         — fraction of seed categories that appear in synthetic
  4. Correlation structure error    — mean |r_synth − r_seed| across numeric column pairs
  5. A1C / race missingness error   — |missing_rate_synth − missing_rate_seed|

Usage:
    python -m clin_synth.exemplar_ab
    python -m clin_synth.exemplar_ab --rows 300 --output-dir /tmp/ab
    python -m clin_synth.exemplar_ab --rows 500 --batch 50 --no-generate
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from clin_synth.config import load_config, get_root
from clin_synth.generate import generate_synthetic_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_ROOT = get_root()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt manipulation
# ─────────────────────────────────────────────────────────────────────────────

def strip_exemplar_section(prompt_text: str) -> str:
    """Remove the '# EXEMPLAR ROWS' section (heading through its trailing ---)."""
    stripped = re.sub(
        r"# EXEMPLAR ROWS\n.*?\n---\n",
        "",
        prompt_text,
        flags=re.DOTALL,
    )
    if stripped == prompt_text:
        logger.warning("No '# EXEMPLAR ROWS' section found — prompts are identical!")
    return stripped


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _class_distribution_mae(synth: pd.DataFrame, seed: pd.DataFrame, col: str) -> dict:
    """Mean absolute error of class proportions vs seed for a categorical column."""
    seed_p  = seed[col].value_counts(normalize=True)
    synth_p = synth[col].value_counts(normalize=True)
    all_cls = seed_p.index.union(synth_p.index)

    deltas = {}
    for cls in all_cls:
        sp = float(seed_p.get(cls, 0.0))
        tp = float(synth_p.get(cls, 0.0))
        deltas[str(cls)] = round(tp - sp, 4)

    mae = float(np.mean(np.abs(list(deltas.values()))))
    return {"mae": round(mae, 4), "per_class_delta": deltas}


def _numeric_ks(synth: pd.DataFrame, seed: pd.DataFrame, num_cols: list[str]) -> dict:
    """Mean KS statistic across numeric columns (lower = better fidelity)."""
    results = {}
    for col in num_cols:
        if col not in synth.columns or col not in seed.columns:
            continue
        s = pd.to_numeric(synth[col], errors="coerce").dropna()
        r = pd.to_numeric(seed[col],  errors="coerce").dropna()
        if len(s) == 0 or len(r) == 0:
            continue
        ks_stat, _ = ks_2samp(s, r)
        results[col] = round(float(ks_stat), 4)
    mean_ks = float(np.mean(list(results.values()))) if results else float("nan")
    return {"mean_ks": round(mean_ks, 4), "per_column": results}


def _rare_category_coverage(synth: pd.DataFrame, seed: pd.DataFrame, cat_cols: list[str]) -> dict:
    """
    For each categorical column: fraction of non-empty seed categories that appear in synthetic.
    Coverage = 1.0 means the synthetic reproduced every category the seed has.
    """
    results = {}
    for col in cat_cols:
        if col not in synth.columns or col not in seed.columns:
            continue
        seed_cats  = set(seed[col].dropna().unique())
        synth_cats = set(synth[col].dropna().unique())
        if not seed_cats:
            continue
        coverage = len(seed_cats & synth_cats) / len(seed_cats)
        missing  = sorted(seed_cats - synth_cats)
        results[col] = {
            "coverage": round(coverage, 4),
            "seed_n_cats": len(seed_cats),
            "missing_from_synth": missing,
        }
    overall = float(np.mean([v["coverage"] for v in results.values()])) if results else float("nan")
    return {"mean_coverage": round(overall, 4), "per_column": results}


def _correlation_mae(synth: pd.DataFrame, seed: pd.DataFrame, num_cols: list[str]) -> dict:
    """Mean absolute error between synth and seed Pearson correlation matrices."""
    cols = [c for c in num_cols if c in synth.columns and c in seed.columns]
    if len(cols) < 2:
        return {"mean_mae": float("nan"), "note": "Too few numeric columns"}

    synth_num = synth[cols].apply(pd.to_numeric, errors="coerce")
    seed_num  = seed[cols].apply(pd.to_numeric, errors="coerce")

    corr_synth = synth_num.corr()
    corr_seed  = seed_num.corr()

    diff = (corr_synth - corr_seed).abs()
    # Upper triangle only (excluding diagonal)
    mask = np.triu(np.ones(diff.shape, dtype=bool), k=1)
    mae  = float(diff.values[mask].mean())
    return {"mean_mae": round(mae, 4)}


def _missingness_error(synth: pd.DataFrame, seed: pd.DataFrame, cols: list[str]) -> dict:
    """Absolute difference in missing rate for specified columns."""
    results = {}
    for col in cols:
        if col not in synth.columns or col not in seed.columns:
            continue
        seed_rate  = float(seed[col].isna().mean())
        synth_rate = float(synth[col].isna().mean())
        results[col] = {
            "seed_rate":  round(seed_rate,  4),
            "synth_rate": round(synth_rate, 4),
            "abs_delta":  round(abs(synth_rate - seed_rate), 4),
        }
    return results


def score_output(synth: pd.DataFrame, seed: pd.DataFrame, cfg: dict) -> dict:
    """Compute all five metrics for one synthetic DataFrame."""
    schema   = cfg.get("schema", {})
    cat_cols = schema.get("categorical_columns", [])
    all_cols = schema.get("expected_columns", [])
    num_cols = [c for c in all_cols if c not in cat_cols]
    missing_cols = schema.get("columns_with_missingness", [])

    target = cfg.get("tstr", {}).get("target_column", "readmitted")

    return {
        "class_distribution": _class_distribution_mae(synth, seed, target) if target in synth.columns else {},
        "numeric_ks":         _numeric_ks(synth, seed, num_cols),
        "category_coverage":  _rare_category_coverage(synth, seed, cat_cols),
        "correlation_mae":    _correlation_mae(synth, seed, num_cols),
        "missingness_error":  _missingness_error(synth, seed, missing_cols or ["A1Cresult", "race"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def _winner(a: float, b: float, lower_is_better: bool = True) -> tuple[str, str]:
    """Return (label_a, label_b) with a '✓' on the better side."""
    if abs(a - b) < 1e-4:
        return ("=", "=")
    if lower_is_better:
        return ("✓", " ") if a < b else (" ", "✓")
    return ("✓", " ") if a > b else (" ", "✓")


def print_report(with_scores: dict, without_scores: dict) -> None:
    W  = "WITH exemplars"
    WO = "WITHOUT exemplars"

    def row(label: str, w: float, wo: float, lower_is_better: bool = True) -> None:
        wa, woa = _winner(w, wo, lower_is_better)
        logger.info("  %-40s %8.4f %s    %8.4f %s", label, w, wa, wo, woa)

    logger.info("")
    logger.info("=" * 72)
    logger.info("  EXEMPLAR ROWS A/B COMPARISON")
    logger.info("  %-40s %9s     %9s", "Metric", "WITH", "WITHOUT")
    logger.info("=" * 72)

    # 1. Class distribution MAE
    logger.info("\n  [1] Class distribution fidelity (lower MAE = better)")
    row("readmitted MAE",
        with_scores["class_distribution"].get("mae", float("nan")),
        without_scores["class_distribution"].get("mae", float("nan")))
    logger.info("\n      Per-class deltas (synth − seed):")
    wd = with_scores["class_distribution"].get("per_class_delta", {})
    od = without_scores["class_distribution"].get("per_class_delta", {})
    for cls in sorted(set(wd) | set(od)):
        wv = wd.get(cls, float("nan"))
        ov = od.get(cls, float("nan"))
        logger.info("    %6s:  WITH=%+.4f   WITHOUT=%+.4f", cls, wv, ov)

    # 2. Numeric marginals
    logger.info("\n  [2] Numeric marginal fidelity (mean KS, lower = better)")
    row("Mean KS statistic",
        with_scores["numeric_ks"]["mean_ks"],
        without_scores["numeric_ks"]["mean_ks"])
    logger.info("\n      Per-column KS:")
    all_ks_cols = sorted(
        set(with_scores["numeric_ks"]["per_column"]) |
        set(without_scores["numeric_ks"]["per_column"])
    )
    for col in all_ks_cols:
        wv = with_scores["numeric_ks"]["per_column"].get(col, float("nan"))
        ov = without_scores["numeric_ks"]["per_column"].get(col, float("nan"))
        wa, woa = _winner(wv, ov)
        logger.info("    %-30s %6.4f %s    %6.4f %s", col, wv, wa, ov, woa)

    # 3. Category coverage
    logger.info("\n  [3] Rare category coverage (higher = better, 1.0 = all seed cats reproduced)")
    row("Mean coverage",
        with_scores["category_coverage"]["mean_coverage"],
        without_scores["category_coverage"]["mean_coverage"],
        lower_is_better=False)
    logger.info("\n      Per-column:")
    all_cat_cols = sorted(
        set(with_scores["category_coverage"]["per_column"]) |
        set(without_scores["category_coverage"]["per_column"])
    )
    for col in all_cat_cols:
        wv = with_scores["category_coverage"]["per_column"].get(col, {})
        ov = without_scores["category_coverage"]["per_column"].get(col, {})
        wc = wv.get("coverage", float("nan"))
        oc = ov.get("coverage", float("nan"))
        wa, woa = _winner(wc, oc, lower_is_better=False)
        missing_w = wv.get("missing_from_synth", [])
        missing_o = ov.get("missing_from_synth", [])
        if missing_w or missing_o:
            logger.info("    %-20s %6.4f %s    %6.4f %s   missing: WITH=%s  WITHOUT=%s",
                        col, wc, wa, oc, woa, missing_w, missing_o)
        else:
            logger.info("    %-20s %6.4f %s    %6.4f %s", col, wc, wa, oc, woa)

    # 4. Correlation MAE
    logger.info("\n  [4] Correlation structure (mean |Δr|, lower = better)")
    row("Correlation MAE",
        with_scores["correlation_mae"]["mean_mae"],
        without_scores["correlation_mae"]["mean_mae"])

    # 5. Missingness
    logger.info("\n  [5] Missingness fidelity (|Δ missing rate|, lower = better)")
    all_miss_cols = sorted(
        set(with_scores["missingness_error"]) |
        set(without_scores["missingness_error"])
    )
    for col in all_miss_cols:
        wm = with_scores["missingness_error"].get(col, {})
        om = without_scores["missingness_error"].get(col, {})
        wv = wm.get("abs_delta", float("nan"))
        ov = om.get("abs_delta", float("nan"))
        wa, woa = _winner(wv, ov)
        logger.info("    %-20s seed=%.3f  WITH=%.3f (Δ=%.4f) %s    WITHOUT=%.3f (Δ=%.4f) %s",
                    col, wm.get('seed_rate', float('nan')),
                    wm.get('synth_rate', float('nan')), wv, wa,
                    om.get('synth_rate', float('nan')), ov, woa)

    logger.info("")
    logger.info("=" * 72)
    logger.info("  SUMMARY")
    logger.info("=" * 72)

    wins_w  = 0
    wins_wo = 0
    metrics = [
        ("Class dist MAE",     with_scores["class_distribution"].get("mae", float("nan")),  without_scores["class_distribution"].get("mae", float("nan")),  True),
        ("Numeric KS",         with_scores["numeric_ks"]["mean_ks"],                         without_scores["numeric_ks"]["mean_ks"],                         True),
        ("Category coverage",  with_scores["category_coverage"]["mean_coverage"],             without_scores["category_coverage"]["mean_coverage"],             False),
        ("Correlation MAE",    with_scores["correlation_mae"]["mean_mae"],                   without_scores["correlation_mae"]["mean_mae"],                   True),
    ]
    for label, wv, ov, lib in metrics:
        if abs(wv - ov) < 1e-4:
            verdict = "tie"
        elif (lib and wv < ov) or (not lib and wv > ov):
            verdict = "WITH wins"
            wins_w  += 1
        else:
            verdict = "WITHOUT wins"
            wins_wo += 1
        logger.info("  %-30s  %s", label, verdict)

    logger.info("")
    if wins_w > wins_wo:
        logger.info("  >> Exemplar rows appear HELPFUL — WITH wins more metrics.")
    elif wins_wo > wins_w:
        logger.info("  >> Exemplar rows appear UNHELPFUL — WITHOUT wins more metrics.")
    else:
        logger.info("  >> Mixed result — exemplar rows have no clear effect at this sample size.")
    logger.info("=" * 72)
    logger.info("")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    cfg     = load_config()
    gen_cfg = cfg.get("generation", {})
    val_cfg = cfg.get("validation", {})

    parser = argparse.ArgumentParser(
        description="A/B test: system prompt with vs without exemplar rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rows",         type=int,   default=300,
                        help="Rows to generate per condition")
    parser.add_argument("--batch",        type=int,   default=gen_cfg.get("batch_size", 100),
                        help="Rows per API call")
    parser.add_argument("--model",        type=str,   default=gen_cfg.get("model", "gpt-4o-mini"),
                        help="LLM model")
    parser.add_argument("--temperature",  type=float, default=gen_cfg.get("temperature", 0.7),
                        help="Generation temperature")
    parser.add_argument("--workers",      type=int,   default=gen_cfg.get("max_workers", 5),
                        help="Concurrent API workers")
    parser.add_argument("--system-prompt", type=str,
                        default=str(_ROOT / gen_cfg.get("system_prompt_file", "prompts/system_prompt.md")),
                        help="Base system prompt path (EXEMPLAR ROWS section will be stripped for B)")
    parser.add_argument("--stats",        type=str,
                        default=str(_ROOT / gen_cfg.get("stats_file", "data/diab_stats_dp.json")),
                        help="Seed stats JSON")
    parser.add_argument("--seed",         type=str,
                        default=str(_ROOT / cfg.get("run", {}).get("seed", "data/diab_seed.csv")),
                        help="Seed CSV for scoring")
    parser.add_argument("--output-dir",   type=str,   default=str(_ROOT / "processed_data" / "exemplar_ab"),
                        help="Directory to write outputs and report")
    parser.add_argument("--no-generate",  action="store_true",
                        help="Skip generation — score existing CSVs in --output-dir")
    parser.add_argument("--report-only",  action="store_true",
                        help="Alias for --no-generate")
    return parser.parse_args()


def main() -> None:
    args    = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()

    # ── Paths for the two output CSVs ─────────────────────────────────────
    csv_with    = out_dir / "synth_with_exemplars.csv"
    csv_without = out_dir / "synth_without_exemplars.csv"
    prompt_path = Path(args.system_prompt)

    skip_gen = args.no_generate or args.report_only

    if not skip_gen:
        # ── Load and strip the system prompt ─────────────────────────────
        prompt_with = prompt_path.read_text(encoding="utf-8")
        prompt_without = strip_exemplar_section(prompt_with)

        char_diff = len(prompt_with) - len(prompt_without)
        logger.info("System prompt: %d chars WITH exemplars, %d chars WITHOUT (Δ=%d)",
                 len(prompt_with), len(prompt_without), char_diff)

        # Write the no-exemplar prompt to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(prompt_without)
            no_exemplar_prompt_path = tmp.name

        # ── Remove stale output CSVs so generate() starts fresh ──────────
        for p in (csv_with, csv_without):
            if p.exists():
                p.unlink()

        common_kwargs = dict(
            total_rows            = args.rows,
            batch_size            = args.batch,
            model                 = args.model,
            temperature           = args.temperature,
            max_workers           = args.workers,
            stats_file            = args.stats,
            seed_csv              = args.seed,
        )

        # ── Condition A: WITH exemplars ───────────────────────────────────
        logger.info("=== Condition A: WITH exemplars ===")
        generate_synthetic_data(
            output_csv          = str(csv_with),
            system_prompt_file  = str(prompt_path),
            **common_kwargs,
        )

        # ── Condition B: WITHOUT exemplars ────────────────────────────────
        logger.info("=== Condition B: WITHOUT exemplars ===")
        generate_synthetic_data(
            output_csv          = str(csv_without),
            system_prompt_file  = no_exemplar_prompt_path,
            **common_kwargs,
        )

        Path(no_exemplar_prompt_path).unlink(missing_ok=True)

    # ── Load CSVs ─────────────────────────────────────────────────────────
    for p in (csv_with, csv_without):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing CSV: {p}\n"
                "Run without --no-generate to produce it first."
            )

    logger.info("Loading CSVs for scoring...")
    synth_with    = pd.read_csv(csv_with)
    synth_without = pd.read_csv(csv_without)
    seed_df       = pd.read_csv(args.seed)

    logger.info("WITH exemplars  : %d rows", len(synth_with))
    logger.info("WITHOUT exemplars: %d rows", len(synth_without))
    logger.info("Seed            : %d rows", len(seed_df))

    # ── Score ─────────────────────────────────────────────────────────────
    logger.info("Computing metrics...")
    scores_with    = score_output(synth_with,    seed_df, cfg)
    scores_without = score_output(synth_without, seed_df, cfg)

    # ── Print report ──────────────────────────────────────────────────────
    print_report(scores_with, scores_without)

    # ── Save JSON report ──────────────────────────────────────────────────
    report = {
        "config": {
            "rows":          args.rows,
            "batch":         args.batch,
            "model":         args.model,
            "temperature":   args.temperature,
            "system_prompt": str(prompt_path),
            "seed":          args.seed,
        },
        "n_rows": {
            "with_exemplars":    len(synth_with),
            "without_exemplars": len(synth_without),
            "seed":              len(seed_df),
        },
        "scores": {
            "with_exemplars":    scores_with,
            "without_exemplars": scores_without,
        },
    }
    report_path = out_dir / "exemplar_ab_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
