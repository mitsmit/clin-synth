"""
Compare seed-only vs synthetic-only XGBoost models.

Seed model:      trained + CV on seed data
Synthetic model: trained on synthetic data, tested on seed data (same held-out set)

Usage:
    python scripts/compare_results.py
"""

import json
import logging
from pathlib import Path

from llm_synth.config import get_root

logger = logging.getLogger(__name__)

RESULTS_DIR  = get_root() / "processed_data"
SEED_RESULT  = RESULTS_DIR / "xgboost_seed_seed_results.json"
SYNTH_RESULT = RESULTS_DIR / "xgboost_synth_seed_results.json"


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def bar(value: float, max_val: float, width: int = 28) -> str:
    filled = int(round(max(value, 0) / max_val * width)) if max_val > 0 else 0
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main():
    seed  = load(SEED_RESULT)
    synth = load(SYNTH_RESULT)

    logger.info("=" * 70)
    logger.info("  Model Comparison: Seed-trained vs Synthetic-trained")
    logger.info("  (both evaluated on real seed data)")
    logger.info("=" * 70)

    # --- Setup ---
    seed_f1  = seed["cv_f1_macro_mean"]   # CV mean on seed hold-out
    synth_f1 = synth["test_f1_macro"]     # test on seed hold-out
    delta    = synth_f1 - seed_f1
    pct      = (delta / seed_f1) * 100

    # --- Dataset ---
    logger.info("\n[ Dataset ]\n")
    logger.info("  %s %s  %s", f"{'':30}", f"{'Seed model':>12}", f"{'Synth model':>12}")
    logger.info("  %s %s  %s", f"{'-'*30}", f"{'-'*12}", f"{'-'*12}")
    logger.info("  %s %s  %s", f"{'Train data':<30}", f"{'seed':>12}", f"{'synthetic':>12}")
    logger.info("  %s %s  %s", f"{'Train rows':<30}", f"{seed['train_rows']:>12,}", f"{synth['train_rows']:>12,}")
    logger.info("  %s %s  %s", f"{'Test data':<30}", f"{'seed CV':>12}", f"{'seed hold-out':>12}")
    logger.info("  %s %s  %s", f"{'Test rows':<30}", f"{seed['train_rows']:>12,}", f"{synth['test_rows']:>12,}")

    # --- Performance ---
    logger.info("\n[ Performance (F1-macro on seed data) ]\n")
    logger.info("  %s %s  %s  %s", f"{'':30}", f"{'Seed model':>12}", f"{'Synth model':>12}", f"{'Delta':>8}")
    logger.info("  %s %s  %s  %s", f"{'-'*30}", f"{'-'*12}", f"{'-'*12}", f"{'-'*8}")
    logger.info("  %s %s  %s  %s", f"{'F1-macro':<30}", f"{seed_f1:>12.4f}", f"{synth_f1:>12.4f}", f"{delta:>+8.4f}")
    if "cv_f1_macro_std" in seed:
        logger.info("  %s %s  %s", f"{'  (std)':<30}", f"{seed['cv_f1_macro_std']:>12.4f}", f"{'n/a':>12}")

    verdict = "BETTER" if delta > 0 else "WORSE"
    logger.info("  >> Synthetic-trained model is %s by %.1f%% (%s F1-macro on real seed data)",
                verdict, abs(pct), 'higher' if delta > 0 else 'lower')

    # --- Feature importances ---
    logger.info("\n[ Feature Importances (permutation F1-macro drop) ]\n")
    all_features = sorted(
        set(seed["feature_importances"]) | set(synth["feature_importances"])
    )
    all_vals = list(seed["feature_importances"].values()) + list(synth["feature_importances"].values())
    max_imp = max(all_vals) if all_vals else 1.0

    ranked = sorted(
        all_features,
        key=lambda f: seed["feature_importances"].get(f, 0),
        reverse=True,
    )

    logger.info("  %s %s  %s  %s", f"{'Feature':<25}", f"{'Seed':>8}", f"{'Synth':>8}", f"{'Delta':>8}")
    logger.info("  %s %s  %s  %s", f"{'-'*25}", f"{'-'*8}", f"{'-'*8}", f"{'-'*8}")
    for feat in ranked:
        s_val = seed["feature_importances"].get(feat, 0.0)
        y_val = synth["feature_importances"].get(feat, 0.0)
        d_val = y_val - s_val
        logger.info("  %s %s  %s  %s  %s seed",
                    f"{feat:<25}", f"{s_val:>8.4f}", f"{y_val:>8.4f}", f"{d_val:>+8.4f}",
                    bar(s_val, max_imp, 20))
        logger.info("  %s %s  %s  %s  %s synth",
                    f"{'':<25}", f"{'':>8}", f"{'':>8}", f"{'':>8}",
                    bar(y_val, max_imp, 20))

    # --- Interpretation ---
    logger.info("\n[ Interpretation ]\n")
    if abs(delta) < 0.02:
        logger.info("  Synthetic data captures the real distribution well —")
        logger.info("  near-identical generalisation to real data.")
    elif delta < 0:
        logger.info("  Gap of %.4f F1-macro suggests the synthetic data does not", abs(delta))
        logger.info("  fully reproduce patterns in the real seed data.")
        logger.info("  Review class balance and feature distributions in the synthetic set.")
    else:
        logger.info("  Synthetic-trained model outperforms — the synthetic data may be")
        logger.info("  over-regularised or the seed CV score is pessimistic (small folds).")

    # --- Save ---
    comparison = {
        "seed_model": {
            "train_data": "seed",
            "train_rows": seed["train_rows"],
            "evaluation": "5-fold CV on seed",
            "f1_macro": seed_f1,
            "f1_macro_std": seed.get("cv_f1_macro_std"),
        },
        "synthetic_model": {
            "train_data": "synthetic",
            "train_rows": synth["train_rows"],
            "evaluation": "held-out seed data",
            "f1_macro": synth_f1,
        },
        "delta": {
            "f1_macro": round(delta, 4),
            "f1_macro_pct_change": round(pct, 2),
            "verdict": verdict,
        },
        "feature_importance_delta": {
            f: round(
                synth["feature_importances"].get(f, 0) - seed["feature_importances"].get(f, 0),
                4,
            )
            for f in ranked
        },
    }
    out_path = RESULTS_DIR / "comparison_report.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("  Report saved to %s", out_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
