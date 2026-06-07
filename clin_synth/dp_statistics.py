"""
dp_statistics.py
================
Applies the Laplace mechanism to seed statistics before they are injected
into the LLM prompt. Every statistic that touches real patient data is noised,
providing (ε, 0)-differential privacy guarantees.

Why this matters
----------------
Even though the LLM generator never sees the seed CSV directly, the statistics
computed from it (proportions, means, quantiles, correlations) are derived from
real patient records. Without DP, a sufficiently precise statistic can in theory
allow membership inference — determining whether a specific patient was in the
seed dataset.

How it works — Laplace mechanism
---------------------------------
For a statistic f(D) with global sensitivity Δf, the DP version is:

    f_dp(D) = f(D) + Lap(0, Δf / ε)

where ε (epsilon) is the privacy budget:
  - ε = 0.1   →  strong privacy,   higher distortion
  - ε = 1.0   →  moderate privacy, moderate distortion (recommended)
  - ε = 10.0  →  weak privacy,     low distortion

Privacy budget composition
---------------------------
We split ε across four stat families using basic composition:

    proportions  : ε * prop_weight
    numeric      : ε * numeric_weight
    missingness  : ε * missing_weight
    correlations : ε * corr_weight    (sum = 1.0)

The total privacy cost of publishing all noised statistics is ε (by composition).

Sensitivity derivation
-----------------------
  Proportions  : Δf = 2/n          (changing one person can shift two categories)
  Means        : Δf = (max - min)/n
  Std          : Δf = (max - min)/n (conservative)
  Quantiles    : Δf = (max - min)/n (smooth approximation)
  Missingness  : Δf = 1/n
  Correlations : Δf = 2/sqrt(n)    (for bounded [-1, 1] inputs, Sheffet 2015)
                 Upper triangle only; lower triangle is the symmetric copy.
                 Result is projected to the nearest PSD correlation matrix.

Usage
------
    python scripts/dp_statistics.py                         # default ε=1.0
    python scripts/dp_statistics.py --epsilon 0.5           # stronger privacy
    python scripts/dp_statistics.py --epsilon 2.0 --seed 42 # reproducible

Output: data/diab_stats_dp.json  (drop-in replacement for diab_stats.json)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np

from clin_synth.config import load_config, get_root as _get_root

logger = logging.getLogger(__name__)

def _dp_defaults() -> tuple[Path, Path]:
    """Read default paths from config.yaml; fall back to data/ if not found."""
    try:
        cfg = load_config().get("dp", {})
        root = _get_root()
        return root / cfg.get("raw_stats", "data/diab_stats.json"), \
               root / cfg.get("output",    "data/diab_stats_dp.json")
    except FileNotFoundError:
        root = Path(__file__).parent.parent
        return root / "data/diab_stats.json", root / "data/diab_stats_dp.json"

_DEFAULT_STATS, _DEFAULT_OUTPUT = _dp_defaults()


# ── Budget allocation weights (must sum to 1.0) ───────────────────────────────
_WEIGHTS = {
    "proportions":  0.40,
    "numeric":      0.30,
    "missingness":  0.10,
    "correlations": 0.20,
}


# ── Laplace noise ─────────────────────────────────────────────────────────────

def _laplace(scale: float, rng: np.random.Generator) -> float:
    """Draw one sample from Lap(0, scale). Returns 0 if scale ≤ 0."""
    if scale <= 0:
        return 0.0
    return float(rng.laplace(loc=0.0, scale=scale))


# ── Categorical / proportion stats ───────────────────────────────────────────

def _dp_proportions(
    freq_table: dict,
    n: int,
    epsilon: float,
    rng: np.random.Generator,
) -> dict:
    """
    Add Laplace noise to category counts, then renormalise.
    Sensitivity = 2/n (one person can move between two categories).
    Returns a noised copy of the frequency_table dict.
    """
    sensitivity = 2.0 / n
    scale = sensitivity / epsilon

    noised_counts: dict[str, float] = {}
    for category, data in freq_table.items():
        true_count = data["count"]
        noised = max(0.0, true_count + _laplace(scale * n, rng))
        noised_counts[category] = noised

    total = sum(noised_counts.values())
    if total == 0:
        total = 1.0  # prevent division by zero on degenerate cases

    noised_table = {}
    for category, noised_count in noised_counts.items():
        noised_prop = noised_count / total
        noised_table[category] = {
            "count":      round(noised_count),
            "proportion": round(noised_prop, 6),
        }
    return noised_table


def _dp_categorical_column(
    col_info: dict,
    n: int,
    epsilon: float,
    rng: np.random.Generator,
) -> dict:
    info = copy.deepcopy(col_info)
    if "frequency_table" in info and info["frequency_table"]:
        info["frequency_table"] = _dp_proportions(
            info["frequency_table"], n, epsilon, rng
        )
        # Recompute mode from noised table
        if info["frequency_table"]:
            mode_cat = max(
                info["frequency_table"],
                key=lambda k: info["frequency_table"][k]["proportion"],
            )
            info["mode"] = mode_cat
            info["mode_frequency"] = round(
                info["frequency_table"][mode_cat]["proportion"], 6
            )
    return info


# ── Numeric stats ─────────────────────────────────────────────────────────────

def _dp_numeric_column(
    col_info: dict,
    n: int,
    epsilon: float,
    rng: np.random.Generator,
) -> dict:
    """
    Add Laplace noise to mean, std, and quantiles.
    Sensitivity = (col_max - col_min) / n for all statistics.
    Noised values are clamped to [col_min, col_max] and std is clamped ≥ 0.

    Quantiles are noised via inter-quantile gaps so that monotonicity is
    preserved structurally rather than by post-hoc clamping. Clamping each
    noised quantile to the previous noised quantile introduces compounding
    upward bias and couples the noise draws across quantiles.
    """
    info = copy.deepcopy(col_info)
    col_min = info.get("min", 0)
    col_max = info.get("max", 1)
    col_range = max(col_max - col_min, 1e-9)
    sensitivity = col_range / n
    scale = sensitivity / epsilon

    def _noise_and_clamp(value: float) -> float:
        noised = value + _laplace(scale, rng)
        return float(np.clip(noised, col_min, col_max))

    if "mean" in info:
        info["mean"] = round(_noise_and_clamp(info["mean"]), 4)
    if "median" in info:
        info["median"] = round(_noise_and_clamp(info["median"]), 4)
    if "std" in info:
        info["std"] = round(
            max(0.0, info["std"] + _laplace(scale, rng)), 4
        )

    if "quantiles" in info:
        labels = list(info["quantiles"].keys())
        non_null = [(lbl, info["quantiles"][lbl])
                    for lbl in labels if info["quantiles"][lbl] is not None]

        if non_null:
            # Build anchor sequence: col_min, q1, q2, ..., qN, col_max
            anchors = [col_min] + [v for _, v in non_null] + [col_max]
            # Compute inter-quantile gaps; clamp to ≥ 0 for degenerate inputs
            gaps = [max(0.0, anchors[i + 1] - anchors[i])
                    for i in range(len(anchors) - 1)]
            # Noise each gap independently, then clamp to ≥ 0
            noised_gaps = [max(0.0, g + _laplace(scale, rng)) for g in gaps]
            # Rescale so the gaps sum to col_range (fit within [col_min, col_max])
            total = sum(noised_gaps)
            if total > 0:
                noised_gaps = [g * col_range / total for g in noised_gaps]
            # Reconstruct quantile positions by cumulative sum from col_min
            reconstructed = []
            running = col_min
            for g in noised_gaps[:-1]:   # last gap reaches col_max; skip it
                running += g
                reconstructed.append(round(running, 4))

            noised_q: dict = {}
            non_null_idx = 0
            for lbl in labels:
                if info["quantiles"][lbl] is None:
                    noised_q[lbl] = None
                else:
                    noised_q[lbl] = reconstructed[non_null_idx]
                    non_null_idx += 1
            info["quantiles"] = noised_q

    return info


# ── Missingness ────────────────────────────────────────────────────────────────

def _dp_missingness(
    col_info: dict,
    n: int,
    epsilon: float,
    rng: np.random.Generator,
) -> dict:
    """
    Noise the n_non_null count (sensitivity = 1) then recompute derived fields.
    """
    info = copy.deepcopy(col_info)
    if "n_non_null" not in info:
        return info

    sensitivity = 1.0
    scale = sensitivity / epsilon
    n_total = info.get("n_total", n)

    noised_non_null = int(np.clip(
        info["n_non_null"] + _laplace(scale, rng),
        0, n_total,
    ))
    info["n_non_null"] = noised_non_null
    return info


# ── Correlations ──────────────────────────────────────────────────────────────

def _project_to_correlation_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Project an approximate correlation matrix to the nearest valid one:
    clip negative eigenvalues to 0 (PSD), then rescale so the diagonal is 1.
    """
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, 0.0)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(psd))
    d = np.where(d > 0, d, 1.0)
    corr = psd / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _dp_correlations(
    correlations: dict,
    n: int,
    epsilon: float,
    rng: np.random.Generator,
) -> dict:
    """
    Add Laplace noise to Pearson r values while preserving valid correlation
    matrix structure (symmetry and positive semi-definiteness).

    Sensitivity = 2/sqrt(n) (Sheffet 2015 for bounded [-1,1] inputs).

    Approach:
    - Build the full k×k matrix from the pearson dict.
    - Noise only the upper triangle; copy to lower triangle (symmetry).
    - Project to the nearest valid correlation matrix via PSD projection.
    - Write noised values back into the dict at the correct nesting level.
    """
    sensitivity = 2.0 / (n ** 0.5)
    scale = sensitivity / epsilon

    noised = copy.deepcopy(correlations)
    pearson = noised.get("pearson", {})
    if not pearson:
        return noised

    cols = list(pearson.keys())
    k = len(cols)
    col_idx = {c: i for i, c in enumerate(cols)}

    # Build matrix from stored values (assume symmetric input)
    mat = np.eye(k)
    for col_a, targets in pearson.items():
        for col_b, r in targets.items():
            if col_b in col_idx:
                mat[col_idx[col_a], col_idx[col_b]] = r

    # Noise upper triangle only, then symmetrize — one draw per pair
    for i in range(k):
        for j in range(i + 1, k):
            noised_r = float(np.clip(mat[i, j] + _laplace(scale, rng), -1.0, 1.0))
            mat[i, j] = noised_r
            mat[j, i] = noised_r

    # Project to nearest valid correlation matrix
    mat = _project_to_correlation_matrix(mat)

    # Write back into the pearson sub-dict (not the top-level noised dict)
    for col_a, targets in pearson.items():
        for col_b in targets:
            if col_b in col_idx:
                noised["pearson"][col_a][col_b] = round(
                    float(mat[col_idx[col_a], col_idx[col_b]]), 4
                )

    return noised


# ── Main entry point ──────────────────────────────────────────────────────────

def apply_dp(
    stats: dict,
    epsilon: float = 1.0,
    seed: int | None = None,
) -> dict:
    """
    Apply (ε, 0)-differential privacy to all statistics via the Laplace mechanism.

    Parameters
    ----------
    stats   : Raw stats dict loaded from diab_stats.json.
    epsilon : Privacy budget. Lower = more private, more distortion.
    seed    : Optional RNG seed for reproducibility.

    Returns
    -------
    dict : DP-noised stats dict, structurally identical to the input.
    """
    rng = np.random.default_rng(seed)
    n   = stats["meta"]["n_rows"]

    eps_prop    = epsilon * _WEIGHTS["proportions"]
    eps_numeric = epsilon * _WEIGHTS["numeric"]
    eps_missing = epsilon * _WEIGHTS["missingness"]
    eps_corr    = epsilon * _WEIGHTS["correlations"]

    dp_stats = copy.deepcopy(stats)

    for col, info in dp_stats["columns"].items():
        semantic_type = info.get("semantic_type", "")

        if semantic_type == "categorical":
            dp_stats["columns"][col] = _dp_categorical_column(
                info, n, eps_prop, rng
            )

        elif semantic_type in ("integer", "float"):
            dp_stats["columns"][col] = _dp_numeric_column(
                info, n, eps_numeric, rng
            )

        # Missingness applied to every column
        dp_stats["columns"][col] = _dp_missingness(
            dp_stats["columns"][col], n, eps_missing, rng
        )

    if "correlations" in dp_stats:
        dp_stats["correlations"] = _dp_correlations(
            dp_stats["correlations"], n, eps_corr, rng
        )

    # Record DP provenance in metadata
    dp_stats["dp"] = {
        "mechanism":    "Laplace",
        "epsilon":      epsilon,
        "delta":        0,
        "n":            n,
        "budget_split": _WEIGHTS,
        "seed":         seed,
    }

    return dp_stats


def privacy_report(raw: dict, dp: dict) -> None:
    """Log a summary comparing raw vs DP statistics for key columns."""
    logger.info("\n%s", '─'*60)
    logger.info("  Differential Privacy Report   ε = %s", dp['dp']['epsilon'])
    logger.info("%s", '─'*60)
    logger.info("  Seed rows (n): %s", f"{dp['dp']['n']:,}")
    logger.info("  Mechanism    : %s", dp['dp']['mechanism'])
    logger.info("  Budget split : %s", dp['dp']['budget_split'])
    logger.info("")

    # Categorical drift
    logger.info("  Categorical proportions (raw vs noised):")
    for col in ["race", "gender", "readmitted", "A1Cresult"]:
        raw_col = raw["columns"].get(col, {})
        dp_col  =  dp["columns"].get(col, {})
        raw_ft  = raw_col.get("frequency_table", {})
        dp_ft   =  dp_col.get("frequency_table", {})
        if raw_ft:
            drifts = [
                abs(raw_ft[k]["proportion"] - dp_ft.get(k, {}).get("proportion", 0))
                for k in raw_ft
            ]
            logger.info("    %-20s  max drift = %.5f", col, max(drifts))

    logger.info("")

    # Numeric drift
    logger.info("  Numeric means (raw vs noised):")
    for col in ["time_in_hospital", "num_medications", "num_lab_procedures"]:
        raw_mean = raw["columns"].get(col, {}).get("mean")
        dp_mean  =  dp["columns"].get(col, {}).get("mean")
        if raw_mean is not None and dp_mean is not None:
            logger.info("    %-25s  %.3f → %.3f  (Δ = %.4f)",
                        col, raw_mean, dp_mean, abs(dp_mean - raw_mean))

    logger.info("")

    # Correlation drift
    raw_pearson = raw.get("correlations", {}).get("pearson", {})
    dp_pearson  =  dp.get("correlations", {}).get("pearson", {})
    if raw_pearson:
        logger.info("  Correlation drift (Pearson r):")
        for col_a, targets in raw_pearson.items():
            for col_b, r in targets.items():
                dp_r = dp_pearson.get(col_a, {}).get(col_b, r)
                drift = abs(r - dp_r)
                if drift > 0.001:
                    logger.info("    %s ↔ %-25s  %.4f → %.4f  (Δ = %.4f)",
                                col_a, col_b, r, dp_r, drift)

    logger.info("%s\n", '─'*60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply differential privacy to seed statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stats",   default=str(_DEFAULT_STATS),
                        help="Input stats JSON path")
    parser.add_argument("--output",  default=str(_DEFAULT_OUTPUT),
                        help="Output DP stats JSON path")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Privacy budget ε (lower = more private)")
    parser.add_argument("--seed",    type=int,   default=None,
                        help="RNG seed for reproducible noise")
    args = parser.parse_args()

    logger.info("Loading stats from : %s", args.stats)
    raw_stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))

    logger.info("Applying DP (ε = %s) ...", args.epsilon)
    dp_stats = apply_dp(raw_stats, epsilon=args.epsilon, seed=args.seed)

    privacy_report(raw_stats, dp_stats)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(dp_stats, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    logger.info("DP stats written to: %s", out_path)
