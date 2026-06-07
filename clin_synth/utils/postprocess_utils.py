"""
postprocess_utils.py
====================
Post-processing utilities for synthetic data generation.

Provides numeric distribution enforcement, Iman-Conover correlation
replication, and categorical strata pinning used by generate.py.
"""

import logging

import numpy as np
import pandas as pd

from clin_synth.config import load_config

logger = logging.getLogger(__name__)


# Module-level defaults — overridden by _init_postprocess_cfg() at call time
_NUMERIC_COLS: list[str]    = []
_READMITTED_ENC: dict       = {}
_READMITTED_DEC: dict       = {}
_READMITTED_VALS: list[str] = []

# Columns whose seed zero-rate exceeds this get two-stage zero-inflated mapping
# instead of plain quantile normalization.
_ZERO_INFLATED_THRESHOLD = 0.5


def _init_postprocess_cfg() -> None:
    """Refresh module-level config from the active (domain-merged) config."""
    global _NUMERIC_COLS, _READMITTED_ENC, _READMITTED_DEC, _READMITTED_VALS
    try:
        cfg          = load_config()
        tstr         = cfg.get("tstr", {})
        target       = tstr.get("target_column", "readmitted")
        target_order = tstr.get("target_order")

        all_cols     = cfg.get("schema", {}).get("expected_columns", [])
        cat_cols     = set(cfg.get("schema", {}).get("categorical_columns", []))
        _NUMERIC_COLS = [c for c in all_cols if c not in cat_cols and c != target] or []

        if target_order:
            _READMITTED_ENC  = {v: i for i, v in enumerate(target_order)}
            _READMITTED_DEC  = {i: v for i, v in enumerate(target_order)}
            _READMITTED_VALS = list(target_order)
        else:
            _READMITTED_ENC  = {}
            _READMITTED_DEC  = {}
            _READMITTED_VALS = []
    except FileNotFoundError:
        pass


def _enforce_numeric_distributions(
    rows: list[list[str]],
    col_indices: dict[str, int],
    seed_df: "pd.DataFrame",
) -> list[list[str]]:
    """
    Enforce seed marginal distributions on synthetic numeric columns via empirical
    CDF matching (rank-preserving). Two strategies are applied automatically:

    Standard columns (zero rate < _ZERO_INFLATED_THRESHOLD):
        Plain quantile normalization — rank percentile mapped to seed CDF value.

    Zero-inflated columns (zero rate >= _ZERO_INFLATED_THRESHOLD):
        Two-stage mapping:
          1. Bottom zero_rate fraction of ranks → 0 (preserves zero spike)
          2. Remaining ranks → non-zero seed CDF, capped at p99 to avoid
             extreme-outlier inflation caused by small batch sizes mapping
             the top rank to the seed maximum.
    """
    if not rows:
        return rows
    rows = [list(r) for r in rows]

    for col in _NUMERIC_COLS:
        idx = col_indices.get(col)
        if idx is None or col not in seed_df.columns:
            continue

        seed_vals = seed_df[col].dropna().to_numpy(dtype=float)
        if len(seed_vals) == 0:
            continue

        valid_pos, vals = [], []
        for i, row in enumerate(rows):
            try:
                vals.append(float(row[idx]))
                valid_pos.append(i)
            except (ValueError, IndexError):
                pass

        if not vals:
            continue

        arr      = np.array(vals)
        old_mean = float(arr.mean())
        n        = len(arr)
        ranks    = np.argsort(np.argsort(arr, kind="stable"), kind="stable")

        zero_rate = float((seed_vals == 0).mean())
        new_vals  = np.zeros(n, dtype=int)

        if zero_rate >= _ZERO_INFLATED_THRESHOLD:
            # ── Two-stage zero-inflated mapping ──────────────────────────────
            nonzero_seed  = np.sort(seed_vals[seed_vals > 0])
            n_nz_seed     = len(nonzero_seed)
            zero_cutoff   = int(round(zero_rate * n))
            nonzero_count = max(n - zero_cutoff, 1)

            # p99 cap index within the non-zero distribution — prevents the
            # top rank in a small batch from mapping to the seed maximum.
            p99_nz_idx = int(np.clip(round(0.99 * (n_nz_seed - 1)), 0, n_nz_seed - 1))

            for local_i in range(len(valid_pos)):
                r = int(ranks[local_i])
                if r >= zero_cutoff and n_nz_seed > 0:
                    nz_rank = r - zero_cutoff
                    q       = nz_rank / max(nonzero_count - 1, 1)
                    nz_idx  = int(np.clip(round(q * p99_nz_idx), 0, p99_nz_idx))
                    new_vals[local_i] = int(nonzero_seed[nz_idx])
                # else: new_vals[local_i] stays 0

            tag = f"  [zero-inflated {zero_rate:.0%} zeros]"
        else:
            # ── Standard quantile normalization ───────────────────────────────
            seed_sorted = np.sort(seed_vals)
            n_seed      = len(seed_sorted)
            q_frac      = ranks / max(n - 1, 1)
            seed_idx    = np.clip(np.round(q_frac * (n_seed - 1)).astype(int), 0, n_seed - 1)
            new_vals    = np.round(seed_sorted[seed_idx]).astype(int)
            tag         = ""

        for local_i, row_idx in enumerate(valid_pos):
            rows[row_idx][idx] = str(new_vals[local_i])

        logger.info(
            f"  [numeric] {col}: mean {old_mean:.2f} → {float(new_vals.mean()):.2f} "
            f"(seed: {float(seed_df[col].mean()):.2f}){tag}"
        )

    return rows


def _ic_apply(
    mat: "np.ndarray",
    R: "np.ndarray",
    rng_seed: int,
) -> "np.ndarray":
    """
    Core Iman-Conover step: reorder each column of mat to match the rank
    structure of correlated normals drawn with correlation matrix R.
    Returns a new matrix with the same marginals but imposed correlations.
    """
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-8)
    R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    L = np.linalg.cholesky(R_psd)

    n, P = mat.shape
    rng = np.random.RandomState(rng_seed)
    T = rng.standard_normal((n, P)) @ L.T

    out = np.empty_like(mat)
    for j in range(P):
        sorted_vals = np.sort(mat[:, j])
        t_ranks = np.argsort(np.argsort(T[:, j], kind="stable"), kind="stable")
        out[:, j] = sorted_vals[t_ranks]
    return out


def _enforce_correlations(
    rows: list[list[str]],
    col_indices: dict[str, int],
    seed_df: "pd.DataFrame",
    stratified: bool = False,
) -> list[list[str]]:
    """
    Apply the Iman-Conover method to impose the seed's Pearson correlation
    structure on numeric columns. Each column's marginal distribution is
    preserved exactly — only the row-level pairing of values changes.

    stratified=False (non-stratified mode):
        Global IC on all numeric columns + readmitted (ordinal-encoded).
        readmitted values are reordered to align with the numeric structure.

    stratified=True (stratified mode):
        IC runs independently within each readmitted stratum using the
        seed's within-stratum correlation matrix. readmitted is never
        reordered, preserving the categorical-numeric joint distribution
        the LLM conditioned on per row.
    """
    if len(rows) < 2:
        return rows

    rows = [list(r) for r in rows]
    num_cols = [c for c in _NUMERIC_COLS if c in col_indices and c in seed_df.columns]

    if stratified:
        readmitted_idx = col_indices.get("readmitted")
        if readmitted_idx is None or len(num_cols) < 2:
            return rows

        for s_i, rv in enumerate(_READMITTED_VALS):
            s_idx = [i for i, r in enumerate(rows) if r[readmitted_idx].strip() == rv]
            if len(s_idx) < 2:
                continue

            seed_s = seed_df[seed_df["readmitted"] == rv]
            if len(seed_s) < 2:
                continue

            P = len(num_cols)
            mat = np.zeros((len(s_idx), P), dtype=float)
            for j, col in enumerate(num_cols):
                idx = col_indices[col]
                for local_i, global_i in enumerate(s_idx):
                    try:
                        mat[local_i, j] = float(rows[global_i][idx])
                    except (ValueError, IndexError):
                        pass

            R = seed_s[num_cols].corr(method="pearson").to_numpy(dtype=float)
            new_mat = _ic_apply(mat, R, rng_seed=42 + s_i)

            for j, col in enumerate(num_cols):
                idx = col_indices[col]
                for local_i, global_i in enumerate(s_idx):
                    rows[global_i][idx] = str(int(round(float(new_mat[local_i, j]))))

            achieved = np.corrcoef(new_mat.T)
            max_gap = float(np.max(np.abs(achieved - R)))
            logger.info(f"  [correlations] IC within readmitted={rv!r} (n={len(s_idx)}) — max |gap|: {max_gap:.3f}")

    else:
        # Global IC: numeric cols + readmitted ordinal-encoded
        ic_cols = num_cols + (["readmitted"] if "readmitted" in col_indices and "readmitted" in seed_df.columns else [])
        if len(ic_cols) < 2:
            return rows

        n, P = len(rows), len(ic_cols)
        mat = np.zeros((n, P), dtype=float)
        for j, col in enumerate(ic_cols):
            idx = col_indices[col]
            for i, row in enumerate(rows):
                if col == "readmitted":
                    mat[i, j] = float(_READMITTED_ENC.get(row[idx].strip(), 0))
                else:
                    try:
                        mat[i, j] = float(row[idx])
                    except (ValueError, IndexError):
                        pass

        seed_enc = seed_df.copy()
        seed_enc["readmitted"] = seed_enc["readmitted"].map(_READMITTED_ENC)
        R = seed_enc[ic_cols].corr(method="pearson").to_numpy(dtype=float)
        new_mat = _ic_apply(mat, R, rng_seed=42)

        for j, col in enumerate(ic_cols):
            idx = col_indices[col]
            for i, row in enumerate(rows):
                if col == "readmitted":
                    key = max(0, min(2, int(round(float(new_mat[i, j])))))
                    row[idx] = _READMITTED_DEC[key]
                else:
                    row[idx] = str(int(round(float(new_mat[i, j]))))

        achieved = np.corrcoef(new_mat.T)
        max_gap = float(np.max(np.abs(achieved - R)))
        logger.info("  [correlations] Iman-Conover applied — max |gap| from seed: %.3f", max_gap)

    return rows


def _enforce_strata(
    rows: list[list[str]],
    batch_categoricals: list[dict],
    col_indices: dict[str, int],
) -> list[list[str]]:
    """
    Overwrite categorical columns with pre-assigned strata values.
    Called after parsing LLM output to guarantee the LLM did not drift.
    Only overwrites rows that were successfully parsed (positional match).
    Iterates over the strata dict's own keys so this works for any domain.
    """
    rows = [list(r) for r in rows]
    for i, row in enumerate(rows):
        if i >= len(batch_categoricals):
            break
        for col, val in batch_categoricals[i].items():
            idx = col_indices.get(col)
            if idx is not None:
                row[idx] = val
    return rows
