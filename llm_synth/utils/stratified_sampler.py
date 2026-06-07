"""
stratified_sampler.py
=====================
Pre-assigns categorical values for every synthetic row by sampling from
P(col | readmitted) computed from the seed CSV.

This replaces the post-hoc _enforce_categorical_proportions step and ensures
that categorical-target and categorical-numeric joint distributions are
preserved from the moment the LLM receives its prompt.

Usage:
    from stratified_sampler import sample_strata
    strata = sample_strata(n_rows=10_000, seed_df=pd.read_csv("data/diab_seed.csv"))
    # strata[i] is a dict with keys: race, gender, age, A1Cresult, metformin, insulin, readmitted
"""

import numpy as np
import pandas as pd

from llm_synth.config import load_config


# Module-level defaults — overridden by _init_strata_cfg() at call time
_CAT_COLS: list[str]              = []
_COLS_WITH_MISSINGNESS: list[str] = []
_CONDITIONING_COL: str            = "readmitted"


def _init_strata_cfg() -> None:
    """Refresh module-level config from the active (domain-merged) config."""
    global _CAT_COLS, _COLS_WITH_MISSINGNESS, _CONDITIONING_COL
    try:
        cfg    = load_config()
        target = cfg.get("tstr", {}).get("target_column", "readmitted")
        schema = cfg.get("schema", {})
        cat_cols  = schema.get("categorical_columns", [])
        miss_cols = schema.get("columns_with_missingness", [])
        miss_set  = set(miss_cols)
        _CAT_COLS              = [c for c in cat_cols if c not in miss_set and c != target]
        _COLS_WITH_MISSINGNESS = miss_cols
        _CONDITIONING_COL      = target
    except FileNotFoundError:
        pass


def _normalise(counts: dict) -> dict:
    """Normalise a {value: count} dict to {value: probability}."""
    total = sum(counts.values())
    if total == 0:
        return counts
    return {k: v / total for k, v in counts.items()}


def compute_conditional_distributions(seed_df: pd.DataFrame) -> dict:
    """
    Compute P(col | readmitted) for each categorical column.

    Returns
    -------
    dict with structure:
      {
        col: {
          readmitted_val: {cat_val: probability}   # for simple categoricals
          readmitted_val: {                         # for cols with missingness
            "_missing_rate": float,
            "_proportions": {cat_val: probability}
          }
        }
      }
    """
    readmitted_vals = seed_df[_CONDITIONING_COL].dropna().unique().tolist()
    distributions: dict = {}

    for col in _CAT_COLS:
        distributions[col] = {}
        for rv in readmitted_vals:
            subset = seed_df[seed_df[_CONDITIONING_COL] == rv][col]
            counts = subset.value_counts().to_dict()
            distributions[col][rv] = _normalise(counts)

    for col in _COLS_WITH_MISSINGNESS:
        distributions[col] = {}
        for rv in readmitted_vals:
            subset = seed_df[seed_df[_CONDITIONING_COL] == rv][col]
            # Treat NaN and empty string as missing
            is_missing = subset.isna() | (subset.astype(str).str.strip() == "")
            missing_rate = float(is_missing.mean())
            non_missing = subset[~is_missing]
            counts = non_missing.value_counts().to_dict()
            distributions[col][rv] = {
                "_missing_rate": missing_rate,
                "_proportions": _normalise(counts),
            }

    return distributions


def _sample_col(rng: np.random.Generator, dist: dict) -> str:
    """Sample one value from a normalised {value: probability} dict."""
    values = list(dist.keys())
    probs = list(dist.values())
    # Fix any floating-point drift so probs sum to exactly 1
    probs_arr = np.array(probs, dtype=float)
    probs_arr /= probs_arr.sum()
    return str(rng.choice(values, p=probs_arr))


def _sample_col_with_missingness(rng: np.random.Generator, dist: dict) -> str:
    """Sample from a distribution that includes a missingness rate."""
    if rng.random() < dist["_missing_rate"]:
        return ""
    props = dist["_proportions"]
    if not props:
        return ""
    return _sample_col(rng, props)


def sample_strata(
    n_rows: int,
    seed_df: pd.DataFrame,
    seed: int = 42,
) -> list[dict]:
    """
    Pre-assign categorical values for n_rows synthetic rows.

    Sampling order:
      1. readmitted  — sampled from marginal P(readmitted)
      2. all others  — sampled from P(col | readmitted)

    Parameters
    ----------
    n_rows   : Total rows to pre-assign.
    seed_df  : The original seed DataFrame (used to compute conditionals).
    seed     : RNG seed for reproducibility.

    Returns
    -------
    list of dicts, one per row, with keys:
      race, gender, age, A1Cresult, metformin, insulin, readmitted
    """
    rng = np.random.default_rng(seed)

    # Marginal distribution for readmitted
    readmitted_counts = seed_df[_CONDITIONING_COL].value_counts()
    readmitted_vals = readmitted_counts.index.tolist()
    readmitted_probs = (readmitted_counts / readmitted_counts.sum()).tolist()

    # Pre-sample all readmitted values at once for efficiency
    sampled_readmitted = rng.choice(readmitted_vals, size=n_rows, p=readmitted_probs)

    cond = compute_conditional_distributions(seed_df)

    rows: list[dict] = []
    for rv in sampled_readmitted:
        row: dict = {_CONDITIONING_COL: str(rv)}

        for col in _CAT_COLS:
            dist = cond[col].get(rv, {})
            row[col] = _sample_col(rng, dist) if dist else ""

        for col in _COLS_WITH_MISSINGNESS:
            dist = cond[col].get(rv, {})
            row[col] = _sample_col_with_missingness(rng, dist) if dist else ""

        rows.append(row)

    return rows


def strata_summary(strata: list[dict]) -> pd.DataFrame:
    """Return a proportion table for each categorical column — useful for QA."""
    df = pd.DataFrame(strata)
    summaries = []
    for col in df.columns:
        vc = df[col].replace("", "<missing>").value_counts(normalize=True)
        for val, prop in vc.items():
            summaries.append({"column": col, "value": val, "proportion": round(prop, 4)})
    return pd.DataFrame(summaries)
