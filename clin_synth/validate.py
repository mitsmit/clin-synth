"""
validate.py
=====================
Validate generated synthetic data against the seed schema and statistics
using GPT-4o-mini as an independent analysis model.

Usage:
    python validate.py generated_data.csv seed_data_statistics.json
    
    OR programmatically:
    from validate import validate_synthetic_data
    report = validate_synthetic_data("generated_data.csv", "seed_data_statistics.json")
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

from clin_synth.clinical_plausibility import (
    check_clinical_plausibility,
    check_rule_based_clinicalplausibility,
    generate_rule_coverage_report,
)
# from clin_synth.utils.plots import plot_synthetic_data
from clin_synth.utils.util import (
    _gower_encode,
    _gower_nn_dist,
    _is_hash_column,
    _psi_status,
    _psi_categorical,
    _psi_numeric,
    compute_psi,
    _print_report,
)
from clin_synth.config import load_config, get_root, get_domain_name

import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from scipy import stats as scipy_stats

_ROOT = get_root()
load_dotenv(_ROOT / ".env")
_openai_client = OpenAI()

# ─────────────────────────────────────────────────────────────────────────────
# Constants — hardcoded defaults, overridden by _init_val_cfg() at call time
# ─────────────────────────────────────────────────────────────────────────────

TOLERANCE                 = 0.05
K_ANONYMITY_THRESHOLD     = 5
QUASI_ID_COLS: list[str]  = ["patient_state", "patient_zip3", "patient_year_of_birth", "pay_type"]

_CARDINALITY_RATIO_THRESHOLD = 1.2
_CORR_DRIFT_THRESHOLD        = 0.15
_LEAKAGE_RATE_THRESHOLD      = 0.01
_DATETIME_CHECKS: dict       = {}
_LLM_MODEL                   = "gpt-4o-mini"
_LLM_MAX_TOKENS              = 4000
_RUN_PRIVACY_TESTS           = True

_MIA_SAMPLE_SIZE        = 2000
_MIA_AUC_THRESHOLD      = 0.60
_DCR_RATIO_THRESHOLD    = 0.80
_NNDR_THRESHOLD         = 0.20
_LINKAGE_QI_COLS: list[str] = ["race", "gender", "age"]
_LINKAGE_RISK_THRESHOLD = 0.05


def _init_val_cfg() -> None:
    """Refresh module-level thresholds from the active (domain-merged) config."""
    global TOLERANCE, K_ANONYMITY_THRESHOLD, QUASI_ID_COLS
    global _CARDINALITY_RATIO_THRESHOLD, _CORR_DRIFT_THRESHOLD, _LEAKAGE_RATE_THRESHOLD
    global _DATETIME_CHECKS, _LLM_MODEL, _LLM_MAX_TOKENS, _RUN_PRIVACY_TESTS
    global _MIA_SAMPLE_SIZE, _MIA_AUC_THRESHOLD, _DCR_RATIO_THRESHOLD
    global _NNDR_THRESHOLD, _LINKAGE_QI_COLS, _LINKAGE_RISK_THRESHOLD

    val_cfg  = load_config().get("validation", {})
    priv_cfg = val_cfg.get("privacy_tests", {})
    cp_cfg   = load_config().get("clinical_plausibility", {})

    TOLERANCE             = val_cfg.get("tolerance",             TOLERANCE)
    K_ANONYMITY_THRESHOLD = val_cfg.get("k_anonymity_threshold", K_ANONYMITY_THRESHOLD)
    QUASI_ID_COLS         = val_cfg.get("quasi_id_cols",         QUASI_ID_COLS)

    _CARDINALITY_RATIO_THRESHOLD = val_cfg.get("cardinality_ratio_threshold", _CARDINALITY_RATIO_THRESHOLD)
    _CORR_DRIFT_THRESHOLD        = val_cfg.get("correlation_drift_threshold", _CORR_DRIFT_THRESHOLD)
    _LEAKAGE_RATE_THRESHOLD      = val_cfg.get("leakage_rate_threshold",      _LEAKAGE_RATE_THRESHOLD)
    _DATETIME_CHECKS             = cp_cfg.get("datetime_checks",              _DATETIME_CHECKS)
    _LLM_MODEL                   = val_cfg.get("llm_model",                   _LLM_MODEL)
    _LLM_MAX_TOKENS              = val_cfg.get("llm_max_completion_tokens",   _LLM_MAX_TOKENS)
    _RUN_PRIVACY_TESTS           = val_cfg.get("run_privacy_tests",           _RUN_PRIVACY_TESTS)

    _MIA_SAMPLE_SIZE        = priv_cfg.get("mia_sample_size",        _MIA_SAMPLE_SIZE)
    _MIA_AUC_THRESHOLD      = priv_cfg.get("mia_auc_threshold",      _MIA_AUC_THRESHOLD)
    _DCR_RATIO_THRESHOLD    = priv_cfg.get("dcr_ratio_threshold",    _DCR_RATIO_THRESHOLD)
    _NNDR_THRESHOLD         = priv_cfg.get("nndr_threshold",         _NNDR_THRESHOLD)
    _LINKAGE_QI_COLS        = priv_cfg.get("qi_cols_linkage",        _LINKAGE_QI_COLS)
    _LINKAGE_RISK_THRESHOLD = priv_cfg.get("linkage_risk_threshold", _LINKAGE_RISK_THRESHOLD)


# ─────────────────────────────────────────────────────────────────────────────
# Programmatic validators (fast, deterministic)
# ─────────────────────────────────────────────────────────────────────────────

def _check_schema(df: pd.DataFrame, seed_stats: dict) -> list[dict]:
    """Validate column presence, types, and nullability."""
    issues = []
    expected_cols = seed_stats["meta"]["column_names"]
    logger.debug("Expected columns :%s", expected_cols)

    missing_cols = [c for c in expected_cols if c not in df.columns]
    extra_cols   = [c for c in df.columns if c not in expected_cols]
    logger.debug("Missing columns  :%s", missing_cols)
    logger.debug("Extra columns    :%s", extra_cols)

    if missing_cols:
        issues.append({"severity": "CRITICAL", "check": "schema/missing_columns",
                        "detail": f"Missing columns: {missing_cols}"})
    if extra_cols:
        issues.append({"severity": "LOW", "check": "schema/extra_columns",
                        "detail": f"Unexpected columns: {extra_cols}"})

    # Nullability
    for col in expected_cols:
        if col not in df.columns:
            continue
        null_rate = df[col].isna().mean()
        seed_missing = seed_stats["missing"].get(col, {}).get("rate", 0.0)
        if null_rate > seed_missing + TOLERANCE:
            issues.append({"severity": "HIGH", "check": f"nullability/{col}",
                            "detail": f"Null rate {null_rate:.3f} vs seed {seed_missing:.3f}"})
    return issues


def _check_row_count(df: pd.DataFrame, expected: int | None = None) -> list[dict]:
    """Flag if the synthetic row count does not match the expected value from config."""
    if expected is None:
        expected = load_config().get("validation", {}).get("expected_row_count")
    if expected is None:
        return []
    issues = []
    if len(df) != expected:
        issues.append({"severity": "HIGH", "check": "row_count",
                        "detail": f"Expected {expected:,} rows, got {len(df):,}"})
    return issues


def _check_uniqueness(df: pd.DataFrame) -> list[dict]:
    """Validate patient_id uniqueness and format (must be 32-char lowercase hex)."""
    issues = []
    if "patient_id" in df.columns:
        dup_rate = df["patient_id"].duplicated().mean()
        if dup_rate > 0:
            issues.append({"severity": "HIGH", "check": "uniqueness/patient_id",
                            "detail": f"{dup_rate*100:.2f}% duplicate patient_ids"})
        valid_ids = df["patient_id"].dropna()
        id_len_bad = (valid_ids.str.len() != 32).mean()
        if id_len_bad > 0:
            issues.append({"severity": "MEDIUM", "check": "format/patient_id_length",
                            "detail": f"{id_len_bad*100:.2f}% of IDs are not 32 chars"})
        hex_pattern = valid_ids.str.match(r"^[0-9a-f]{32}$")
        if not hex_pattern.all():
            issues.append({"severity": "MEDIUM", "check": "format/patient_id_hex",
                            "detail": f"{(~hex_pattern).sum()} IDs not valid lowercase hex"})
    return issues


def _check_categorical_distributions(df: pd.DataFrame, seed_stats: dict) -> list[dict]:
    """Check each categorical column for novel values, absent values, proportion drift, and cardinality."""
    issues = []
    for col in seed_stats["meta"]["categorical_columns"]:
        if col not in df.columns:
            continue
        seed_col = seed_stats["columns"][col]
        seed_freq = seed_col.get("frequency_table", {})
        if not seed_freq:
            continue

        synth_counts = df[col].value_counts(normalize=True)

        # Check for values not in seed
        seed_values = set(seed_freq.keys())
        synth_values = set(synth_counts.index.astype(str))
        novel_values = synth_values - seed_values
        if novel_values:
            issues.append({"severity": "HIGH", "check": f"distribution/{col}/novel_values",
                            "detail": f"Values not in seed schema: {novel_values}"})

        # Check missing seed values
        absent = seed_values - synth_values
        if absent:
            issues.append({"severity": "MEDIUM", "check": f"distribution/{col}/absent_values",
                            "detail": f"Seed values missing from synthetic: {absent}"})

        # Proportion drift for top categories
        drifted = []
        for val, seed_prop in seed_freq.items():
            synth_prop = float(synth_counts.get(val, 0))
            seed_p = seed_prop["proportion"]
            drift = abs(synth_prop - seed_p)
            if drift > TOLERANCE:
                drifted.append(f"{val}: synth={synth_prop:.3f}, seed={seed_p:.3f}, Δ={drift:.3f}")
        if drifted:
            issues.append({"severity": "MEDIUM", "check": f"distribution/{col}/proportion_drift",
                            "detail": "; ".join(drifted[:5])})

        # n_unique check
        synth_unique = len(synth_counts)
        seed_unique  = seed_col["n_unique"]
        if synth_unique > seed_unique * _CARDINALITY_RATIO_THRESHOLD:
            issues.append({"severity": "LOW", "check": f"distribution/{col}/cardinality",
                            "detail": f"Synthetic has {synth_unique} unique values vs seed {seed_unique}"})

    return issues


def _check_datetime_columns(df: pd.DataFrame, seed_stats: dict) -> list[dict]:
    """Validate date and year columns against configured range, day-of-month, and logical ordering rules."""
    issues = []

    datetime_checks = _DATETIME_CHECKS

    for col, rules in datetime_checks.items():
        if col not in df.columns:
            continue

        if rules.get("is_year"):
            years = pd.to_numeric(df[col], errors="coerce")
            out_of_range = ((years < rules["min"]) | (years > rules["max"])).sum()
            if out_of_range:
                issues.append({"severity": "HIGH", "check": f"datetime/{col}/range",
                                "detail": f"{out_of_range} records outside [{rules['min']}, {rules['max']}]"})
        else:
            dates = pd.to_datetime(df[col], errors="coerce")
            bad_parse = dates.isna().sum()
            if bad_parse:
                issues.append({"severity": "HIGH", "check": f"datetime/{col}/parse_error",
                                "detail": f"{bad_parse} dates could not be parsed"})
                continue

            out_of_range = ((dates < rules["min"]) | (dates > rules["max"])).sum()
            if out_of_range:
                issues.append({"severity": "MEDIUM", "check": f"datetime/{col}/range",
                                "detail": f"{out_of_range} dates outside expected range"})

            if rules.get("first_of_month"):
                bad = (dates.dt.day != 1).sum()
                if bad:
                    issues.append({"severity": "HIGH", "check": f"datetime/{col}/first_of_month",
                                    "detail": f"{bad} records where {col} is not 1st of month"})

            if rules.get("last_of_month"):
                next_month = dates + pd.offsets.MonthBegin(1)
                last_days  = next_month - pd.Timedelta(days=1)
                bad = (dates.dt.day != last_days.dt.day).sum()
                if bad:
                    issues.append({"severity": "HIGH", "check": f"datetime/{col}/last_of_month",
                                    "detail": f"{bad} records where {col} is not last day of month"})

    # date_end >= date_start
    if "date_start" in df.columns and "date_end" in df.columns:
        ds = pd.to_datetime(df["date_start"], errors="coerce")
        de = pd.to_datetime(df["date_end"],   errors="coerce")
        bad = (de < ds).sum()
        if bad:
            issues.append({"severity": "CRITICAL", "check": "datetime/end_before_start",
                            "detail": f"{bad} records where date_end < date_start"})

    # patient must be born before date_start year
    if "patient_year_of_birth" in df.columns and "date_start" in df.columns:
        birth = pd.to_numeric(df["patient_year_of_birth"], errors="coerce")
        ds    = pd.to_datetime(df["date_start"], errors="coerce").dt.year
        bad   = (birth > ds).sum()
        if bad:
            issues.append({"severity": "HIGH", "check": "datetime/birth_after_service",
                            "detail": f"{bad} records where birth year > date_start year"})

    return issues


def _check_geo_consistency(df: pd.DataFrame) -> list[dict]:
    """Validate zip3 prefix belongs to the assigned state using known US ZIP3 ranges."""
    issues = []
    if "patient_zip3" not in df.columns or "patient_state" not in df.columns:
        return issues

    # Major US state → valid zip3 prefix ranges (first digit groupings)
    STATE_ZIP3_RANGES = {
        "CT": range(60, 70), "MA": range(10, 28), "ME": range(39, 50),
        "NH": range(30, 39), "NJ": range(70, 90), "NY": range(100, 149),
        "RI": range(28, 30), "VT": range(50, 60),
        "DE": range(197, 200), "MD": range(206, 220), "PA": range(150, 196),
        "DC": range(200, 206),
        "AL": range(350, 370), "FL": range(320, 350), "GA": range(300, 320),
        "KY": range(400, 428), "MS": range(386, 400), "NC": range(270, 290),
        "SC": range(290, 300), "TN": range(370, 386), "VA": range(220, 246),
        "WV": range(246, 270),
        "IL": range(600, 630), "IN": range(460, 480), "MI": range(480, 500),
        "MN": range(550, 568), "OH": range(430, 460), "WI": range(530, 550),
        "AR": range(716, 730), "LA": range(700, 716), "MO": range(630, 660),
        "OK": range(730, 750), "TX": range(750, 800),
        "IA": range(500, 530), "KS": range(660, 680), "NE": range(680, 700),
        "SD": range(570, 580),
        "AZ": range(850, 866), "CA": range(900, 962), "CO": range(800, 816),
        "HI": range(967, 969), "ID": range(832, 840), "NV": range(889, 900),
        "OR": range(970, 980), "UT": range(840, 848), "WA": range(980, 995),
        "WY": range(820, 832),
    }

    zip3_str = df["patient_zip3"].astype(str).str.lstrip("0")
    zip3_numeric = pd.to_numeric(
        zip3_str.where(zip3_str != "", "0"), errors="coerce"
    ).fillna(-1).astype(int)

    valid_mask = pd.Series(True, index=df.index)
    for state, r in STATE_ZIP3_RANGES.items():
        state_mask = df["patient_state"] == state
        if state_mask.any():
            valid_mask[state_mask] = zip3_numeric[state_mask].between(r.start, r.stop - 1)

    bad_geo = (~valid_mask).sum()
    if bad_geo:
        rate = bad_geo / len(df)
        severity = "CRITICAL" if rate > 0.1 else "HIGH" if rate > 0.02 else "MEDIUM"
        issues.append({"severity": severity, "check": "geo/zip3_state_mismatch",
                        "detail": f"{bad_geo} ({rate*100:.2f}%) records with zip3 inconsistent with state"})
    return issues


def _quasi_id_groups(df: pd.DataFrame) -> tuple[list[str], "pd.Series | None"]:
    """Return available quasi-identifier columns and their group size Series, or None if fewer than 2 are present."""
    available = [c for c in QUASI_ID_COLS if c in df.columns]
    if len(available) < 2:
        return available, None
    return available, df.groupby(available).size()


def _check_k_anonymity(df: pd.DataFrame, group_sizes: "pd.Series | None" = None) -> list[dict]:
    """Flag quasi-identifier groups smaller than k, indicating re-identification risk."""
    issues = []
    available = [c for c in QUASI_ID_COLS if c in df.columns]
    if len(available) < 2:
        return issues

    if group_sizes is None:
        group_sizes = df.groupby(available).size()
    violations  = (group_sizes < K_ANONYMITY_THRESHOLD).sum()
    total       = len(group_sizes)
    rate        = violations / total

    if violations:
        severity = "CRITICAL" if rate > 0.20 else "HIGH" if rate > 0.05 else "MEDIUM"
        issues.append({
            "severity": severity,
            "check": "privacy/k_anonymity",
            "detail": (f"{violations}/{total} quasi-identifier groups ({rate*100:.1f}%) "
                       f"have fewer than k={K_ANONYMITY_THRESHOLD} members — "
                       f"re-identification risk. Columns: {available}")
        })
    return issues


def _check_correlations(df: pd.DataFrame, seed_stats: dict) -> list[dict]:
    """Compare Cramér's V for categorical column pairs between synthetic and seed; flag drift above threshold."""
    issues = []
    seed_cv = seed_stats.get("correlations", {}).get("categorical_cramers_v", {})
    if not seed_cv:
        return issues

    cat_cols = [c for c in seed_stats["meta"]["categorical_columns"] if c in df.columns]

    _corr_thresh = _CORR_DRIFT_THRESHOLD
    for c1 in cat_cols:
        for c2 in cat_cols:
            if c1 >= c2:
                continue
            seed_v = seed_cv.get(c1, {}).get(c2)
            if seed_v is None:
                continue

            ct   = pd.crosstab(df[c1], df[c2])
            chi2 = scipy_stats.chi2_contingency(ct, correction=False)[0]
            n    = ct.sum().sum()
            phi2 = chi2 / n
            r, k = ct.shape
            synth_v = float(np.sqrt(phi2 / min(k-1, r-1))) if min(k-1, r-1) > 0 else 0.0
            drift = abs(synth_v - seed_v)

            if drift > _corr_thresh:
                severity = "HIGH" if drift > _corr_thresh * 2 else "MEDIUM"
                issues.append({
                    "severity": severity,
                    "check": f"correlation/cramers_v/{c1}__{c2}",
                    "detail": (f"Cramér's V: synth={synth_v:.3f}, seed={seed_v:.3f}, "
                               f"drift={drift:.3f} — correlation structure degraded")
                })
    return issues


def _check_leakage(df: pd.DataFrame, seed_df: pd.DataFrame | None) -> list[dict]:
    """Check for direct row leakage from seed into synthetic data."""
    issues = []
    if seed_df is None:
        return issues

    # Exclude patient_id and any column whose synthetic values look like hash IDs
    # (the generate.py replace_patient_ids call overwrites column 0 with a hash)
    check_cols = [
        c for c in seed_df.columns
        if c != "patient_id"
        and c in df.columns
        and not _is_hash_column(df[c])
    ]
    _sep = "\x00"
    def _row_key(frame: pd.DataFrame) -> pd.Series:
        """Concatenate all check columns into a single null-byte-delimited string key per row."""
        result = frame[check_cols[0]].astype(str)
        for _col in check_cols[1:]:
            result = result + _sep + frame[_col].astype(str)
        return result
    seed_tuples  = set(_row_key(seed_df))
    synth_tuples = _row_key(df)

    leaked = synth_tuples.isin(seed_tuples).sum()
    rate   = leaked / len(df)

    # ── Compute expected coincidental match rate ──────────────────────────
    # In a low-dimensional dataset with a large seed, many synthetic rows will
    # coincidentally match seed rows by distribution alone (not memorization).
    # We estimate the coincidental baseline: what fraction of RANDOMLY SHUFFLED
    # synthetic rows would match the seed? We approximate by checking how many
    # of the synthetic's unique combinations exist in the seed.
    synth_unique   = set(synth_tuples.unique())
    seed_set_keys  = seed_tuples
    unique_matched = len(synth_unique & seed_set_keys)
    unique_total   = len(synth_unique)
    coincidental_coverage = unique_matched / max(unique_total, 1)

    _leakage_thresh = _LEAKAGE_RATE_THRESHOLD

    # Only flag if a significant share of RARE combinations are reproduced.
    # Rare = appears only once in seed. Reproducing common combinations is expected.
    seed_key_counts = pd.Series(list(seed_tuples)).value_counts() if False else None
    # Simplified: flag if unique combination coverage is suspiciously high AND
    # the synthetic has low diversity (few unique combos relative to row count).
    diversity_ratio = unique_total / max(len(df), 1)

    if rate > _leakage_thresh:
        if coincidental_coverage >= 0.99 and diversity_ratio < 0.05:
            severity = "MEDIUM"
            detail = (
                f"{leaked} ({rate*100:.2f}%) synthetic rows match seed rows, but "
                f"synthetic has only {unique_total} unique combinations ({diversity_ratio*100:.1f}% "
                f"diversity). This indicates mode collapse, not memorization — "
                f"all {unique_total} synthetic templates appear in the large seed by coincidence."
            )
        else:
            severity = "CRITICAL" if rate > _leakage_thresh * 10 else "HIGH"
            detail = (
                f"{leaked} ({rate*100:.2f}%) synthetic rows are exact matches of seed rows "
                f"(excluding patient_id). {unique_total} unique synthetic combinations, "
                f"{unique_matched} found in seed. Possible memorization."
            )
        issues.append({"severity": severity, "check": "leakage/row_overlap", "detail": detail})

    # ── Always flag mode collapse regardless of leakage rate ─────────────
    if diversity_ratio < 0.05:
        issues.append({
            "severity": "HIGH",
            "check": "leakage/mode_collapse",
            "detail": (
                f"Synthetic has only {unique_total} unique row combinations "
                f"for {len(df)} rows ({diversity_ratio*100:.1f}% diversity). "
                f"The LLM is repeating the same templates. Reduce batch size or "
                f"increase generation temperature."
            ),
        })
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Privacy tests: Membership Inference, Record Linkage, Nearest-Neighbor
# ─────────────────────────────────────────────────────────────────────────────

def _check_membership_inference(
    df:         pd.DataFrame,
    seed_df:    pd.DataFrame,
    seed_stats: dict,
) -> tuple[list[dict], dict]:
    """Membership inference attack (MIA).

    Splits seed into train/holdout, computes each record's nearest-neighbor
    Gower distance to synthetic data, trains a logistic classifier on that
    distance, and reports the AUC.

    AUC ≈ 0.5  → safe (synthetic looks the same to train and holdout records)
    AUC > threshold → attacker can distinguish training records from holdout
                      using proximity to the synthetic data — memorisation risk.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    cat_cols = [c for c in seed_stats["meta"].get("categorical_columns", [])
                if c in df.columns and c in seed_df.columns]
    num_cols = [c for c in seed_stats["meta"].get("numeric_columns", [])
                if c in df.columns and c in seed_df.columns]
    if not cat_cols and not num_cols:
        return [], {"note": "No usable columns — skipped."}

    rng      = np.random.default_rng(42)
    n_sample = min(_MIA_SAMPLE_SIZE, len(seed_df) // 2)
    idx      = rng.permutation(len(seed_df))
    train_df   = seed_df.iloc[idx[:n_sample]          ].reset_index(drop=True)
    holdout_df = seed_df.iloc[idx[n_sample:n_sample*2]].reset_index(drop=True)
    synth_s    = df.sample(n=min(_MIA_SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

    # Shared encoding: query = train + holdout, ref = synth
    all_query = pd.concat([train_df, holdout_df], ignore_index=True)
    enc_query, enc_synth, col_ranges, is_cat = _gower_encode(all_query, synth_s, cat_cols, num_cols)
    enc_train_q   = enc_query[:len(train_df)]
    enc_holdout_q = enc_query[len(train_df):]

    d_train   = _gower_nn_dist(enc_train_q,   enc_synth, col_ranges, is_cat)
    d_holdout = _gower_nn_dist(enc_holdout_q, enc_synth, col_ranges, is_cat)

    X = np.concatenate([d_train, d_holdout]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(d_train)), np.zeros(len(d_holdout))])
    clf = LogisticRegression(max_iter=200).fit(X, y)
    auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))

    metrics = {
        "auc":               round(auc, 4),
        "train_mean_dist":   round(float(d_train.mean()),   4),
        "holdout_mean_dist": round(float(d_holdout.mean()), 4),
        "n_train_sample":    len(train_df),
        "n_holdout_sample":  len(holdout_df),
    }

    if auc > _MIA_AUC_THRESHOLD:
        severity = "CRITICAL" if auc > 0.75 else "HIGH"
        detail   = (
            f"MIA AUC={auc:.3f} (threshold ≤{_MIA_AUC_THRESHOLD}) — "
            f"an attacker can distinguish training records from holdout using "
            f"nearest-synthetic distances. "
            f"Train mean dist={d_train.mean():.4f}, holdout={d_holdout.mean():.4f}."
        )
    else:
        severity = "LOW"
        detail   = (
            f"MIA AUC={auc:.3f} — within safe range (≤{_MIA_AUC_THRESHOLD}). "
            f"Train mean dist={d_train.mean():.4f}, holdout={d_holdout.mean():.4f}."
        )
    return [{"severity": severity, "check": "privacy/membership_inference", "detail": detail}], metrics


def _check_record_linkage(
    df:      pd.DataFrame,
    seed_df: pd.DataFrame,
) -> tuple[list[dict], dict]:
    """Record linkage risk via quasi-identifier join.

    For each synthetic record, counts how many seed records share the same
    quasi-identifier combination (race, gender, age by default).

    unique_linkage_rate — fraction of synthetic records whose QI combo maps to
    exactly one seed record; these are trivially re-identifiable.
    no_match_rate       — fraction whose QI combo never appeared in the seed
                          (novel combinations = good for privacy).
    """
    qi_cols = [c for c in _LINKAGE_QI_COLS if c in df.columns and c in seed_df.columns]
    if not qi_cols:
        return [], {"note": f"No QI columns found from {_LINKAGE_QI_COLS} — skipped."}

    seed_groups = seed_df.groupby(qi_cols).size().reset_index(name="seed_count")
    merged      = df[qi_cols].merge(seed_groups, on=qi_cols, how="left")

    unique_link_rate = float((merged["seed_count"] == 1).mean())
    no_match_rate    = float(merged["seed_count"].isna().mean())
    mean_group       = float(merged["seed_count"].mean()) if not merged["seed_count"].isna().all() else 0.0
    pct_le5          = float((merged["seed_count"] <= 5).dropna().mean()) if not merged["seed_count"].isna().all() else 0.0

    metrics = {
        "qi_cols":              qi_cols,
        "unique_linkage_rate":  round(unique_link_rate, 4),
        "no_match_rate":        round(no_match_rate,    4),
        "mean_seed_group_size": round(mean_group,       2),
        "pct_linkable_to_le5":  round(pct_le5,          4),
    }

    if unique_link_rate > _LINKAGE_RISK_THRESHOLD:
        severity = "HIGH" if unique_link_rate > 0.10 else "MEDIUM"
        detail   = (
            f"{unique_link_rate*100:.1f}% of synthetic records link uniquely to one seed record "
            f"via {qi_cols} (threshold {_LINKAGE_RISK_THRESHOLD*100:.0f}%) — re-identification risk. "
            f"Mean seed group size: {mean_group:.1f}."
        )
    else:
        severity = "LOW"
        detail   = (
            f"Unique linkage rate {unique_link_rate*100:.1f}% (threshold {_LINKAGE_RISK_THRESHOLD*100:.0f}%) "
            f"via {qi_cols}. No-match rate {no_match_rate*100:.1f}% (novel QI combos)."
        )
    return [{"severity": severity, "check": "privacy/record_linkage", "detail": detail}], metrics


def _check_nn_similarity(
    df:         pd.DataFrame,
    seed_df:    pd.DataFrame,
    seed_stats: dict,
) -> tuple[list[dict], dict]:
    """Nearest-neighbor distance metrics: DCR and NNDR.

    DCR (Distance to Closest Record)
        Mean Gower distance from each synthetic row to its nearest seed row.
        Compared against a holdout baseline: if dcr_ratio = dcr_synth / dcr_holdout < threshold,
        synthetic data is suspiciously close to seed — possible memorisation.

    NNDR (Nearest-Neighbor Distance Ratio)
        For each synthetic row: dist_to_nearest_seed / dist_to_nearest_other_synthetic.
        Low ratio → the record blends into the real data more than into the synthetic set.
        NNDR mean < threshold flags re-identification risk.
    """
    cat_cols = [c for c in seed_stats["meta"].get("categorical_columns", [])
                if c in df.columns and c in seed_df.columns]
    num_cols = [c for c in seed_stats["meta"].get("numeric_columns", [])
                if c in df.columns and c in seed_df.columns]
    if not cat_cols and not num_cols:
        return [], {"note": "No usable columns — skipped."}

    rng      = np.random.default_rng(42)
    n_sample = min(_MIA_SAMPLE_SIZE, len(seed_df) // 2)
    idx      = rng.permutation(len(seed_df))
    train_s   = seed_df.iloc[idx[:n_sample]          ].reset_index(drop=True)
    holdout_s = seed_df.iloc[idx[n_sample:n_sample*2]].reset_index(drop=True)
    synth_s   = df.sample(n=min(_MIA_SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

    # DCR: synth → seed_train  vs  holdout → seed_train  (shared encoding)
    all_query = pd.concat([synth_s, holdout_s], ignore_index=True)
    enc_query, enc_train, col_ranges, is_cat = _gower_encode(all_query, train_s, cat_cols, num_cols)
    enc_synth_q   = enc_query[:len(synth_s)]
    enc_holdout_q = enc_query[len(synth_s):]

    dcr_synth   = _gower_nn_dist(enc_synth_q,   enc_train, col_ranges, is_cat)
    dcr_holdout = _gower_nn_dist(enc_holdout_q, enc_train, col_ranges, is_cat)
    dcr_ratio   = float(dcr_synth.mean()) / max(float(dcr_holdout.mean()), 1e-9)

    # NNDR: synth-to-synth distances on a smaller sub-sample to bound memory
    nn_n    = min(500, len(synth_s))
    synth_nn = synth_s.iloc[:nn_n].reset_index(drop=True)
    enc_a, enc_b, cr_s, ic_s = _gower_encode(synth_nn, synth_nn, cat_cols, num_cols)

    # Full (nn_n × nn_n) distance matrix; exclude self by setting diagonal to inf
    n_cols = enc_a.shape[1]
    dist_ss = np.zeros((nn_n, nn_n), dtype=np.float32)
    for j in range(n_cols):
        if ic_s[j]:
            dist_ss += (enc_a[:, j:j+1] != enc_b[:, j].reshape(1, -1)).astype(np.float32)
        else:
            dist_ss += np.abs(enc_a[:, j:j+1] - enc_b[:, j].reshape(1, -1)) / cr_s[j]
    dist_ss /= n_cols
    np.fill_diagonal(dist_ss, np.inf)
    nn_synth_dist = dist_ss.min(axis=1)

    nndr = dcr_synth[:nn_n] / np.maximum(nn_synth_dist, 1e-9)

    metrics = {
        "dcr_synth_mean":    round(float(dcr_synth.mean()),            4),
        "dcr_holdout_mean":  round(float(dcr_holdout.mean()),          4),
        "dcr_ratio":         round(dcr_ratio,                          4),
        "dcr_p5":            round(float(np.percentile(dcr_synth,  5)), 4),
        "dcr_p50":           round(float(np.percentile(dcr_synth, 50)), 4),
        "pct_dcr_below_0_1": round(float((dcr_synth < 0.1).mean()),    4),
        "nndr_mean":         round(float(nndr.mean()),                  4),
        "nndr_p5":           round(float(np.percentile(nndr,  5)),      4),
        "nndr_p50":          round(float(np.percentile(nndr, 50)),      4),
    }

    issues = []
    if dcr_ratio < _DCR_RATIO_THRESHOLD:
        severity = "HIGH" if dcr_ratio < 0.5 else "MEDIUM"
        issues.append({
            "severity": severity,
            "check":    "privacy/nn_similarity_dcr",
            "detail":   (
                f"DCR ratio={dcr_ratio:.3f} (threshold >{_DCR_RATIO_THRESHOLD}) — "
                f"synthetic is systematically closer to seed than holdout is "
                f"(synth mean={dcr_synth.mean():.4f}, holdout mean={dcr_holdout.mean():.4f}). "
                f"Possible memorisation."
            ),
        })
    else:
        issues.append({
            "severity": "LOW",
            "check":    "privacy/nn_similarity_dcr",
            "detail":   (
                f"DCR ratio={dcr_ratio:.3f} — safe range (>{_DCR_RATIO_THRESHOLD}). "
                f"Synth mean dist={dcr_synth.mean():.4f}, holdout={dcr_holdout.mean():.4f}."
            ),
        })

    if nndr.mean() < _NNDR_THRESHOLD:
        issues.append({
            "severity": "HIGH",
            "check":    "privacy/nn_similarity_nndr",
            "detail":   (
                f"NNDR mean={nndr.mean():.3f} (threshold >{_NNDR_THRESHOLD}) — "
                f"synthetic records are much closer to real records than to other synthetic records. "
                f"Re-identification risk elevated."
            ),
        })
    else:
        issues.append({
            "severity": "LOW",
            "check":    "privacy/nn_similarity_nndr",
            "detail":   (
                f"NNDR mean={nndr.mean():.3f} — safe range (>{_NNDR_THRESHOLD})."
            ),
        })

    return issues, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Residual Risk Assessment
# ─────────────────────────────────────────────────────────────────────────────

def _compute_residual_risk(
    mia_metrics: dict,
    rl_metrics:  dict,
    nn_metrics:  dict,
    df:          "pd.DataFrame",
    seed_df:     "pd.DataFrame",
    seed_stats:  dict,
) -> dict:
    """Derive composite Identity, Attribute, and Linkage disclosure risk scores
    from existing privacy metrics plus a lightweight attribute inference test.

    Each dimension is scored 0–1 (0 = no risk, 1 = maximum risk) and mapped to
    LOW / MEDIUM / HIGH.  An overall residual risk level is the max of the three.
    """
    def _band(score: float) -> str:
        if score < 0.33: return "LOW"
        if score < 0.66: return "MEDIUM"
        return "HIGH"

    # ── Identity Disclosure ───────────────────────────────────────────────────
    # Sources: MIA AUC (0.5–1.0 range mapped to 0–1), DCR ratio (inverted),
    #          NNDR (values ≤ 0.2 indicate near-duplicates).
    mia_auc   = mia_metrics.get("auc", 0.5)
    dcr_ratio = nn_metrics.get("dcr_ratio", 1.0)
    nndr_mean = min(nn_metrics.get("nndr_mean", 1.0), 10.0)  # cap at 10

    mia_risk  = max(0.0, (mia_auc - 0.5) / 0.5)          # 0 at AUC=0.5, 1 at AUC=1.0
    dcr_risk  = max(0.0, 1.0 - dcr_ratio / 0.8)           # 0 when ratio≥0.8, rises below
    nndr_risk = max(0.0, 1.0 - nndr_mean / 0.2)           # 0 when nndr≥0.2
    identity_score = round(float(0.5 * mia_risk + 0.3 * dcr_risk + 0.2 * nndr_risk), 4)

    # ── Linkage Risk ──────────────────────────────────────────────────────────
    # Sources: unique linkage rate (direct risk), pct linkable to ≤5 records,
    #          mean group size (small groups = higher risk).
    ulr    = rl_metrics.get("unique_linkage_rate", 0.0)
    pct_le5 = rl_metrics.get("pct_linkable_to_le5", 0.0)
    mgs    = rl_metrics.get("mean_seed_group_size", 1000.0)
    mgs_risk = max(0.0, 1.0 - min(mgs, 100.0) / 100.0)   # risk rises for small groups
    linkage_score = round(float(0.5 * ulr + 0.3 * pct_le5 + 0.2 * mgs_risk), 4)

    # ── Attribute Disclosure ─────────────────────────────────────────────────
    # Measures how predictable sensitive columns are from QI columns in the
    # synthetic data vs the seed.  Uses normalised mutual information (NMI):
    #   NMI(sensitive | QI) on synthetic vs seed.
    # If synthetic NMI > seed NMI the synthetic leaks more attribute information.
    qi_cols        = rl_metrics.get("qi_cols", [])
    sensitive_cols = seed_stats.get("meta", {}).get("categorical_columns", [])
    sensitive_cols = [c for c in sensitive_cols if c not in qi_cols
                      and c in df.columns and c in seed_df.columns]

    attr_risks: list[float] = []
    attr_details: dict      = {}

    for col in sensitive_cols:
        try:
            from sklearn.metrics import normalized_mutual_info_score as nmi
            qi_avail = [q for q in qi_cols if q in df.columns and q in seed_df.columns]
            if not qi_avail:
                continue
            # Encode QI as a single composite string for NMI computation
            seed_qi  = seed_df[qi_avail].astype(str).agg("-".join, axis=1)
            synth_qi = df[qi_avail].astype(str).agg("-".join, axis=1)
            seed_nmi  = nmi(seed_df[col].astype(str),  seed_qi,  average_method="arithmetic")
            synth_nmi = nmi(df[col].astype(str),        synth_qi, average_method="arithmetic")
            # Risk: how much *more* predictable in synthetic vs seed (clamped to 0–1)
            excess = max(0.0, synth_nmi - seed_nmi)
            risk   = min(1.0, excess / max(seed_nmi, 0.01))
            attr_risks.append(risk)
            attr_details[col] = {
                "seed_nmi":  round(seed_nmi,  4),
                "synth_nmi": round(synth_nmi, 4),
                "excess":    round(excess,    4),
                "risk":      round(risk,      4),
            }
        except Exception:
            continue

    attribute_score = round(float(sum(attr_risks) / len(attr_risks)) if attr_risks else 0.0, 4)

    # ── Overall ───────────────────────────────────────────────────────────────
    overall_score = round(max(identity_score, linkage_score, attribute_score), 4)

    return {
        "identity_disclosure": {
            "score":       identity_score,
            "level":       _band(identity_score),
            "components":  {
                "mia_risk":  round(mia_risk,  4),
                "dcr_risk":  round(dcr_risk,  4),
                "nndr_risk": round(nndr_risk, 4),
            },
        },
        "attribute_disclosure": {
            "score":        attribute_score,
            "level":        _band(attribute_score),
            "per_column":   attr_details,
        },
        "linkage_risk": {
            "score":      linkage_score,
            "level":      _band(linkage_score),
            "components": {
                "unique_linkage_rate": round(ulr,      4),
                "pct_linkable_to_le5": round(pct_le5,  4),
                "group_size_risk":     round(mgs_risk, 4),
            },
        },
        "overall": {
            "score": overall_score,
            "level": _band(overall_score),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM-based deep analysis (GPT-4o-mini)
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm_analysis_prompt(
    programmatic_issues: list[dict],
    synth_profile: dict,
    seed_stats: dict,
) -> str:
    """Build the prompt sent to GPT-4o-mini for deep analysis, combining seed stats, synthetic profile, and programmatic issues."""
    return f"""You are a data quality auditor specialising in synthetic healthcare data.

You will be given:
1. SEED_STATS: Statistical profile of the original seed data
2. SYNTH_PROFILE: Statistical profile of the generated synthetic data
3. PROGRAMMATIC_ISSUES: Issues already detected by deterministic checks

Your task is to:
A) Identify additional risks or quality issues NOT already listed in PROGRAMMATIC_ISSUES
B) Score each existing issue by clinical / privacy impact
C) Provide a concise, prioritised improvement strategy

Return your response as valid JSON with this exact structure:
{{
  "overall_quality_score": <integer 0-100>,
  "overall_verdict": "<PASS|WARN|FAIL>",
  "executive_summary": "<2-3 sentence summary>",
  "additional_issues": [
    {{
      "severity": "<CRITICAL|HIGH|MEDIUM|LOW>",
      "category": "<distribution|privacy|schema|temporal|correlation|leakage|other>",
      "issue": "<concise title>",
      "detail": "<explanation>",
      "impact": "<clinical or business impact>"
    }}
  ],
  "improvement_strategies": [
    {{
      "priority": <1-10>,
      "strategy": "<title>",
      "rationale": "<why this matters>",
      "implementation": "<how to fix it — concrete steps>",
      "effort": "<low|medium|high>"
    }}
  ],
  "privacy_risk_assessment": {{
    "overall_risk": "<low|medium|high|critical>",
    "re_identification_risk": "<assessment>",
    "data_leakage_risk": "<assessment>",
    "recommendations": ["<rec1>", "<rec2>"]
  }}
}}

=== SEED_STATS ===
{json.dumps(seed_stats, indent=2, default=str)}

=== SYNTH_PROFILE ===
{json.dumps(synth_profile, indent=2, default=str)}

=== PROGRAMMATIC_ISSUES ===
{json.dumps(programmatic_issues, indent=2)}

Return ONLY the JSON object. No preamble, no markdown fences.
"""


def _run_llm_analysis(
    programmatic_issues: list[dict],
    synth_profile: dict,
    seed_stats: dict,
) -> dict:
    """Call GPT-4o-mini to perform deep analysis and return structured JSON."""
    prompt = _build_llm_analysis_prompt(programmatic_issues, synth_profile, seed_stats)

    response = _openai_client.chat.completions.create(
        model=_LLM_MODEL,
        max_completion_tokens=_LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$",     "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {"error": f"LLM returned invalid JSON: {e}", "raw_response": raw}


# ─────────────────────────────────────────────────────────────────────────────
# Profile builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_synth_profile(df: pd.DataFrame, seed_stats: dict, group_sizes: "pd.Series | None" = None) -> dict:
    """Build a compact statistical profile of the synthetic data to feed the LLM."""
    profile: dict = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "missing_rates": df.isnull().mean().round(4).to_dict(),
        "columns": {},
    }

    for col in seed_stats["meta"]["categorical_columns"]:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(normalize=True)
        profile["columns"][col] = {
            "type": "categorical",
            "n_unique": len(vc),
            "top_proportions": {str(k): round(float(v), 4) for k, v in vc.head(10).items()},
        }

    for col in ["patient_year_of_birth"]:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        profile["columns"][col] = {
            "type": "year",
            "min": int(vals.min()), "max": int(vals.max()),
            "mean": round(float(vals.mean()), 1),
            "median": round(float(vals.median()), 1),
            "std": round(float(vals.std()), 2),
        }

    for col in ["date_start", "date_end"]:
        if col not in df.columns:
            continue
        dates = pd.to_datetime(df[col], errors="coerce").dropna()
        profile["columns"][col] = {
            "type": "datetime",
            "min": str(dates.min().date()),
            "max": str(dates.max().date()),
            "median": str(dates.median().date()),
            "n_unique": int(dates.nunique()),
        }

    # Coverage duration distribution
    if "date_start" in df.columns and "date_end" in df.columns:
        ds = pd.to_datetime(df["date_start"], errors="coerce")
        de = pd.to_datetime(df["date_end"],   errors="coerce")
        dur_days = (de - ds).dt.days.dropna()
        profile["coverage_duration_days"] = {
            "min": round(float(dur_days.min()), 1),
            "max": round(float(dur_days.max()), 1),
            "mean": round(float(dur_days.mean()), 1),
            "median": round(float(dur_days.median()), 1),
            "p25": round(float(dur_days.quantile(0.25)), 1),
            "p75": round(float(dur_days.quantile(0.75)), 1),
        }

    # Cramér's V for key pairs
    cat_cols = [c for c in seed_stats["meta"]["categorical_columns"] if c in df.columns]
    cramers = {}
    key_pairs = [("patient_zip3", "patient_state"),
                 ("patient_zip3", "pay_type"),
                 ("patient_state", "pay_type")]
    for c1, c2 in key_pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        ct = pd.crosstab(df[c1], df[c2])
        chi2 = scipy_stats.chi2_contingency(ct, correction=False)[0]
        n    = ct.sum().sum()
        phi2 = chi2 / n
        r, k = ct.shape
        v = float(np.sqrt(phi2 / min(k-1, r-1))) if min(k-1, r-1) > 0 else 0.0
        cramers[f"{c1}__{c2}"] = round(v, 4)
    profile["cramers_v_key_pairs"] = cramers

    # K-anonymity summary
    available_qi = [c for c in QUASI_ID_COLS if c in df.columns]
    if len(available_qi) >= 2:
        if group_sizes is None:
            group_sizes = df.groupby(available_qi).size()
        profile["k_anonymity"] = {
            "quasi_id_cols": available_qi,
            "n_groups": len(group_sizes),
            "min_group_size": int(group_sizes.min()),
            "pct_below_k5": round(float((group_sizes < 5).mean() * 100), 2),
            "pct_below_k11": round(float((group_sizes < 11).mean() * 100), 2),
        }

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Plausibility HTML report builder
# ─────────────────────────────────────────────────────────────────────────────

def _sev_badge(sev: str) -> str:
    col = {"CRITICAL": "#c0392b", "HIGH": "#e67e22", "MEDIUM": "#f1c40f", "LOW": "#27ae60"}.get(sev, "#95a5a6")
    return f'<span style="background:{col};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8em;font-weight:bold">{sev}</span>'


def _issues_table_html(issues: list[dict]) -> str:
    if not issues:
        return '<p style="color:#27ae60">No issues found.</p>'
    rows = "".join(
        f"<tr><td>{_sev_badge(i['severity'])}</td>"
        f"<td style='font-family:monospace;font-size:0.85em'>{i['check']}</td>"
        f"<td>{i['detail']}</td></tr>"
        for i in issues
    )
    return (
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:4px 8px'>Severity</th>"
        "<th style='text-align:left;padding:4px 8px'>Check</th>"
        "<th style='text-align:left;padding:4px 8px'>Detail</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _metrics_table_html(metrics: dict) -> str:
    rows = "".join(
        f"<tr><td style='font-family:monospace;font-size:0.85em'>{k}</td>"
        f"<td style='text-align:right'>{v}</td></tr>"
        for k, v in metrics.items()
    )
    return (
        "<table style='border-collapse:collapse'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:4px 8px'>Metric</th>"
        "<th style='text-align:right;padding:4px 8px'>Value</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _coverage_table_html(records: list[dict]) -> str:
    if not records:
        return "<p>No coverage data.</p>"
    cols = ["rule_id", "rule_description", "times_triggered", "times_passed", "times_failed"]
    header = "".join(f"<th style='text-align:left;padding:4px 8px'>{c}</th>" for c in cols)
    header += "<th style='text-align:left;padding:4px 8px'>Pass rate</th>"
    rows = ""
    for r in records:
        trig = r.get("times_triggered", 0)
        passed = r.get("times_passed", 0)
        failed = r.get("times_failed", 0)
        pass_rate = f"{passed/trig*100:.0f}%" if trig > 0 else "—"
        fail_col = "#c0392b" if failed > 0 else "#27ae60"
        cells = "".join(f"<td style='padding:4px 8px'>{r.get(c,'')}</td>" for c in cols)
        rows += f"<tr>{cells}<td style='padding:4px 8px;color:{fail_col}'>{pass_rate}</td></tr>"
    return (
        "<table style='width:100%;border-collapse:collapse'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _build_plausibility_html(plausibility_data: dict) -> str:
    """Render the clinical plausibility report as a self-contained HTML string."""
    ts  = plausibility_data.get("generated_at", "")
    n   = plausibility_data.get("n_rows", "?")
    cp  = plausibility_data.get("clinical_plausibility",   {})
    rb  = plausibility_data.get("rule_based_plausibility", {})
    cov = plausibility_data.get("rule_coverage",           {})

    score      = rb.get("metrics", {}).get("plausibility_score", 100.0)
    score_col  = "#27ae60" if float(score) >= 90 else "#e67e22" if float(score) >= 70 else "#c0392b"
    all_issues = cp.get("issues", []) + rb.get("issues", [])
    sev_counts = {s: sum(1 for i in all_issues if i["severity"] == s)
                  for s in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]}

    scorecard = (
        "<div style='display:flex;gap:2em;flex-wrap:wrap;margin-bottom:1.5em'>"
        + "".join(
            f"<div style='padding:12px 20px;border-radius:8px;background:#ecf0f1;"
            f"min-width:120px;text-align:center'>"
            f"<div style='font-size:2em;font-weight:bold;color:{c}'>{v}</div>"
            f"<div style='font-size:0.8em;color:#555'>{k}</div></div>"
            for k, v, c in [
                ("Rows tested",    n,                             "#2c3e50"),
                ("Total issues",   len(all_issues),              "#e74c3c" if all_issues else "#27ae60"),
                ("CRITICAL",       sev_counts["CRITICAL"],       "#c0392b"),
                ("HIGH",           sev_counts["HIGH"],           "#e67e22"),
                ("MEDIUM",         sev_counts["MEDIUM"],         "#f39c12"),
                ("LOW",            sev_counts["LOW"],            "#27ae60"),
                ("Plausibility %", score,                        score_col),
            ]
        )
        + "</div>"
    )

    rb_m = rb.get("metrics", {})
    rb_summary = (
        f"<p><b>Rules evaluated:</b> {rb_m.get('rules_evaluated','?')} &nbsp;|&nbsp; "
        f"<b>Rules with violations:</b> {rb_m.get('rules_with_violations','?')} &nbsp;|&nbsp; "
        f"<b>Plausibility score:</b> "
        f"<span style='font-size:1.2em;font-weight:bold;color:{score_col}'>{score}%</span></p>"
    )

    if cov.get("skipped"):
        cov_body = (
            f'<p style="color:#e67e22">⚠ {cov.get("reason","")}</p>'
            "<p>Populate the domain's <code>_clin_hard_rules.csv</code> to enable this check.</p>"
        )
    else:
        cov_body = (
            f"<p><b>Rows passed:</b> {cov.get('n_passed_rows',0):,} &nbsp;|&nbsp; "
            f"<b>Rows failed:</b> {cov.get('n_failed_rows',0):,} &nbsp;|&nbsp; "
            f"<b>Pass rate:</b> {(cov.get('pass_rate') or 0)*100:.1f}%</p>"
            + _coverage_table_html(cov.get("coverage_table", []))
        )

    def _section(title: str, body: str, anchor: str = "") -> str:
        return (
            f'<section id="{anchor}" style="margin-bottom:2em">'
            f'<h2 style="border-bottom:2px solid #3498db;padding-bottom:4px">{title}</h2>'
            f"{body}</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Clinical Plausibility Report</title>
<style>
  body  {{ font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#2c3e50 }}
  table {{ border-collapse:collapse;width:100% }}
  th,td {{ border:1px solid #ddd;padding:6px 10px;vertical-align:top }}
  th    {{ background:#f2f2f2 }}
  tr:nth-child(even) {{ background:#fafafa }}
  h1    {{ color:#2c3e50 }}
  code  {{ background:#f0f0f0;padding:1px 4px;border-radius:3px }}
</style>
</head>
<body>
<h1>Clinical Plausibility Report</h1>
<p style="color:#777">Generated: {ts}</p>
<nav style="background:#2c3e50;padding:10px 20px;margin-bottom:2em;border-radius:6px">
  <a href="#clinical"  style="color:#ecf0f1;margin-right:16px">Clinical Plausibility</a>
  <a href="#rule-based" style="color:#ecf0f1;margin-right:16px">Rule-Based</a>
  <a href="#coverage"  style="color:#ecf0f1">Association Rule Coverage</a>
</nav>
{scorecard}
{_section("1. Clinical Plausibility",
          _issues_table_html(cp.get("issues", [])) + "<br>" + _metrics_table_html(cp.get("metrics", {})),
          "clinical")}
{_section("2. Rule-Based Clinical Plausibility",
          rb_summary + _issues_table_html(rb.get("issues", [])) + "<br>" + _metrics_table_html(rb_m),
          "rule-based")}
{_section("3. Association-Rule Coverage", cov_body, "coverage")}
</body>
</html>"""


def _write_plausibility_report(plausibility_data: dict, out_dir: Path) -> None:
    """Write plausibility_report.json and plausibility_report.html to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "plausibility_report.json"
    html_path = out_dir / "plausibility_report.html"

    with open(json_path, "w") as f:
        json.dump(plausibility_data, f, indent=2, default=str)

    html_path.write_text(_build_plausibility_html(plausibility_data), encoding="utf-8")
    logger.info("Plausibility report → %s", html_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def validate_synthetic_data(
    synthetic_csv: str,
    seed_stats_json: str,
    seed_csv: str | None = None,
    output_report: str | None = None,
    run_llm: bool = True,
    run_privacy_tests: bool = True,
) -> dict:
    """
    Validate synthetic data against seed schema and statistics.

    Parameters
    ----------
    synthetic_csv    : Path to the generated synthetic CSV.
    seed_stats_json  : Path to the seed_statistics json produced by seed_statistics.py.
    seed_csv         : Optional path to original seed CSV (enables leakage detection).
    output_report    : Optional path to save the full JSON report.
    run_llm          : Whether to run GPT model deep analysis (requires API key).

    Returns
    -------
    dict with keys:
        programmatic_issues  : List of deterministic check results
        llm_analysis         : structured analysis
        synth_profile        : Statistical profile of the synthetic data
        summary              : Quick-read counts by severity
    """
    _init_val_cfg()

    # ── Load data ─────────────────────────────────────────────────────────
    syn_path   = Path(synthetic_csv)
    stats_path = Path(seed_stats_json)

    if not syn_path.exists():
        raise FileNotFoundError(f"Synthetic CSV not found: {synthetic_csv}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Seed stats JSON not found: {seed_stats_json}")

    logger.info("Loading synthetic data...")
    df = pd.read_csv(syn_path, dtype=str)   # load all as str to avoid coercion masking issues

    logger.info("Loading seed statistics...")
    with open(stats_path) as f:
        seed_stats = json.load(f)

    seed_df = None
    if seed_csv and Path(seed_csv).exists():
        logger.info("Loading seed CSV for leakage detection...however consider carefully what constititue leakage")
        seed_df = pd.read_csv(seed_csv, dtype=str, index_col=0)

    # ── Programmatic checks ───────────────────────────────────────────────
    logger.info("Running programmatic checks...")
    _qi_available, _qi_groups = _quasi_id_groups(df)
    issues: list[dict] = []
    issues += _check_schema(df, seed_stats)
    issues += _check_categorical_distributions(df, seed_stats)
    issues += _check_datetime_columns(df, seed_stats)
    issues += _check_correlations(df, seed_stats)
    clinical_issues, clinical_metrics = check_clinical_plausibility(df, seed_df)
    issues += clinical_issues
    rule_issues, rule_metrics = check_rule_based_clinicalplausibility(df)
    issues += rule_issues
    # issues += _check_uniqueness(df)
    # issues += _check_geo_consistency(df)
    # issues += _check_k_anonymity(df, _qi_groups)
    # issues += _check_leakage(df, seed_df)

    # ── Association-rule coverage ─────────────────────────────────────────
    logger.info("Running association-rule coverage check...")
    _domain_name = get_domain_name()
    _rules_csv   = _ROOT / "domains" / f"{_domain_name}_clin_hard_rules.csv" if _domain_name else None
    coverage_result: dict = {}
    if _rules_csv and _rules_csv.exists() and _rules_csv.stat().st_size > 0:
        _passed_df, _failed_queue, _coverage_df = generate_rule_coverage_report(
            str(syn_path), str(_rules_csv)
        )
        _n_pass = len(_passed_df)
        _n_fail = len(_failed_queue)
        coverage_result = {
            "n_passed_rows":  _n_pass,
            "n_failed_rows":  _n_fail,
            "pass_rate":      round(_n_pass / (_n_pass + _n_fail), 4) if (_n_pass + _n_fail) else None,
            "coverage_table": _coverage_df.to_dict(orient="records"),
        }
        logger.info("  → %d rows passed, %d failed", _n_pass, _n_fail)
    else:
        _reason = (
            f"Rules file not found or empty: {_rules_csv}"
            if _rules_csv else "No active domain — cannot locate rules file."
        )
        coverage_result = {"skipped": True, "reason": _reason}
        logger.info("  → Skipped. %s", _reason)

    # ── Privacy tests ─────────────────────────────────────────────────────
    privacy_metrics: dict = {}
    if seed_df is not None and run_privacy_tests and _RUN_PRIVACY_TESTS:
        logger.info("Running privacy tests (MIA, record linkage, nearest-neighbor similarity)...")
        mia_issues, mia_metrics = _check_membership_inference(df, seed_df, seed_stats)
        rl_issues,  rl_metrics  = _check_record_linkage(df, seed_df)
        nn_issues,  nn_metrics  = _check_nn_similarity(df, seed_df, seed_stats)
        issues += mia_issues + rl_issues + nn_issues
        residual_risk = _compute_residual_risk(
            mia_metrics, rl_metrics, nn_metrics, df, seed_df, seed_stats
        )
        privacy_metrics = {
            "membership_inference": mia_metrics,
            "record_linkage":       rl_metrics,
            "nn_similarity":        nn_metrics,
            "residual_risk":        residual_risk,
        }

    # ── Build synthetic profile ───────────────────────────────────────────
    logger.info("Building synthetic data profile...")
    synth_profile = _build_synth_profile(df, seed_stats, _qi_groups)

    # ── PSI ───────────────────────────────────────────────────────────────
    logger.info("Computing Population Stability Index (PSI)...")
    psi_report = compute_psi(df, seed_stats)

    # ── LLM analysis ─────────────────────────────────────────────────────
    llm_result = {}
    if run_llm:
        logger.info("Running GPT-4o-mini deep analysis...")
        llm_result = _run_llm_analysis(issues, synth_profile, seed_stats)

    # ── Summary counts ────────────────────────────────────────────────────
    severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    summary = {s: sum(1 for i in issues if i["severity"] == s) for s in severity_order}
    summary["total"] = len(issues)
    summary["llm_verdict"] = llm_result.get("overall_verdict", "N/A")
    summary["llm_score"]   = llm_result.get("overall_quality_score", "N/A")

    report = {
        "programmatic_issues":     issues,
        "llm_analysis":            llm_result,
        "synth_profile":           synth_profile,
        "psi":                     psi_report,
        "privacy_metrics":         privacy_metrics,
        "clinical_plausibility":   clinical_metrics,
        "rule_based_plausibility": rule_metrics,
        "rule_coverage":           coverage_result,
        "summary":                 summary,
    }

    if output_report:
        with open(output_report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Full report saved to %s", output_report)

    # ── Plausibility sub-report (HTML + JSON) ─────────────────────────────
    # Written to processed_data/<domain>/ so the dashboard can find it.
    _plaus_dir: Path | None = None
    if _domain_name:
        _plaus_dir = _ROOT / "processed_data" / _domain_name
    elif output_report:
        _plaus_dir = Path(output_report).parent
    if _plaus_dir:
        _plaus_data = {
            "generated_at":            datetime.now().isoformat(),
            "synthetic_csv":           str(syn_path),
            "n_rows":                  len(df),
            "clinical_plausibility":   {"issues": clinical_issues,  "metrics": clinical_metrics},
            "rule_based_plausibility": {"issues": rule_issues,       "metrics": rule_metrics},
            "rule_coverage":           coverage_result,
        }
        _write_plausibility_report(_plaus_data, _plaus_dir)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # if len(sys.argv) < 3:
    #     print("Usage: python3 validate.py <synthetic_patients.csv> <seed_stats.json> [seed.csv] [report.json]")
    #     sys.exit(1)

    # synthetic_csv    = sys.argv[1]
    # seed_stats_json  = sys.argv[2]
    # seed_csv         = sys.argv[3] if len(sys.argv) > 3 else None
    # output_report    = sys.argv[4] if len(sys.argv) > 4 else "validation_report.json"
    
    _val_cfg        = load_config().get("validation", {})
    seed_csv        = str(_ROOT / _val_cfg.get("seed_csv",        "data/diab_seed.csv"))
    seed_stats_json = str(_ROOT / _val_cfg.get("seed_stats_json", "data/diab_stats.json"))
    synthetic_csv   = str(_ROOT / _val_cfg.get("synthetic_csv",   "synthetic_diabetes6.csv"))
    output_report   = str(_ROOT / _val_cfg.get("output_report",   "diab_validation_report.json"))

    report = validate_synthetic_data(
        synthetic_csv=synthetic_csv,
        seed_stats_json=seed_stats_json,
        seed_csv=seed_csv,
        output_report=output_report,
        run_llm=_val_cfg.get("run_llm", True),
    )

    # plot generated data distribution vs seed stats for key columns
    # plot_synthetic_data(pd.read_csv(synthetic_csv), json.loads(Path(seed_stats_json).read_text()))

    logger.info("validation complete")
    _print_report(report, thresholds={
        "mia_auc":      _MIA_AUC_THRESHOLD,
        "linkage_risk": _LINKAGE_RISK_THRESHOLD,
        "dcr_ratio":    _DCR_RATIO_THRESHOLD,
        "nndr":         _NNDR_THRESHOLD,
    })