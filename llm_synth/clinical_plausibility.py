"""
clinical_plausibility.py
========================
Clinical plausibility checks for synthetic data.

All disease-specific rules are loaded from the active domain config
(clinical_plausibility.rules and clinical_plausibility.correlation_pairs).
See domains/diabetes.yaml for the schema and examples.

Exported
--------
check_clinical_plausibility(df, seed_df=None) -> (list[dict], dict)
check_rule_based_clinicalplausibility(df, rules=None) -> (list[dict], dict)
generate_rule_coverage_report(synthetic_data_path, rules_path)
    -> (passed_df, failed_rows_with_feedback, coverage_df)
"""

import ast
import logging

import numpy as np
import pandas as pd

from llm_synth.config import load_config

logger = logging.getLogger(__name__)


def _load_plausibility_cfg() -> tuple[dict, float, float, str, str, list, list, list]:
    """Load all plausibility settings from the domain config."""
    try:
        cfg = load_config().get("clinical_plausibility", {})
        raw_bounds  = cfg.get("column_bounds", {})
        bounds      = {col: tuple(v) for col, v in raw_bounds.items()}
        r30_range   = cfg.get("readmit_30_range",  [0.05, 0.30])
        readmit_col = cfg.get("readmit_30_column", "readmitted")
        readmit_val = cfg.get("readmit_30_value",  "<30")
        corr_pairs  = [tuple(p) for p in cfg.get("correlation_pairs", [])]
        rules       = cfg.get("rules", [])
        grp_rules   = cfg.get("group_comparison_rules", [])
        return (bounds, float(r30_range[0]), float(r30_range[1]),
                readmit_col, readmit_val, corr_pairs, rules, grp_rules)
    except FileNotFoundError:
        return {}, 0.05, 0.30, "readmitted", "<30", [], [], []

_CLINICAL_BOUNDS, _READMIT_30_MIN, _READMIT_30_MAX, _READMIT_COL, _READMIT_VAL, \
    _CORR_PAIRS, CLINICAL_RULES, _GRP_CMP_RULES = _load_plausibility_cfg()


def _resolve_group_mask(series: pd.Series, spec: str | dict) -> pd.Series:
    """Return a boolean mask for rows matching a group spec.

    spec can be:
    - a scalar  → equality match (series == spec)
    - a dict    → {operator: "!=" | "in" | "not_in", value: ...}
    """
    if isinstance(spec, dict):
        op    = spec.get("operator", "==")
        value = spec["value"]
        ops = {
            "==":     lambda s: s.astype(str) == str(value),
            "!=":     lambda s: s.astype(str) != str(value),
            "in":     lambda s: s.astype(str).isin([str(v) for v in value]),
            "not_in": lambda s: ~s.astype(str).isin([str(v) for v in value]),
        }
        if op not in ops:
            raise ValueError(f"Unsupported group operator: {op!r}. Use ==, !=, in, not_in.")
        return ops[op](series)
    return series.astype(str) == str(spec)


def _check_group_comparisons(df: pd.DataFrame, rules: list) -> tuple[list[dict], dict]:
    """Evaluate group comparison rules loaded from the domain config.

    Each rule compares an aggregate metric (mean or proportion) between two
    subgroups of a column and flags if the expected direction is violated.
    """
    def _num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    issues: list[dict] = []
    metrics: dict = {}

    for rule in rules:
        name        = rule.get("name", "unnamed")
        description = rule.get("description", "")
        severity    = rule.get("severity", "MEDIUM")
        grp_col     = rule.get("group_column")
        metric_col  = rule.get("metric_column")
        metric_type = rule.get("metric_type", "mean")
        prop_vals   = set(str(v) for v in rule.get("proportion_values", []))
        direction   = rule.get("expected_direction", ">")
        spec_a      = rule.get("group_a")
        spec_b      = rule.get("group_b")

        if grp_col not in df.columns or metric_col not in df.columns:
            continue

        mask_a = _resolve_group_mask(df[grp_col], spec_a)
        mask_b = _resolve_group_mask(df[grp_col], spec_b)

        if metric_type == "proportion":
            val_a = float(df.loc[mask_a, metric_col].astype(str).isin(prop_vals).mean()) if mask_a.any() else np.nan
            val_b = float(df.loc[mask_b, metric_col].astype(str).isin(prop_vals).mean()) if mask_b.any() else np.nan
        else:
            val_a = float(_num(df.loc[mask_a, metric_col]).mean()) if mask_a.any() else np.nan
            val_b = float(_num(df.loc[mask_b, metric_col]).mean()) if mask_b.any() else np.nan

        metrics[f"grp_{name}_a"] = round(val_a, 4) if not np.isnan(val_a) else None
        metrics[f"grp_{name}_b"] = round(val_b, 4) if not np.isnan(val_b) else None

        if np.isnan(val_a) or np.isnan(val_b):
            continue

        violated = {
            ">":  val_a <= val_b,
            ">=": val_a <  val_b,
            "<":  val_a >= val_b,
            "<=": val_a >  val_b,
        }.get(direction, False)

        if violated:
            def _label(spec):
                if isinstance(spec, dict):
                    return f"{spec.get('operator','==')} {spec['value']}"
                return str(spec)
            issues.append({
                "severity": severity,
                "check":    f"clinical/group_comparison/{name}",
                "detail":   (
                    f"{description} "
                    f"Group A ({grp_col} {_label(spec_a)}): {metric_col}={val_a:.4f}; "
                    f"Group B ({grp_col} {_label(spec_b)}): {metric_col}={val_b:.4f}. "
                    f"Expected: A {direction} B."
                ),
            })

    return issues, metrics


def check_clinical_plausibility(
    df: pd.DataFrame,
    seed_df: pd.DataFrame | None = None,
) -> tuple[list[dict], dict]:
    """
    Clinical plausibility checks driven by the active domain config.

    Checks
    ------
    1. Hard numeric bounds     — values outside clinically valid ranges
    2. Target prevalence rate  — fraction of rows matching the target value
                                 must fall within the configured range
    3. Group comparison rules  — aggregate metrics compared between subgroups
                                 (e.g. mean, proportion) per domain config
    4. Positive correlations   — clinically expected positive associations
                                 between numeric column pairs
    5. Seed drift (optional)   — flag if any check metric deviates more than
                                 20 % from the seed baseline

    Parameters
    ----------
    df       : Synthetic DataFrame (values may be str or numeric).
    seed_df  : Optional seed DataFrame for drift comparison.

    Returns
    -------
    issues  : list of dicts with keys severity / check / detail
    metrics : dict of computed scalar values for downstream reporting
    """
    issues: list[dict] = []
    metrics: dict = {}

    def _num(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    # ── 1. Hard numeric bounds ────────────────────────────────────────────────
    for col, (lo, hi) in _CLINICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        vals = _num(df[col]).dropna()
        if len(vals) == 0:
            continue
        n = len(vals)
        violations = pd.Series(False, index=vals.index)
        if lo is not None:
            violations |= vals < lo
        if hi is not None:
            violations |= vals > hi
        count = int(violations.sum())
        rate  = count / n
        metrics[f"{col}_out_of_bounds_rate"] = round(rate, 4)
        if count:
            sev = "CRITICAL" if rate > 0.10 else "HIGH" if rate > 0.02 else "MEDIUM"
            bound_str = f"[{lo}, {hi}]" if lo is not None and hi is not None \
                        else f">= {lo}" if hi is None else f"<= {hi}"
            issues.append({
                "severity": sev,
                "check":    f"clinical/bounds/{col}",
                "detail":   (f"{count} ({rate*100:.1f}%) values outside clinical range {bound_str}. "
                             f"min={vals.min():.1f}  max={vals.max():.1f}"),
            })

    # ── 2. Early readmission rate ─────────────────────────────────────────────
    if _READMIT_COL in df.columns:
        readmit_30_rate = float((df[_READMIT_COL].astype(str) == _READMIT_VAL).mean())
        metrics["readmit_30_rate"] = round(readmit_30_rate, 4)
        if readmit_30_rate < _READMIT_30_MIN or readmit_30_rate > _READMIT_30_MAX:
            sev = "HIGH" if readmit_30_rate < 0.02 or readmit_30_rate > 0.50 else "MEDIUM"
            issues.append({
                "severity": sev,
                "check":    "clinical/readmission_rate",
                "detail":   (f"Early readmission (<30 d) rate = {readmit_30_rate*100:.1f}% "
                             f"(expected {_READMIT_30_MIN*100:.0f}%–{_READMIT_30_MAX*100:.0f}%). "
                             f"Clinically implausible prevalence."),
            })

    # ── 3. Group comparison rules (from domain config) ───────────────────────
    gc_issues, gc_metrics = _check_group_comparisons(df, _GRP_CMP_RULES)
    issues.extend(gc_issues)
    metrics.update(gc_metrics)

    # ── 4. Clinically expected positive correlations (from domain config) ───────
    for col_a, col_b, rationale in _CORR_PAIRS:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        a = _num(df[col_a])
        b = _num(df[col_b])
        valid = a.notna() & b.notna()
        if valid.sum() < 30:
            continue
        r = float(np.corrcoef(a[valid], b[valid])[0, 1])
        metrics[f"corr_{col_a}__{col_b}"] = round(r, 4)
        if r < 0:
            issues.append({
                "severity": "MEDIUM",
                "check":    f"clinical/correlation/{col_a}__{col_b}",
                "detail":   (f"Pearson r({col_a}, {col_b}) = {r:.3f} (expected > 0). "
                             f"Clinically: {rationale}."),
            })

    # ── 5. Seed drift (optional) ──────────────────────────────────────────────
    if seed_df is not None:
        seed_metrics: dict = {}
        for col, (lo, hi) in _CLINICAL_BOUNDS.items():
            if col not in seed_df.columns:
                continue
            sv = _num(seed_df[col]).dropna()
            if len(sv) == 0:
                continue
            n = len(sv)
            viol = pd.Series(False, index=sv.index)
            if lo is not None:
                viol |= sv < lo
            if hi is not None:
                viol |= sv > hi
            seed_metrics[f"{col}_out_of_bounds_rate"] = int(viol.sum()) / n

        for key, seed_val in seed_metrics.items():
            synth_val = metrics.get(key)
            if synth_val is None or seed_val is None:
                continue
            denom = max(seed_val, 1e-6)
            if abs(synth_val - seed_val) / denom > 0.20:
                issues.append({
                    "severity": "LOW",
                    "check":    f"clinical/seed_drift/{key}",
                    "detail":   (f"Synthetic {key} = {synth_val:.4f}, "
                                 f"seed = {seed_val:.4f} — "
                                 f">20% relative drift from seed baseline."),
                })

    return issues, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based plausibility check
# ─────────────────────────────────────────────────────────────────────────────
#
# Rules are loaded from the domain config (clinical_plausibility.rules).
# See domains/diabetes.yaml for the schema and examples.
# CLINICAL_RULES is exposed here so callers can inspect or override it.

# Severity → numeric weight used in the plausibility score
_SEVERITY_WEIGHT: dict[str, int] = {
    "CRITICAL": 4,
    "HIGH":     3,
    "MEDIUM":   2,
    "LOW":      1,
}


def _eval_rule(series: pd.Series, operator: str, value: str | int | float | list) -> pd.Series:
    """Return a boolean mask: True where the rule is VIOLATED."""
    op = operator.strip()
    if op in (">", ">=", "<", "<=", "==", "!="):
        numeric = pd.to_numeric(series, errors="coerce")
        ops = {
            ">":  lambda s: s <= value,
            ">=": lambda s: s <  value,
            "<":  lambda s: s >= value,
            "<=": lambda s: s >  value,
            "==": lambda s: s != value,
            "!=": lambda s: s == value,
        }
        return ops[op](numeric) | numeric.isna()
    if op == "between":
        lo, hi = value
        numeric = pd.to_numeric(series, errors="coerce")
        return (numeric < lo) | (numeric > hi) | numeric.isna()
    if op == "in":
        allowed = set(value)
        return ~series.astype(str).isin(allowed)
    if op == "not_in":
        forbidden = set(value)
        return series.astype(str).isin(forbidden)
    raise ValueError(f"Unknown operator: {op!r}. Use >, >=, <, <=, ==, !=, in, not_in, between.")


def check_rule_based_clinicalplausibility(
    df: pd.DataFrame,
    rules: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """
    Evaluate manually defined clinical rules and return a plausibility score.

    Each rule in ``rules`` (defaults to ``CLINICAL_RULES``) specifies a column,
    an operator, a threshold value, a severity, and a description.  For every
    row that violates the rule a violation is counted.

    Plausibility score
    ------------------
    score = 100 × (1 − weighted_violation_rate)

    where::

        weighted_violation_rate = Σ (severity_weight × rule_violation_rate)
                                  ─────────────────────────────────────────
                                        Σ severity_weight

    Score of 100 means zero violations across all rules.
    Score of 0 means every row violates every rule at maximum weight.

    Parameters
    ----------
    df    : DataFrame to evaluate (values may be str or numeric).
    rules : List of rule dicts.  Defaults to ``CLINICAL_RULES``.
            Pass a custom list to override or extend the built-in rules.

    Returns
    -------
    issues  : list of dicts with keys severity / check / detail  (one per violated rule)
    metrics : dict containing per-rule violation rates and the overall plausibility_score
    """
    if rules is None:
        rules = CLINICAL_RULES

    issues: list[dict] = []
    metrics: dict = {}
    total_weight      = 0.0
    weighted_penalty  = 0.0

    for rule in rules:
        col      = rule["column"]
        name     = rule["name"]
        operator = rule["operator"]
        value    = rule["value"]
        severity = rule["severity"]
        desc     = rule["description"]

        if col not in df.columns:
            metrics[f"rule_{name}_skipped"] = True
            continue

        violated  = _eval_rule(df[col], operator, value)
        n_total   = len(df)
        n_violated = int(violated.sum())
        rate       = n_violated / n_total if n_total else 0.0

        weight = _SEVERITY_WEIGHT.get(severity, 1)
        total_weight     += weight
        weighted_penalty += weight * rate

        metrics[f"rule_{name}_violation_rate"] = round(rate, 4)
        metrics[f"rule_{name}_violations"]     = n_violated

        if n_violated:
            sample = df.loc[violated, col].head(3).tolist()
            issues.append({
                "severity": severity,
                "check":    f"clinical/rule/{name}",
                "detail":   (
                    f"{n_violated} ({rate*100:.1f}%) rows violate rule: {desc}. "
                    f"Sample offending values: {sample}"
                ),
            })

    plausibility_score = round(
        100.0 * (1.0 - weighted_penalty / total_weight) if total_weight else 100.0,
        2,
    )
    metrics["plausibility_score"]     = plausibility_score
    metrics["rules_evaluated"]        = sum(1 for r in rules if r["column"] in df.columns)
    metrics["rules_with_violations"]  = len(issues)

    return issues, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Association-rule coverage validation
# ─────────────────────────────────────────────────────────────────────────────

def _parse_frozenset_string(set_string: str) -> set:
    """Convert a serialised frozenset/set string to a Python set."""
    if set_string.startswith("frozenset("):
        set_string = set_string[10:-1]
    elif set_string.startswith("set("):
        set_string = set_string[4:-1]
    return set(ast.literal_eval(set_string))


def _bin_synthetic_row(row: dict) -> set:
    """Map raw synthetic row values to the binned feature strings used by mined rules."""
    binned: set = set()

    for col in ['race', 'gender', 'age', 'A1Cresult', 'metformin', 'insulin', 'readmitted']:
        if col in row and pd.notna(row[col]):
            binned.add(f"{col}_{row[col]}")

    if 'time_in_hospital' in row:
        t = row['time_in_hospital']
        if t <= 2:   binned.add('time_in_hospital_short_stay')
        elif t <= 5: binned.add('time_in_hospital_med_stay')
        else:        binned.add('time_in_hospital_long_stay')

    if 'num_lab_procedures' in row:
        l = row['num_lab_procedures']
        if l < 30:   binned.add('num_lab_procedures_low_labs')
        elif l < 60: binned.add('num_lab_procedures_med_labs')
        else:        binned.add('num_lab_procedures_high_labs')

    if 'num_medications' in row:
        m = row['num_medications']
        if m < 12:   binned.add('num_medications_low_meds')
        elif m < 22: binned.add('num_medications_med_meds')
        else:        binned.add('num_medications_high_meds')

    if 'number_outpatient' in row:
        binned.add('number_outpatient_no_outpatient' if row['number_outpatient'] == 0
                   else 'number_outpatient_has_outpatient')

    if 'number_emergency' in row:
        binned.add('number_emergency_no_emergency' if row['number_emergency'] == 0
                   else 'number_emergency_has_emergency')

    if 'number_inpatient' in row:
        binned.add('number_inpatient_no_prior_inpatient' if row['number_inpatient'] == 0
                   else 'number_inpatient_has_prior_inpatient')

    return binned


def generate_rule_coverage_report(
    synthetic_data_path: str,
    rules_path: str,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Validate synthetic data against auto-mined association rules.

    For every rule (antecedents → consequents) in ``rules_path``:
    - counts how many rows *trigger* the antecedent (times_triggered)
    - counts how many of those also satisfy the consequent (times_passed / times_failed)

    Parameters
    ----------
    synthetic_data_path : Path to the synthetic CSV.
    rules_path          : Path to the mined-rules CSV
                          (must have ``antecedents`` and ``consequents`` columns).

    Returns
    -------
    passed_df                : DataFrame of rows that violated no rules.
    failed_rows_with_feedback: List of dicts — one per failing row with
                               ``row_index``, ``invalid_row_data``, and
                               ``regeneration_prompt``.
    coverage_df              : Aggregated rule-level metrics
                               (rule_id, rule_description, times_triggered,
                               times_passed, times_failed).
    """
    syn_df   = pd.read_csv(synthetic_data_path)
    rules_df = pd.read_csv(rules_path)

    rules_df['antecedents'] = rules_df['antecedents'].apply(_parse_frozenset_string)
    rules_df['consequents'] = rules_df['consequents'].apply(_parse_frozenset_string)

    rule_metrics: dict[int, dict] = {}
    for idx, rule in rules_df.iterrows():
        ant = list(rule['antecedents'])
        con = list(rule['consequents'])
        rule_metrics[idx] = {
            "rule_id":          f"RULE_{idx:03d}",
            "rule_description": f"IF {ant} -> THEN {con}",
            "times_triggered":  0,
            "times_passed":     0,
            "times_failed":     0,
        }

    passed_rows: list[dict] = []
    failed_rows_with_feedback: list[dict] = []

    logger.info("Analyzing %s synthetic rows against %s rules...", len(syn_df), len(rules_df))

    for idx, raw_row in syn_df.iterrows():
        row_features = _bin_synthetic_row(raw_row.to_dict())
        row_failed = False
        violated_explanations: list[str] = []

        for rule_idx, rule in rules_df.iterrows():
            antecedent = rule['antecedents']
            consequent = rule['consequents']

            if antecedent.issubset(row_features):
                rule_metrics[rule_idx]["times_triggered"] += 1
                if consequent.issubset(row_features):
                    rule_metrics[rule_idx]["times_passed"] += 1
                else:
                    rule_metrics[rule_idx]["times_failed"] += 1
                    row_failed = True
                    violated_explanations.append(rule_metrics[rule_idx]["rule_description"])

        if row_failed:
            failed_rows_with_feedback.append({
                "row_index":          idx,
                "invalid_row_data":   raw_row.to_dict(),
                "regeneration_prompt": f"Failed: {violated_explanations}",
            })
        else:
            passed_rows.append(raw_row.to_dict())

    coverage_df = pd.DataFrame(rule_metrics.values())
    return pd.DataFrame(passed_rows), failed_rows_with_feedback, coverage_df
