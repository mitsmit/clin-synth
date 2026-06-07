"""
clin_synth/build_system_prompt.py
=================================
Generate a system prompt for synthetic data generation from:
  - domains/<condition>.yaml              (schema, rules, bounds)
  - processed_data/.../stat_profile/...  (column stats JSON)
  - data/<seed>.csv                       (seed rows for exemplars)
  - domains/<condition>*soft_rules.csv    (mined association rules)

Output: prompts/system_prompt_<condition>.md

Usage
-----
  python -m clin_synth.build_system_prompt heart_failure
  python -m clin_synth.build_system_prompt diabetes --output prompts/my_prompt.md
  python -m clin_synth.build_system_prompt heart_failure --top-rules 15 --exemplar-rows 7
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import textwrap
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_domain(root: Path, condition: str) -> dict:
    path = root / "domains" / f"{condition}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Domain config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_stats(root: Path, domain: dict) -> dict:
    """Load raw stats JSON (columns dict). Prefers raw stats over DP-noised stats."""
    raw_path = root / domain["dp"]["raw_stats"]
    dp_path  = root / domain["generation"]["stats_file"]
    for p in (raw_path, dp_path):
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data.get("columns", {})
    raise FileNotFoundError(f"No stats JSON found (tried {raw_path}, {dp_path})")


def _load_seed(root: Path, domain: dict) -> pd.DataFrame | None:
    path = root / domain["validation"]["seed_csv"]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.columns[0].lower().startswith("unnamed"):
        df = df.iloc[:, 1:]
    return df


def _load_soft_rules(root: Path, condition: str) -> pd.DataFrame | None:
    candidates = sorted((root / "domains").glob(f"{condition}*soft_rules.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _human_name(condition: str) -> str:
    return condition.replace("_", " ")


def _fmt_num(x: float, decimals: int = 0) -> str:
    if decimals == 0:
        return str(int(round(x)))
    return f"{x:.{decimals}f}"


def _col_type(col: str, categorical_cols: set, stats: dict) -> str:
    # Binary int columns (0/1) are shown as "integer" even if in categorical_columns
    if _is_binary(col, stats, categorical_cols):
        return "integer"
    if col in categorical_cols:
        return "categorical"
    dtype = stats.get(col, {}).get("dtype", "")
    if "float" in dtype:
        return "float"
    return "integer"


def _is_binary(col: str, stats: dict, categorical_cols: set) -> bool:
    col_stats = stats.get(col, {})
    # Binary: n_unique == 2, min == 0, max == 1 (works for int64 cols whether or not
    # the domain YAML lists them in categorical_columns)
    return (
        col_stats.get("n_unique", 99) == 2
        and col_stats.get("min", -1) == 0
        and col_stats.get("max", -1) == 1
        and "float" not in str(col_stats.get("dtype", ""))
        and "str" not in str(col_stats.get("dtype", ""))
    )


# Human-readable 0/1 label pairs for known binary column names
_BINARY_LABELS: dict[str, tuple[str, str]] = {
    "sex":               ("female", "male"),
    "gender":            ("female", "male"),
    "anaemia":           ("no", "yes"),
    "diabetes":          ("no", "yes"),
    "high_blood_pressure": ("no", "yes"),
    "smoking":           ("no", "yes"),
    "DEATH_EVENT":       ("survived", "died"),
}


def _binary_note(col: str) -> str:
    zero_lbl, one_lbl = _BINARY_LABELS.get(col, ("no", "yes"))
    return f"Binary flag: `0` ({zero_lbl}) or `1` ({one_lbl})"


def _col_format_notes(col: str, col_type: str, stats: dict, domain: dict) -> str:
    """Build the 'Format / Notes' cell for the schema table."""
    col_stats = stats.get(col, {})
    cat_cols   = set(domain.get("schema", {}).get("categorical_columns", []))
    bounds     = domain.get("clinical_plausibility", {}).get("column_bounds", {})

    if col_type == "categorical":
        freq = col_stats.get("frequency_table", {})
        if freq:
            vals = [f"`{v}`" for v in freq]
            n_total   = col_stats.get("n_total", 1)
            n_non_null = col_stats.get("n_non_null", n_total)
            note = ", ".join(vals[:6])
            if len(vals) > 6:
                note += ", …"
            missing_pct = (n_total - n_non_null) / n_total * 100 if n_total else 0
            if missing_pct > 0.5:
                note += f" (or blank — {missing_pct:.1f}% missing)"
            return note
        return "categorical"

    if _is_binary(col, stats, cat_cols):
        return _binary_note(col)

    # Numeric column
    lo = col_stats.get("min")
    hi = col_stats.get("max")
    # Override with explicit bounds if present
    if col in bounds:
        blo, bhi = bounds[col]
        if blo is not None:
            lo = blo
        if bhi is not None:
            hi = bhi

    mean   = col_stats.get("mean")
    median = col_stats.get("median")
    skew   = col_stats.get("skewness", 0.0) or 0.0

    range_str = ""
    if lo is not None and hi is not None:
        lo_str = _fmt_num(lo, 1 if col_type == "float" else 0)
        hi_str = _fmt_num(hi, 1 if col_type == "float" else 0)
        range_str = f"**{lo_str} – {hi_str}**"

    notes = [range_str] if range_str else []

    if abs(skew) > 2.0 and median is not None:
        notes.append(f"right-skewed, median ~{_fmt_num(median, 0)}")
    elif mean is not None and range_str:
        notes.append(f"mean ~{_fmt_num(mean, 0 if col_type == 'integer' else 1)}")

    return "; ".join(notes)


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_role(condition: str) -> str:
    name = _human_name(condition)
    return textwrap.dedent(f"""\
        # ROLE

        You are a synthetic clinical data generator specialising in {name} patient records.
        Your task is to generate realistic, statistically faithful patient records that
        mirror the distributional properties, numeric correlations, and clinical relationships
        described in the statistical profile you will receive.""")


def _build_task(condition: str) -> str:
    name = _human_name(condition)
    return textwrap.dedent(f"""\
        # TASK

        Generate synthetic {name} patient records in CSV format following the schema,
        clinical logic, and statistical profile provided in the user message. The exact number
        of rows per batch will be stated at the end of the user message.""")


def _build_schema(domain: dict, stats: dict) -> str:
    expected = domain["schema"]["expected_columns"]
    cat_cols  = set(domain["schema"].get("categorical_columns", []))

    header   = "| Column | Type | Format / Notes |"
    divider  = "|--------|------|----------------|"
    rows_md  = []
    for col in expected:
        ctype = _col_type(col, cat_cols, stats)
        notes = _col_format_notes(col, ctype, stats, domain)
        rows_md.append(f"| `{col}` | {ctype} | {notes} |")

    col_order = ",".join(expected)
    table     = "\n".join([header, divider] + rows_md)

    return "\n".join([
        "# SCHEMA",
        "",
        table,
        "",
        "**Column order (CSV header):**",
        "```",
        col_order,
        "```",
    ])


# ── Exemplar rows ─────────────────────────────────────────────────────────────

def _age_bucket(age_val) -> str:
    try:
        age = float(age_val)
    except (TypeError, ValueError):
        return ""
    if age >= 75:
        return "Elderly"
    if age >= 65:
        return "Older"
    if age >= 50:
        return "Middle-aged"
    return "Younger"


def _describe_numeric_feature(col: str, val, col_stats: dict) -> str | None:
    """Return 'low X' / 'high X' if the value is in the bottom/top quartile."""
    q25 = (col_stats.get("quantiles") or {}).get("p25")
    q75 = (col_stats.get("quantiles") or {}).get("p75")
    if q25 is None or q75 is None:
        return None
    label = col.replace("_", " ")
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if v <= q25:
        return f"low {label}"
    if v >= q75:
        return f"high {label}"
    return None


def _describe_time_feature(col: str, val, col_stats: dict) -> str | None:
    """Short / long descriptor for time / stay length columns."""
    median = col_stats.get("median") or col_stats.get("mean")
    if median is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if col in ("time",):
        return "short follow-up" if v < median * 0.6 else ("long follow-up" if v > median * 1.4 else None)
    if "time_in_hospital" in col:
        return "long stay" if v >= 7 else ("short stay" if v <= 2 else None)
    return None


def _annotate_row(row: pd.Series, domain: dict, stats: dict) -> str:
    """Build a one-line clinical annotation comment for an exemplar row."""
    cat_cols   = set(domain.get("schema", {}).get("categorical_columns", []))
    clin       = domain.get("clinical_plausibility", {})
    target_col = domain.get("tstr", {}).get("target_column", "")

    # Key metric columns from group_comparison_rules — numeric only
    key_metrics: list[str] = [
        r["metric_column"]
        for r in clin.get("group_comparison_rules", [])
        if r.get("metric_column") and r["metric_column"] != target_col
            and r["metric_column"] not in cat_cols
    ]

    parts: list[str] = []

    # ── Age ──────────────────────────────────────────────────────────────────
    age_val = row.get("age")
    if age_val is not None:
        bucket = _age_bucket(age_val)
        if bucket:
            parts.append(bucket)

    # ── Sex / gender ─────────────────────────────────────────────────────────
    if "sex" in row.index:
        parts.append("male" if row["sex"] == 1 else "female")
    elif "gender" in row.index:
        g = str(row.get("gender", "")).strip()
        if g and g.lower() not in ("nan", "unknown/invalid", ""):
            parts.append(g.lower())

    # ── Key numeric metrics (low / high) ─────────────────────────────────────
    for col in key_metrics:
        if col not in row.index:
            continue
        desc = _describe_numeric_feature(col, row[col], stats.get(col, {}))
        if desc:
            parts.append(desc)

    # ── Time / stay ───────────────────────────────────────────────────────────
    for time_col in ("time", "time_in_hospital"):
        if time_col in row.index:
            desc = _describe_time_feature(time_col, row[time_col], stats.get(time_col, {}))
            if desc:
                parts.append(desc)
            break

    # ── Notable binary comorbidities (value = 1, not sex/target) ─────────────
    skip_binary = {"sex", "gender", target_col}
    for col in domain["schema"]["expected_columns"]:
        if col in skip_binary or col in cat_cols:
            continue
        if _is_binary(col, stats, cat_cols) and col not in key_metrics:
            try:
                if int(row.get(col, 0)) == 1:
                    parts.append(col.replace("_", " "))
            except (TypeError, ValueError):
                pass

    # ── Outcome ──────────────────────────────────────────────────────────────
    outcome = "?"
    if target_col and target_col in row.index:
        tv = row[target_col]
        outcome_map = _BINARY_LABELS.get(target_col, {})
        if outcome_map and isinstance(tv, (int, float)):
            outcome = outcome_map[1] if int(tv) == 1 else outcome_map[0]
        else:
            outcome = str(tv)

    label = ", ".join(p for p in parts if p) or "typical patient"
    return f"# {label} → {outcome}"


def _sample_exemplar_rows(seed_df: pd.DataFrame, domain: dict, stats: dict, n: int) -> list[tuple[str, str]]:
    """
    Return list of (annotation, csv_row_string) for n exemplar rows.
    Stratified by target class; within each class picks diverse profiles.
    """
    target_col   = domain.get("tstr", {}).get("target_column")
    expected_cols = domain["schema"]["expected_columns"]

    # Keep only the expected columns present in the seed
    available = [c for c in expected_cols if c in seed_df.columns]
    df = seed_df[available].copy()

    if not df.shape[0]:
        return []

    selected_rows: list[pd.Series] = []

    if target_col and target_col in df.columns:
        classes = df[target_col].dropna().unique()
        per_class = max(1, n // len(classes))
        remainder = n - per_class * len(classes)

        for i, cls in enumerate(classes):
            sub = df[df[target_col] == cls]
            # Pick rows that are most "interesting" (extreme in key metric columns)
            key_metrics = [
                r["metric_column"]
                for r in domain.get("clinical_plausibility", {}).get("group_comparison_rules", [])
                if r.get("metric_column") and r["metric_column"] != target_col
                   and r["metric_column"] in sub.columns
            ]
            extra = 1 if i < remainder else 0
            count = per_class + extra

            # Only keep metrics that are actually numeric in this DataFrame
            num_key_metrics = [
                m for m in key_metrics
                if m in sub.columns and pd.api.types.is_numeric_dtype(sub[m])
            ]
            if num_key_metrics:
                # Score each row by mean abs-z-score across key metrics
                z_scores = sub[num_key_metrics].apply(
                    lambda col_: (col_ - col_.mean()) / (col_.std() + 1e-9)
                )
                score = z_scores.abs().mean(axis=1)
            elif key_metrics:
                # Fall back to any available key metric (even non-numeric → zeros)
                score = pd.Series(0.0, index=sub.index)
            if num_key_metrics or key_metrics:
                # Take half from most extreme, half from near-median
                n_extreme = max(1, count // 2)
                n_typical  = count - n_extreme
                extreme_idx = score.nlargest(n_extreme).index
                typical_idx = score.nsmallest(n_typical).index
                selected = pd.concat([sub.loc[extreme_idx], sub.loc[typical_idx]])
            else:
                selected = sub.sample(min(count, len(sub)), random_state=42)

            selected_rows.extend(selected.itertuples(index=False, name=None))
    else:
        rows = df.sample(min(n, len(df)), random_state=42)
        selected_rows = list(rows.itertuples(index=False, name=None))

    cat_cols_set  = set(domain.get("schema", {}).get("categorical_columns", []))
    binary_cols_s = {c for c in expected_cols if _is_binary(c, stats, cat_cols_set)}

    exemplars = []
    for row_tuple in selected_rows[:n]:
        row = pd.Series(dict(zip(available, row_tuple)))
        annotation = _annotate_row(row, domain, stats)
        csv_vals: list[str] = []
        for c in available:
            val = row[c]
            # Replace pandas NaN / float nan with empty string (missingness)
            try:
                import math
                if isinstance(val, float) and math.isnan(val):
                    csv_vals.append("")
                    continue
            except (TypeError, ValueError):
                pass
            # Binary columns: force integer representation (no ".0")
            if c in binary_cols_s:
                try:
                    csv_vals.append(str(int(float(val))))
                    continue
                except (TypeError, ValueError):
                    pass
            csv_vals.append(str(val))
        exemplars.append((annotation, ",".join(csv_vals)))

    return exemplars


def _build_exemplar_rows(seed_df: pd.DataFrame | None, domain: dict, stats: dict, n: int) -> str:
    if seed_df is None:
        return "# EXEMPLAR ROWS\n\n_No seed data available._"

    exemplars = _sample_exemplar_rows(seed_df, domain, stats, n)
    if not exemplars:
        return "# EXEMPLAR ROWS\n\n_No exemplar rows available._"

    lines = [
        "# EXEMPLAR ROWS",
        "",
        "These rows are representative of the target distribution. Study them — do not copy them.",
        "",
        "```",
    ]
    for annotation, csv_row in exemplars:
        lines.append(annotation)
        lines.append(csv_row)
        lines.append("")
    lines.append("```")
    return "\n".join(lines)


# ── Clinical logic constraints ────────────────────────────────────────────────

def _build_clinical_logic(domain: dict, stats: dict) -> str:
    clin       = domain.get("clinical_plausibility", {})
    cat_cols   = set(domain.get("schema", {}).get("categorical_columns", []))
    target_col = domain.get("tstr", {}).get("target_column", "")

    lines = ["# CLINICAL LOGIC CONSTRAINTS", "", "Follow these clinical relationships when generating each row.", ""]

    # ── Group comparison rules → subsections ─────────────────────────────────
    for rule in clin.get("group_comparison_rules", []):
        metric  = rule.get("metric_column", "")
        grp_col = rule.get("group_column", "")
        ga      = rule.get("group_a")
        gb      = rule.get("group_b")
        dir_    = rule.get("expected_direction", ">")
        desc    = rule.get("description", "")
        sev     = rule.get("severity", "MEDIUM")

        header = f"## {metric.replace('_', ' ').title()} and {grp_col.replace('_', ' ')}"
        lines.append(header)

        # Direction sentence
        dir_word = "higher" if dir_ in (">", ">=") else "lower"
        dir_word_b = "lower" if dir_ in (">", ">=") else "higher"

        if isinstance(ga, dict):
            op_a, val_a = ga.get("operator", "!="), ga.get("value", "")
            ga_str = f"{op_a} {val_a}"
        else:
            ga_str = str(ga)
        gb_str = str(gb) if not isinstance(gb, dict) else str(gb.get("value", gb))

        col_stats = stats.get(metric, {})
        q25 = (col_stats.get("quantiles") or {}).get("p25")
        q75 = (col_stats.get("quantiles") or {}).get("p75")

        bullet = f"- `{grp_col}={ga_str}` rows should have {dir_word} mean `{metric}` than `{grp_col}={gb_str}` rows."
        lines.append(bullet)
        if desc:
            lines.append(f"- {desc.strip()}")
        if q25 is not None and q75 is not None:
            lines.append(f"- Typical range: **{_fmt_num(q25, 1)} – {_fmt_num(q75, 1)}** (IQR).")
        lines.append("")

    # ── Correlation pairs ─────────────────────────────────────────────────────
    corr_pairs = clin.get("correlation_pairs", [])
    if corr_pairs:
        lines.append("## Numeric Correlations")
        for pair in corr_pairs:
            if len(pair) >= 3:
                c1, c2, rationale = pair[0], pair[1], pair[2]
                lines.append(f"- `{c1}` ↔ `{c2}`: {rationale}.")
        lines.append("")

    # ── Binary column prevalences ─────────────────────────────────────────────
    binary_cols = [
        c for c in domain["schema"]["expected_columns"]
        if c not in cat_cols and _is_binary(c, stats, cat_cols) and c != target_col
    ]
    if binary_cols:
        lines.append("## Demographics and Comorbidities")
        for col in binary_cols:
            mean_val = stats.get(col, {}).get("mean")
            if mean_val is not None:
                pct = mean_val * 100
                label = col.replace("_", " ")
                lines.append(f"- `{col}=1` ({label}) is present in ~{pct:.0f}% of rows.")
        if target_col and target_col in stats:
            target_mean = stats[target_col].get("mean")
            if target_mean is not None:
                pct = target_mean * 100
                lines.append(f"- `{target_col}=1` is present in ~{pct:.0f}% of rows.")
        lines.append("")

    # ── Hard plausibility rules (HIGH/CRITICAL non-binary bounds) ────────────
    rule_blocks: list[str] = []
    for rule in clin.get("rules", []):
        sev  = rule.get("severity", "")
        col  = rule.get("column", "")
        op   = rule.get("operator", "")
        val  = rule.get("value")
        name = rule.get("description", rule.get("name", ""))

        if sev not in ("CRITICAL", "HIGH"):
            continue
        if _is_binary(col, stats, cat_cols):
            continue  # covered above
        if op == "in":
            vals_str = ", ".join(f"`{v}`" for v in (val or []))
            rule_blocks.append(f"- **{col}** must be one of: {vals_str}.")
        elif op == "between" and isinstance(val, list) and len(val) == 2:
            rule_blocks.append(f"- **{col}** must be between `{val[0]}` and `{val[1]}`.")
        elif op in (">", ">=", "<", "<="):
            rule_blocks.append(f"- **{col}** must be `{op} {val}`.")

    if rule_blocks:
        lines.append("## Hard Constraints")
        lines.extend(rule_blocks)
        lines.append("")

    return "\n".join(lines).rstrip()


# ── Clinical behavior narratives (soft rules) ─────────────────────────────────

def _parse_frozenset_str(s: str) -> list[str]:
    """Parse "frozenset({'a', 'b'})" → ['a', 'b']."""
    items = re.findall(r"'([^']+)'", str(s))
    return sorted(items)


def _build_narratives(soft_rules_df: pd.DataFrame | None, top_n: int) -> str | None:
    if soft_rules_df is None or soft_rules_df.empty:
        return None

    required = {"antecedents", "consequents", "confidence"}
    if not required.issubset(soft_rules_df.columns):
        return None

    df = soft_rules_df.copy()

    # Score: confidence × lift (lift defaults to 1 if missing)
    df["_score"] = df["confidence"] * df.get("lift", 1.0).fillna(1.0)
    df = df[df["confidence"] >= 0.6].sort_values("_score", ascending=False).head(top_n)

    if df.empty:
        return None

    lines = [
        "### CLINICAL BEHAVIOR NARRATIVES (Soft Probabilistic Rules)",
        "When generating individual patient profiles, allow these real-world clinical "
        "tendencies, co-morbidities, and prescribing patterns to guide your token selections naturally:",
    ]
    for _, row in df.iterrows():
        antecedents = _parse_frozenset_str(row["antecedents"])
        consequents = _parse_frozenset_str(row["consequents"])
        freq = int(round(float(row["confidence"]) * 100))
        ant_str = str(antecedents)
        con_str = str(consequents)
        lines.append(
            f"- Profiles matching the baseline criteria {ant_str} should statistically "
            f"favor showing the outcome {con_str} "
            f"(target alignment frequency: ~{freq}%)."
        )

    return "\n".join(lines)


# ── Missingness section ───────────────────────────────────────────────────────

def _build_missingness(domain: dict, stats: dict) -> str | None:
    missing_cols = domain.get("schema", {}).get("columns_with_missingness", [])
    if not missing_cols:
        return None

    rows = []
    for col in missing_cols:
        col_stats = stats.get(col, {})
        n_total    = col_stats.get("n_total", 1) or 1
        n_non_null = col_stats.get("n_non_null", n_total) or n_total
        pct = (n_total - n_non_null) / n_total * 100
        rows.append(f"| `{col}` | {pct:.1f}% | Blank CSV field (`,`) |")

    if not rows:
        return None

    lines = [
        "# MISSINGNESS",
        "",
        "| Column | Missing rate | Representation |",
        "|--------|-------------|----------------|",
    ] + rows + [
        "",
        "**Never use:** `NaN`, `None`, `NULL`, `NA`, or any placeholder string.",
        "A blank field is produced by simply leaving nothing between two commas.",
    ]
    return "\n".join(lines)


# ── Stratified generation mode ────────────────────────────────────────────────

def _build_stratified(domain: dict, seed_df: pd.DataFrame | None) -> str:
    cat_cols  = domain.get("schema", {}).get("categorical_columns", [])
    all_cols  = domain["schema"]["expected_columns"]
    num_cols  = [c for c in all_cols if c not in cat_cols]

    # Build template format description
    template_parts = []
    for col in all_cols:
        template_parts.append(col if col in cat_cols else "___")
    template_str = ",".join(template_parts)

    # Sample 2 real template rows from seed data if available
    template_examples = []
    if seed_df is not None:
        available = [c for c in all_cols if c in seed_df.columns]
        sample = seed_df[available].sample(min(2, len(seed_df)), random_state=7)
        for _, r in sample.iterrows():
            parts = []
            for col in all_cols:
                if col not in available:
                    parts.append("___" if col not in cat_cols else "")
                elif col in cat_cols:
                    parts.append(str(r[col]))
                else:
                    parts.append("___")
            template_examples.append(",".join(parts))

    example_block = "\n".join(template_examples) if template_examples else template_str

    cat_str = ", ".join(f"`{c}`" for c in cat_cols)
    num_str = ", ".join(f"`{c}`" for c in num_cols)

    return "\n".join([
        "# STRATIFIED GENERATION MODE",
        "",
        "In some batches you will receive row templates instead of a generation count.",
        "Each template has its categorical columns pre-filled and numeric slots marked `___`:",
        "",
        "```",
        example_block,
        "```",
        "",
        "When you see templates:",
        f"- **Do NOT change** any pre-filled categorical value ({cat_str}).",
        "- **Replace every `___`** with a realistic value that fits the clinical profile of that row.",
        f"- The numeric slots to fill are: {num_str}.",
        "- Use the target outcome and other categorical context to guide numeric choices.",
        "- Output the completed rows only — same format, no header, no explanation.",
    ])


# ── Output rules ──────────────────────────────────────────────────────────────

def _build_output_rules(domain: dict, stats: dict) -> str:
    cat_cols     = set(domain.get("schema", {}).get("categorical_columns", []))
    expected     = domain["schema"]["expected_columns"]
    missing_cols = set(domain.get("schema", {}).get("columns_with_missingness", []))

    float_cols   = [c for c in expected if _col_type(c, cat_cols, stats) == "float"]
    int_cols     = [c for c in expected if _col_type(c, cat_cols, stats) == "integer"
                    and not _is_binary(c, stats, cat_cols)]
    binary_cols  = [c for c in expected if _is_binary(c, stats, cat_cols)]

    col_order = ",".join(expected)
    lines = [
        "# OUTPUT RULES",
        "",
        "- Output **raw CSV rows only** — no header, no explanation, no markdown fences.",
        f"- Column order: `{col_order}`",
    ]

    if float_cols:
        fc_str = ", ".join(f"`{c}`" for c in float_cols)
        lines.append(f"- {fc_str}: decimals allowed (1 decimal place).")
    if int_cols:
        ic_str = ", ".join(f"`{c}`" for c in int_cols)
        lines.append(f"- {ic_str}: whole integers only, no decimals.")
    if binary_cols:
        bc_str = ", ".join(f"`{c}`" for c in binary_cols)
        lines.append(f"- All binary columns ({bc_str}) must be exactly `0` or `1` — no other values.")
    if missing_cols:
        mc_str = ", ".join(f"`{c}`" for c in missing_cols)
        lines.append(f"- Missing fields ({mc_str}): blank between commas — no placeholder strings.")

    lines += [
        "- No surrounding quotes unless a value contains a comma.",
        "- No index column.",
        "- Do not copy rows from the exemplars above.",
        "- Generate the exact row count specified in the user message.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level assembler
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(
    condition: str,
    root: Path,
    top_rules: int = 15,
    exemplar_rows: int = 7,
) -> str:
    domain    = _load_domain(root, condition)
    stats     = _load_stats(root, domain)
    seed_df   = _load_seed(root, domain)
    soft_rules = _load_soft_rules(root, condition)

    sections: list[str] = [
        _build_role(condition),
        "---",
        _build_task(condition),
        "---",
        _build_schema(domain, stats),
        "---",
        # _build_exemplar_rows disabled — exemplars drawn from real seed data violate
        # the synthetic-only contract; replace with synthesised exemplars or remove.
        # _build_exemplar_rows(seed_df, domain, stats, exemplar_rows),
        _build_clinical_logic(domain, stats),
    ]

    narratives = _build_narratives(soft_rules, top_rules)
    if narratives:
        sections.append(narratives)

    sections.append("---")

    missingness = _build_missingness(domain, stats)
    if missingness:
        sections += [missingness, "---"]

    sections += [
        _build_stratified(domain, seed_df),
        "---",
        _build_output_rules(domain, stats),
    ]

    return "\n\n".join(sections) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a system prompt markdown file for a clinical condition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "condition",
        help="Condition slug matching domains/<condition>.yaml (e.g. heart_failure, diabetes)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path (default: prompts/system_prompt_<condition>.md)",
    )
    parser.add_argument(
        "--top-rules", type=int, default=15,
        help="Number of top soft rules to include in CLINICAL BEHAVIOR NARRATIVES",
    )
    parser.add_argument(
        "--exemplar-rows", type=int, default=7,
        help="Number of exemplar rows to include in the EXEMPLAR ROWS section",
    )
    return parser.parse_args()


def main() -> None:
    from clin_synth.config import get_root
    args   = _parse_args()
    root   = get_root()
    output = Path(args.output) if args.output else root / "prompts" / f"system_prompt_{args.condition}.md"

    logger.info("Building system prompt for: %s", args.condition)
    prompt = build_system_prompt(
        condition=args.condition,
        root=root,
        top_rules=args.top_rules,
        exemplar_rows=args.exemplar_rows,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(prompt, encoding="utf-8")
    logger.info("Written: %s  (%s chars)", output.resolve(), f"{len(prompt):,}")


if __name__ == "__main__":
    main()
