"""
build_user_message.py
=====================
Builds the per-batch user message from diab_stats.json.

Structure of the user message:
  1. Categorical distributions  (frequency tables as JSON)
  2. Numeric distributions      (quantile profiles + shape as JSON)
  3. Critical reminders         (hard constraints)
  4. Correlations               (at end — Pearson r pairs as JSON)
  5. Archetype instruction      (rotated per batch — forces subgroup coverage)
  6. Diversity instruction      (explicit anti-repetition rules)
  7. Generation instruction     ("Generate N rows. Batch X of Y.")

Note: missingness rates are communicated once via the system prompt MISSINGNESS
section (built by build_system_prompt.py). They are not repeated here.

Usage:
    from build_user_message import build_user_message
    msg = build_user_message(stats, batch_size=500, batch_num=1, n_batches=20)
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ── DISABLED: hardcoded diabetes categorical fallbacks ────────────────────────
# These were used when the stats JSON lacked frequency tables for categorical
# columns (semantic_type="unknown"). Now relying on domain YAML categorical_fallbacks
# or the stats JSON directly. Re-enable if a domain's stats JSON is missing
# frequency tables and no YAML fallbacks are defined.
#
# _FALLBACK_DISTRIBUTIONS: dict[str, dict] = {
#     "race":       {"proportions_of_non_missing": {"Caucasian": 0.7649, ...}},
#     "gender":     {"proportions": {"Female": 0.5376, "Male": 0.4624, ...}},
#     "age":        {"proportions": {"[70-80)": 0.2562, ...}},
#     "A1Cresult":  {"proportions_of_non_missing": {">8": 0.4828, "Norm": 0.2932, ">7": 0.2240}},
#     "metformin":  {"proportions": {"No": 0.8036, "Steady": 0.1803, ...}},
#     "insulin":    {"proportions": {"No": 0.4656, "Steady": 0.3031, ...}},
#     "readmitted": {"proportions": {"NO": 0.5391, ">30": 0.3493, "<30": 0.1116}},
# }
_FALLBACK_DISTRIBUTIONS: dict[str, dict] = {}  # disabled — see comment above

# ── DISABLED: hardcoded diabetes critical reminders ──────────────────────────
# Replaced by build_critical_reminders() which derives the same content from
# the stats JSON and domain YAML dynamically, so any domain works without
# hand-written strings.
#
# _CRITICAL_REMINDERS = """CRITICAL REMINDERS (enforce in every row):
# - metformin: No in EXACTLY 80% of rows, Steady 18%, Up 1%, Down <1%. ...
# - insulin: No in EXACTLY 47% of rows, Steady 30%, Down 12%, Up 11%. ...
# - readmitted: NO in EXACTLY 54% of rows, >30 in EXACTLY 35%, <30 in EXACTLY 11%. ...
# - number_outpatient must be 0 in ~75% of rows (highly zero-inflated)
# - number_emergency must be 0 in ~85% of rows (extremely zero-inflated)
# - time_in_hospital is strongly correlated with num_medications (+0.466)
# """

# ── Patient archetypes — rotated per batch to force subgroup coverage ─────────
# Each archetype describes a clinically distinct patient profile.
# Rotating these across batches prevents mode collapse and ensures rare
# subgroups (young patients, high utilisers, non-Caucasian patients) are
# represented proportionally rather than being crowded out by the modal patient.
_ARCHETYPES: list[dict] = [
    {
        "name": "Elderly long-stay",
        "description": (
            "Patients aged 70–90 with longer hospital stays (time_in_hospital 6–14), "
            "high medication burden (num_medications 18–40), many lab procedures "
            "(num_lab_procedures 50–90). Readmitted >30 days is more common for these patients. "
            "Insulin=Steady is the most common insulin status; metformin=No for most. "
            "When A1Cresult is present (only ~17% of rows), prefer >8 or >7."
        ),
    },
    {
        "name": "Young low-complexity",
        "description": (
            "Patients aged 20–40, short stays (time_in_hospital 1–3), "
            "minimal medications (num_medications 4–10), few lab procedures "
            "(num_lab_procedures 15–35). Readmitted=NO for most. "
            "Metformin=No (most), insulin=No or Steady. A1Cresult mostly blank."
        ),
    },
    {
        "name": "High utiliser",
        "description": (
            "Patients with multiple prior emergency visits (number_emergency 2–5) "
            "and inpatient visits (number_inpatient 2–6). Moderate to long stays. "
            "High medication burden. Readmitted >30 days is elevated; <30 days is rare (still only ~11% globally). "
            "Metformin=No for most; insulin varies (No, Steady, or Up in proportion)."
        ),
    },
    {
        "name": "Well-controlled diabetic",
        "description": (
            "Low medication burden (num_medications 5–14), short stays (time_in_hospital 1–4), "
            "no emergency or inpatient prior visits. Metformin=Steady for some, No for most. "
            "Insulin=No or Down. Readmitted=NO. "
            "When A1Cresult is present (only ~17% of rows), use Norm."
        ),
    },
    {
        "name": "Insulin-dependent complex",
        "description": (
            "Patients on active insulin (Steady is 3× more common than Up), higher medication counts (num_medications 20–45), "
            "longer stays (time_in_hospital 5–12). Age 50–80. Readmission: NO ~54%, >30 ~35%, <30 ~11%. "
            "Metformin=No for most of these patients. "
            "When A1Cresult is present (only ~17% of rows), prefer >8."
        ),
    },
    {
        "name": "Minority demographic",
        "description": (
            "Skew toward underrepresented race groups (AfricanAmerican ~40%, Hispanic ~8%, "
            "Asian ~3%, Other ~5%, Caucasian ~44%) while keeping race blank in ~2.2% of rows. "
            "All age groups, clinical profiles vary. Mix of insulin and metformin statuses. "
            "Reflect realistic clinical diversity across readmission outcomes."
        ),
    },
    {
        "name": "Short-stay moderate risk",
        "description": (
            "Patients with short stays (time_in_hospital 1–3), moderate lab procedures "
            "(num_lab_procedures 25–50), and moderate medications (num_medications 8–18). "
            "Mixed readmission outcomes. A1Cresult mostly blank (test not ordered). "
            "Age 50–70. Outpatient and emergency visits mostly 0."
        ),
    },
    {
        "name": "Mid-age outpatient-active",
        "description": (
            "Patients aged 40–60 with notable outpatient visit history "
            "(number_outpatient 1–5). Moderate hospital stays. Mixed A1C results. "
            "Active diabetes management: mix of metformin and insulin statuses. "
            "Readmission rate higher than average. number_inpatient 1–3."
        ),
    },
]

# ── DISABLED: hardcoded diabetes diversity instruction ────────────────────────
# Replaced by build_diversity_instruction() which derives spread and
# under-representation guidance from the stats JSON dynamically.
#
# _DIVERSITY_INSTRUCTION = """DIVERSITY REQUIREMENTS (strictly enforce):
# - Do NOT generate the same row twice within this batch.
# - Spread time_in_hospital across the full range 1–14, not clustered at 4.
# - ...
# """


def _extract_categorical(stats: dict) -> dict[str, dict]:
    """Extract frequency tables for all categorical columns.
    Falls back to _FALLBACK_DISTRIBUTIONS for columns whose semantic_type
    was not resolved or whose frequency_table was not computed
    (e.g. high-missingness columns like A1Cresult).
    """
    categorical = {}
    all_cols = stats.get("columns", {})

    # First pass: typed categorical columns from stats
    for col, info in all_cols.items():
        if info.get("semantic_type") == "categorical":
            freq = info.get("frequency_table") or {}
            # Domain YAML fallbacks take precedence over hardcoded ones
            _domain_fallbacks = _DOMAIN_CFG.get("generation", {}).get("categorical_fallbacks", {})
            if freq:
                categorical[col] = {
                    "proportions": {
                        val: round(data["proportion"], 4)
                        for val, data in freq.items()
                    },
                }
            elif col in _domain_fallbacks:
                categorical[col] = _domain_fallbacks[col]
            elif col in _FALLBACK_DISTRIBUTIONS:
                categorical[col] = _FALLBACK_DISTRIBUTIONS[col]
            else:
                categorical[col] = {}

    # Second pass: columns missing from stats — check domain YAML then hardcoded fallbacks
    _domain_fallbacks = _DOMAIN_CFG.get("generation", {}).get("categorical_fallbacks", {})
    for col, fallback in {**_FALLBACK_DISTRIBUTIONS, **_domain_fallbacks}.items():
        if col not in categorical:
            categorical[col] = fallback

    return categorical


def _extract_numeric(stats: dict) -> dict[str, dict]:
    """Extract quantile profiles and shape stats for all numeric columns."""
    numeric = {}
    for col, info in stats.get("columns", {}).items():
        if info.get("semantic_type") in ("integer", "float"):
            q = info.get("quantiles", {})
            numeric[col] = {
                "mean":     round(info.get("mean", 0), 3),
                "median":   round(info.get("median", 0), 3),
                "std":      round(info.get("std", 0), 3),
                "min":      info.get("min"),
                "max":      info.get("max"),
                "skewness": round(info.get("skewness", 0), 3),
                "quantiles": {
                    "p25": q.get("p25"),
                    "p50": q.get("p50"),
                    "p75": q.get("p75"),
                    "p90": q.get("p90"),
                    "p95": q.get("p95"),
                    "p99": q.get("p99"),
                },
            }
    return numeric



def _extract_correlations(stats: dict) -> list[dict[str, object]]:
    """Extract numeric correlations sorted by absolute strength descending."""
    pairs = []
    correlations = stats.get("correlations", {}).get("pearson", {})
    seen = set()
    for col_a, targets in correlations.items():
        for col_b, r in targets.items():
            key = tuple(sorted([col_a, col_b]))
            if key not in seen:
                seen.add(key)
                pairs.append({
                    "columns": list(key),
                    "pearson_r": round(r, 3),
                })
    # Sort by absolute correlation strength descending
    pairs.sort(key=lambda p: abs(p["pearson_r"]), reverse=True)
    return pairs


# ── Domain config cache — populated by _init_build_cfg() ─────────────────────
_DOMAIN_CFG: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic builders — derive reminders, diversity, and archetypes from stats
# so any domain works without hand-writing these strings.
# ─────────────────────────────────────────────────────────────────────────────

def build_critical_reminders(stats: dict, domain: dict) -> str:
    """Auto-generate CRITICAL REMINDERS from stats and domain config."""
    lines = ["CRITICAL REMINDERS (enforce in every row):"]
    cols_s    = stats.get("columns", {})
    cat_cols  = set(domain.get("schema", {}).get("categorical_columns", []))
    target    = domain.get("tstr", {}).get("target_column", "")
    pearson   = stats.get("correlations", {}).get("pearson", {})

    # 1. Target column class distribution
    if target:
        t_s  = cols_s.get(target, {})
        freq = t_s.get("frequency_table", {})
        if freq:
            parts = [
                f"{v}={d['proportion']*100:.0f}%"
                for v, d in sorted(freq.items(), key=lambda x: -x[1]["proportion"])
            ]
            lines.append(f"- {target}: {', '.join(parts)} — reproduce this ratio exactly.")
        elif t_s.get("mean") is not None:
            pct = t_s["mean"] * 100
            lines.append(f"- {target}=1 in ~{pct:.0f}% of rows; =0 in ~{100-pct:.0f}%.")

    # 2. Categorical columns — distribution skew
    for col in sorted(cat_cols):
        if col == target:
            continue
        c_s  = cols_s.get(col, {})
        freq = c_s.get("frequency_table", {})

        for val, data in freq.items():
            pct = data["proportion"] * 100
            if pct > 70:
                lines.append(
                    f"- {col}={val!r} dominates at ~{pct:.0f}% — "
                    f"do NOT inflate other {col} values."
                )
            elif pct < 8:
                lines.append(
                    f"- {col}={val!r} is rare (~{pct:.0f}%) — do NOT under-generate it."
                )

    # 3. Highly skewed or zero-inflated numeric columns
    for col, c_s in cols_s.items():
        if col in cat_cols or col == target:
            continue
        if c_s.get("semantic_type") not in ("integer", "float"):
            continue
        q    = c_s.get("quantiles", {}) or {}
        skew = c_s.get("skewness", 0) or 0

        if q.get("p50") == 0 and q.get("p75") == 0:
            lines.append(f"- {col} is zero-inflated: >75% of rows must be 0.")
        elif q.get("p50") == 0:
            lines.append(f"- {col} is zero-inflated: >50% of rows must be 0.")
        elif abs(skew) > 2:
            median = c_s.get("median", 0)
            p95    = q.get("p95", c_s.get("max", "?"))
            lines.append(
                f"- {col} is right-skewed (skew={skew:.1f}): "
                f"most values near median ~{median:.0f}; "
                f"extreme values (>{p95:.0f}) appear in only ~5% of rows."
            )

    # 4. Strong Pearson correlations (|r| > 0.35)
    seen: set = set()
    for col_a, targets_map in pearson.items():
        for col_b, r in targets_map.items():
            key = tuple(sorted([col_a, col_b]))
            if key not in seen and abs(r) > 0.35:
                seen.add(key)
                direction = "positively" if r > 0 else "inversely"
                lines.append(
                    f"- {col_a} is {direction} correlated with {col_b} "
                    f"(r={r:+.2f}) — preserve this relationship."
                )

    return "\n".join(lines)


def build_diversity_instruction(stats: dict, domain: dict) -> str:
    """Auto-generate DIVERSITY REQUIREMENTS from stats and domain config."""
    lines = [
        "DIVERSITY REQUIREMENTS (strictly enforce):",
        "- Do NOT generate the same row twice within this batch.",
    ]
    cols_s   = stats.get("columns", {})
    cat_cols = set(domain.get("schema", {}).get("categorical_columns", []))
    target   = domain.get("tstr", {}).get("target_column", "")

    # Numeric: spread guidance for clustered or zero-inflated columns
    for col, c_s in cols_s.items():
        if col in cat_cols or col == target:
            continue
        if c_s.get("semantic_type") not in ("integer", "float"):
            continue
        q      = c_s.get("quantiles", {}) or {}
        lo, hi = c_s.get("min"), c_s.get("max")
        median = c_s.get("median")
        p25, p75 = q.get("p25"), q.get("p75")

        if lo is None or hi is None:
            continue

        if q.get("p50") == 0 and q.get("p75") == 0:
            p90 = q.get("p90", 0)
            if p90:
                lines.append(
                    f"- {col}: mostly 0, but non-zero values (up to {hi:.0f}) "
                    f"must appear in ~10–25% of rows."
                )
            continue

        if p25 is not None and p75 is not None and (hi - lo) > 0:
            iqr  = p75 - p25
            rang = hi - lo
            if iqr / rang < 0.25:
                lines.append(
                    f"- Spread {col} across the full range "
                    f"{lo:.0f}–{hi:.0f}, not clustered near {median:.0f}."
                )

    # Categorical: surface under-represented values (5–20%)
    for col in sorted(cat_cols):
        if col == target:
            continue
        c_s  = cols_s.get(col, {})
        freq = c_s.get("frequency_table", {})
        for val, data in freq.items():
            pct = data["proportion"] * 100
            if 3 < pct < 20:
                lines.append(
                    f"- Include {col}={val!r} in ~{pct:.0f}% of rows — "
                    f"do not omit under-represented values."
                )

    return "\n".join(lines)


def _auto_archetypes(stats: dict, domain: dict) -> list[dict]:
    """Generate archetypes from stats when none are defined in the domain YAML."""
    target   = domain.get("tstr", {}).get("target_column", "")
    clin     = domain.get("clinical_plausibility", {})
    cols_s   = stats.get("columns", {})
    cat_cols = set(domain.get("schema", {}).get("categorical_columns", []))
    result: list[dict] = []

    # One archetype per target class value
    if target:
        t_s  = cols_s.get(target, {})
        freq = t_s.get("frequency_table", {})
        vals = list(freq.keys()) if freq else (["1", "0"] if t_s.get("mean") else [])

        for val in vals:
            pct = (freq[val]["proportion"] * 100) if freq else None
            hdr = f"{target}={val}" + (f" ({pct:.0f}% of rows)" if pct else "")
            desc_parts: list[str] = [hdr + "."]

            # Add key metric guidance from group_comparison_rules
            for rule in clin.get("group_comparison_rules", []):
                if rule.get("group_column") != target:
                    continue
                ga = rule.get("group_a")
                if str(ga) != str(val) and not (isinstance(ga, dict)):
                    continue
                metric = rule.get("metric_column", "")
                if not metric or metric in cat_cols:
                    continue
                direction = rule.get("expected_direction", "")
                q = (cols_s.get(metric) or {}).get("quantiles", {}) or {}
                if direction in (">", ">=") and q.get("p75") is not None:
                    desc_parts.append(f"Higher {metric} (above p75={q['p75']:.1f}).")
                elif direction in ("<", "<=") and q.get("p25") is not None:
                    desc_parts.append(f"Lower {metric} (below p25={q['p25']:.1f}).")

            result.append({"name": f"{target}={val}", "description": " ".join(desc_parts)})

    # Always add a "typical patient" archetype
    typical: list[str] = ["Typical patient near dataset medians."]
    for col, c_s in cols_s.items():
        if col in cat_cols or col == target:
            continue
        if c_s.get("semantic_type") in ("integer", "float"):
            m = c_s.get("median")
            if m is not None:
                typical.append(f"{col}≈{m:.0f}.")
            if len(typical) > 5:
                break
    result.append({"name": "Typical/median profile", "description": " ".join(typical)})

    return result


def _get_archetypes_list(stats: dict, domain: dict) -> list[dict]:
    """Return archetypes: YAML-defined if present, otherwise auto-generated."""
    yaml_archetypes = domain.get("generation", {}).get("archetypes", [])
    if yaml_archetypes:
        return yaml_archetypes
    # Fall back to hardcoded diabetes list when domain config is absent/default
    if not domain.get("schema", {}).get("expected_columns"):
        return _ARCHETYPES
    return _auto_archetypes(stats, domain)


def _get_archetype(batch_num: int, archetypes: list[dict] | None = None) -> dict:
    """Return the archetype for this batch, rotating through the list."""
    pool = archetypes or _ARCHETYPES
    return pool[(batch_num - 1) % len(pool)]


def build_user_message(
    stats: dict,
    batch_size: int,
    batch_num: int,
    n_batches: int,
) -> str:
    """
    Build the full user message for one generation batch.

    Injects: statistical profile (JSON) → critical reminders → correlations
             → archetype instruction (rotated) → diversity rules → generate command.

    Parameters
    ----------
    stats       : Loaded diab_stats_dp.json dict.
    batch_size  : Number of rows to generate in this batch.
    batch_num   : 1-based batch number.
    n_batches   : Total number of batches.

    Returns
    -------
    str : The complete user message for {"role": "user", "content": ...}
    """
    categorical  = _extract_categorical(stats)
    numeric      = _extract_numeric(stats)
    correlations = _extract_correlations(stats)

    stat_profile = {
        "categorical_distributions": categorical,
        "numeric_distributions":     numeric,
        "numeric_correlations":      correlations,
    }

    # Always use dynamic builders; fall back to empty domain dict if none loaded.
    domain_cfg = _DOMAIN_CFG or {}
    critical        = build_critical_reminders(stats, domain_cfg)
    diversity       = build_diversity_instruction(stats, domain_cfg)
    archetypes_list = _get_archetypes_list(stats, domain_cfg)

    archetype = _get_archetype(batch_num, archetypes_list)
    archetype_block = (
        f"PATIENT ARCHETYPE FOR THIS BATCH — {archetype['name'].upper()}\n"
        f"Skew the rows in this batch toward this profile "
        f"(while still respecting overall statistical distributions above):\n"
        f"{archetype['description']}"
    )

    return (
        f"STATISTICAL PROFILE:\n"
        f"{json.dumps(stat_profile, indent=2)}\n\n"
        f"{critical}\n\n"
        f"{archetype_block}\n\n"
        f"{diversity}\n\n"
        f"Generate exactly {batch_size} CSV rows "
        f"(no header, no explanation). "
        f"Batch {batch_num} of {n_batches}. "
        f"Output ONLY the raw CSV rows."
    )


# Column order matches EXPECTED_COLS in generate.py
# Defaults are diabetes-specific; _init_build_cfg() overrides these from the domain config.
_COL_ORDER: list[str] = [
    "race", "gender", "age",
    "time_in_hospital", "num_lab_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "A1Cresult", "metformin", "insulin", "readmitted",
]
_NUMERIC_SLOTS: set[str] = {
    "time_in_hospital", "num_lab_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
}
_CAT_SLOTS: set[str] = set()

_STRATIFIED_INSTRUCTIONS: str = ""  # rebuilt by _init_build_cfg()

_DEFAULT_STRATIFIED_INSTRUCTIONS = """TASK — FILL IN NUMERIC VALUES ONLY
Each row below is a 13-column CSV template. The categorical columns are pre-filled; replace every ___ with a realistic integer.
Column order: race, gender, age, time_in_hospital, num_lab_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, A1Cresult, metformin, insulin, readmitted.
Rules:
- Output EXACTLY 13 comma-separated values per row, in the same column order.
- Keep race, gender, age, A1Cresult, metformin, insulin, readmitted exactly as shown — do NOT drop or move them.
- Replace only the 6 ___ slots (columns 4–9) with integers.
- No header, no explanation, no row numbers."""


def _init_build_cfg() -> None:
    """Align module-level config from the active domain YAML.
    Call this after set_domain() in generate.py."""
    global _COL_ORDER, _NUMERIC_SLOTS, _CAT_SLOTS, _STRATIFIED_INSTRUCTIONS, _DOMAIN_CFG
    try:
        from clin_synth.config import load_config
        cfg = load_config()
        _DOMAIN_CFG = cfg
    except Exception:
        _STRATIFIED_INSTRUCTIONS = _DEFAULT_STRATIFIED_INSTRUCTIONS
        return

    schema = cfg.get("schema", {})
    all_cols = schema.get("expected_columns", [])
    cat_cols = set(schema.get("categorical_columns", []))
    target   = cfg.get("tstr", {}).get("target_column", "")

    if not all_cols:
        _STRATIFIED_INSTRUCTIONS = _DEFAULT_STRATIFIED_INSTRUCTIONS
        return

    _COL_ORDER   = list(all_cols)
    _CAT_SLOTS   = cat_cols | ({target} if target else set())
    _NUMERIC_SLOTS = set(c for c in all_cols if c not in _CAT_SLOTS)

    n_cols    = len(_COL_ORDER)
    n_num     = len(_NUMERIC_SLOTS)
    n_cat     = len(_CAT_SLOTS)
    col_list  = ", ".join(_COL_ORDER)
    cat_list  = ", ".join(c for c in _COL_ORDER if c in _CAT_SLOTS)
    _STRATIFIED_INSTRUCTIONS = (
        f"TASK — FILL IN NUMERIC VALUES ONLY\n"
        f"Each row below is a {n_cols}-column CSV template. "
        f"The categorical columns are pre-filled; replace every ___ with a realistic value.\n"
        f"Column order: {col_list}.\n"
        f"Rules:\n"
        f"- Output EXACTLY {n_cols} comma-separated values per row, in the same column order.\n"
        f"- Keep {cat_list} exactly as shown — do NOT drop, move, or change them.\n"
        f"- Replace only the {n_num} ___ slots with numeric values.\n"
        f"- No header, no explanation, no row numbers."
    )


def build_stratified_user_message(
    stats: dict,
    batch_categoricals: list[dict],
    batch_num: int,
    n_batches: int,
) -> str:
    """
    Build the user message for stratified generation.

    Categorical values are pre-assigned per row; the LLM fills in only the
    six numeric slots (marked ___). This preserves the joint distribution
    between categoricals and the numeric columns.

    Parameters
    ----------
    stats              : Loaded stats JSON dict (used for numeric profile only).
    batch_categoricals : List of dicts with pre-assigned categorical values,
                         one dict per row. Keys: race, gender, age, A1Cresult,
                         metformin, insulin, readmitted.
    batch_num          : 1-based batch number.
    n_batches          : Total number of batches.
    """
    numeric      = _extract_numeric(stats)
    correlations = _extract_correlations(stats)

    stat_profile = {
        "numeric_distributions": numeric,
        "numeric_correlations":  correlations,
    }

    # Build row templates — categorical values filled, numeric slots are ___
    templates = []
    for cats in batch_categoricals:
        cells = []
        for col in _COL_ORDER:
            if col in _NUMERIC_SLOTS:
                cells.append("___")
            else:
                cells.append(cats.get(col, ""))
        templates.append(",".join(cells))

    templates_block = "\n".join(templates)

    archetype = _get_archetype(batch_num)
    archetype_block = (
        f"PATIENT ARCHETYPE FOR THIS BATCH — {archetype['name'].upper()}\n"
        f"{archetype['description']}"
    )

    return (
        f"NUMERIC STATISTICAL PROFILE:\n"
        f"{json.dumps(stat_profile, indent=2)}\n\n"
        f"{_STRATIFIED_INSTRUCTIONS}\n\n"
        f"{templates_block}\n\n"
        f"{archetype_block}\n\n"
        f"Batch {batch_num} of {n_batches}. "
        f"Output ONLY the {len(batch_categoricals)} completed CSV rows."
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    stats_path = Path(__file__).parent.parent / "data" / "diab_stats_dp.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    logger.info("Archetypes defined : %s", len(_ARCHETYPES))
    logger.info("Rotation cycle     : %s batches per full cycle\n", len(_ARCHETYPES))
    logger.info("Archetype rotation preview:")
    for i in range(1, len(_ARCHETYPES) + 1):
        a = _get_archetype(i)
        logger.info("  Batch %2d: %s", i, a['name'])

    logger.info("")
    msg = build_user_message(stats, batch_size=100, batch_num=1, n_batches=20)
    logger.info("Message length (batch 1): %s chars", f"{len(msg):,}")
    logger.info("\n--- Archetype + Diversity section (last 800 chars) ---")
    logger.info("%s", msg[-800:])
