"""Association-rule mining (ARM) for clinical seed data.

Discovers two tiers of rules from a binned/encoded seed dataset:

  - "Hard" rules  — high support / high confidence, treated as near-deterministic
                    clinical facts. Saved for use in validation / clinical-
                    plausibility checks (see clin_synth.clinical_plausibility).
  - "Soft" rules  — lower-confidence probabilistic tendencies, pruned and turned
                    into natural-language guidance for the LLM system prompt
                    (see generate_soft_rules_instruction_block).

Mining is fully domain-driven: which columns to bin (and how) comes from the
active domain's `arm_rules.binning` config section, and which columns map
straight to categorical features comes from `schema.categorical_columns` —
nothing here is hardcoded to a particular condition. To enable ARM mining for
a new domain, add `arm_rules: {binning, hard_rules, soft_rules}` to its YAML
(see domains/heart_failure.yaml for a worked example).

Outputs are written to domains/<condition>/{hard_rules,soft_rules}.csv —
where validate.py and build_system_prompt.py expect to find them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from clin_synth.config import get_root, load_config

logger = logging.getLogger(__name__)


def _slugify(condition: str) -> str:
    return condition.lower().replace(" ", "_").replace("-", "_")


def _domain_rules_dir(condition: str, domains_dir: Path | str | None = None) -> Path:
    """domains/<condition>/ — where both rule CSVs for this condition live,
    read by validate.py (hard_rules.csv) and build_system_prompt.py (soft_rules.csv)."""
    d = Path(domains_dir) if domains_dir else get_root() / "domains"
    return d / _slugify(condition)


def generate_soft_rules_instruction_block(pruned_soft_rules_df: pd.DataFrame) -> str:
    if pruned_soft_rules_df is None or pruned_soft_rules_df.empty:
        return "### CLINICAL BEHAVIOR NARRATIVES\n- Maintain natural medical variance across patient attributes."

    instruction_lines = []
    instruction_lines.append("### CLINICAL BEHAVIOR NARRATIVES (Soft Probabilistic Rules)")
    instruction_lines.append(
        "When generating individual patient profiles, allow these real-world clinical tendencies, "
        "co-morbidities, and prescribing patterns to guide your token selections naturally:"
    )

    for _, row in pruned_soft_rules_df.iterrows():
        antecedent = list(row['antecedents'])
        consequent = list(row['consequents'])
        confidence_percentage = row['confidence'] * 100
        instruction_lines.append(
            f"- Profiles matching the baseline criteria {antecedent} should statistically favor "
            f"showing the outcome {consequent} (target alignment frequency: ~{confidence_percentage:.0f}%)."
        )

    return "\n".join(instruction_lines)


def prune_and_minimize_soft_rules(all_rules_df: pd.DataFrame, soft_cfg: dict) -> pd.DataFrame:
    max_rules = soft_cfg["max_rules"]
    confidence_ceiling = soft_cfg["confidence_ceiling"]
    fallback_floor = soft_cfg["fallback_confidence_floor"]
    max_repeats = soft_cfg["max_repeats_per_consequent"]

    logger.info("%d total rules mined. Pruning to top soft rules...", len(all_rules_df))
    soft_pool = all_rules_df[all_rules_df['confidence'] < confidence_ceiling].copy()
    soft_pool = soft_pool.sort_values(by=['lift', 'support'], ascending=[False, False])

    if len(soft_pool) == 0:
        soft_pool = all_rules_df[all_rules_df['confidence'] >= fallback_floor].copy()
        soft_pool = soft_pool.sort_values(by=['lift', 'support'], ascending=[False, False])

    pruned_rules = []
    seen_consequents = set()

    for _, row in soft_pool.iterrows():
        cons = tuple(sorted(list(row['consequents'])))

        if len(cons) > 1:
            continue

        if list(cons)[0] in seen_consequents:
            matches = sum(1 for r in pruned_rules if list(r['consequents'])[0] == list(cons)[0])
            if matches >= max_repeats:
                continue

        pruned_rules.append(row)
        seen_consequents.add(list(cons)[0])

        if len(pruned_rules) >= max_rules:
            break

    return pd.DataFrame(pruned_rules)


def _bin_continuous_column(series: pd.Series, spec: dict) -> pd.Series:
    method = spec.get("method", "cut")
    labels = spec["labels"]

    if method == "cut":
        return pd.cut(series, bins=spec["bins"], labels=labels)
    if method == "qcut":
        return pd.qcut(series, q=spec["q"], labels=labels)

    raise ValueError(
        f"Unknown binning method '{method}' for column '{spec.get('column')}' "
        "(expected 'cut' or 'qcut')"
    )


def load_and_preprocess_data(filepath: str | Path, schema_cfg: dict, binning_cfg: dict) -> pd.DataFrame:
    """
    Loads a seed dataset and turns it into one-hot-encoded binary features for ARM:

      - columns listed in `schema.categorical_columns` map straight to "<col>_<value>"
      - columns listed in `arm_rules.binning.continuous` are discretized via
        pd.cut/pd.qcut per their spec (column, method, bins/q, labels)

    Both pieces of config come from the active domain — see
    domains/heart_failure.yaml for a worked example.
    """
    df = pd.read_csv(filepath)

    expected_cols = set(schema_cfg.get("expected_columns", []))
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Seed file is missing expected columns: {sorted(missing)}\n"
            f"Found columns: {sorted(df.columns.tolist())}"
        )

    binned_df = pd.DataFrame()

    for col in schema_cfg.get("categorical_columns", []):
        if col in df.columns:
            binned_df[col] = df[col].astype(str)

    for spec in binning_cfg.get("continuous", []):
        col = spec["column"]
        if col not in df.columns:
            continue
        binned_df[col] = _bin_continuous_column(df[col], spec)

    return pd.get_dummies(binned_df, dtype=bool)


def mine_rules(
    seed_csv: str | Path,
    condition: str,
    arm_cfg: dict | None = None,
    domains_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Mine hard and soft association rules from a clinical seed dataset.

    Args:
        seed_csv: path to the seed CSV (must contain the active domain's
            `schema.expected_columns`).
        condition: condition label, e.g. "heart_failure". Determines the
            output folder (domains/<slug>/).
        arm_cfg: optional override for the `arm_rules` config block (must include
            a `binning` section); defaults to the active domain's `arm_rules`
            (call clin_synth.set_domain() first, or pass this explicitly).
        domains_dir: optional override for where rule CSVs are written; defaults
            to <project_root>/domains — where build_system_prompt.py and
            validate.py look for them (domains/<condition>/{hard,soft}_rules.csv).

    Returns:
        (hard_rules_df, soft_rules_df, soft_rules_instruction_block)
    """
    cfg = load_config()

    if arm_cfg is None:
        arm_cfg = cfg.get("arm_rules")
        if arm_cfg is None:
            raise ValueError(
                f"No 'arm_rules' section found in the domain config for '{condition}'. "
                f"Add `arm_rules: {{binning, hard_rules, soft_rules}}` to "
                f"domains/{_slugify(condition)}.yaml (see domains/heart_failure.yaml "
                "for a worked example), or pass arm_cfg explicitly."
            )

    binning_cfg = arm_cfg.get("binning")
    if not binning_cfg or not binning_cfg.get("continuous"):
        raise ValueError(
            f"No 'arm_rules.binning' section found for '{condition}'. ARM mining "
            "needs binning rules to discretize continuous columns into ARM-ready "
            "features — see domains/heart_failure.yaml for the expected format."
        )

    hcfg = arm_cfg["hard_rules"]
    scfg = arm_cfg["soft_rules"]
    schema_cfg = cfg.get("schema", {})

    encoded_df = load_and_preprocess_data(seed_csv, schema_cfg, binning_cfg)
    logger.info("Dataset encoded into %d unique binary features.", encoded_df.shape[1])

    discovered_rules = pd.DataFrame()
    all_rules = pd.DataFrame()

    for support in hcfg["support_search_ladder"]:
        logger.info("Trying min_support=%.2f (%d of %d patients)...",
                    support, int(support * len(encoded_df)), len(encoded_df))
        frequent_itemsets = fpgrowth(encoded_df, min_support=support, use_colnames=True)
        all_rules = association_rules(frequent_itemsets, metric="confidence",
                                      min_threshold=hcfg["fpgrowth_min_threshold"])
        discovered_rules = all_rules[
            (all_rules['support'] >= hcfg["min_support"]) &
            (all_rules['confidence'] >= hcfg["min_confidence"])
        ]
        if len(discovered_rules) > 0:
            logger.info("Found %d hard rules at support=%.2f, confidence>=%.2f",
                        len(discovered_rules), support, hcfg["min_confidence"])
            break

    if discovered_rules.empty:
        raise RuntimeError(
            f"No hard rules discovered across the support search ladder "
            f"{hcfg['support_search_ladder']}. Check whether the seed columns "
            "contain unique IDs or unbinned continuous values."
        )

    rules_dir = _domain_rules_dir(condition, domains_dir)
    rules_dir.mkdir(parents=True, exist_ok=True)

    hard_rules = discovered_rules.sort_values(by='support', ascending=False)
    hard_path = rules_dir / "hard_rules.csv"
    hard_rules[['antecedents', 'consequents', 'support', 'confidence']].to_csv(hard_path, index=False)
    logger.info("Hard rules saved -> %s", hard_path)

    soft_rules = prune_and_minimize_soft_rules(all_rules, scfg)
    soft_path = rules_dir / "soft_rules.csv"
    soft_rules[['antecedents', 'consequents', 'support', 'confidence']].to_csv(soft_path, index=False)
    logger.info("Soft rules pruned to %d -> %s", len(soft_rules), soft_path)

    instruction_block = generate_soft_rules_instruction_block(soft_rules)

    return (
        hard_rules[['antecedents', 'consequents', 'support', 'confidence']],
        soft_rules[['antecedents', 'consequents', 'support', 'confidence']],
        instruction_block,
    )


if __name__ == "__main__":
    import sys

    from clin_synth.config import set_domain

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 3:
        print("Usage: python -m clin_synth.ruleex.arm <condition> <seed_csv>")
        raise SystemExit(1)

    condition_arg, seed_arg = sys.argv[1], sys.argv[2]
    set_domain(get_root() / "domains" / f"{_slugify(condition_arg)}.yaml")

    try:
        hard_df, soft_df, instructions = mine_rules(seed_arg, condition_arg)
        print("\n" + "═" * 70)
        print("SOFT RULES INSTRUCTION BLOCK FOR LLM SYSTEM PROMPT:\n")
        print(instructions)
    except FileNotFoundError:
        print(f"Error: could not find seed file at {seed_arg}.")
