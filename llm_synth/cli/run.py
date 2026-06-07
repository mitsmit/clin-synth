"""
llm-synth CLI entry point.

Subcommands
-----------
  run           Full pipeline: profile → dp → generate → validate → tstr
  profile       Extract seed statistics only
  dp            Apply differential privacy to a stats JSON
  generate      Generate synthetic rows from DP stats
  validate      Validate synthetic CSV against seed stats
  tstr          Train-on-Synthetic / Test-on-Real evaluation
  build-prompt  Generate system prompt from domain config and seed data

Output layout (produced by 'run')
----------------------------------
  processed_data/<condition>/synthetic_output.csv
  processed_data/<condition>/stat_profile/<condition>_stats.json
  processed_data/<condition>/stat_profile/<condition>_dp_stats.json
  processed_data/<condition>/seed_data/<seed-filename>.csv

Usage
-----
  llm-synth run      --seed data/diab_seed.csv --condition diabetes
  llm-synth profile  --seed data/diab_seed.csv --output processed_data/diabetes/stat_profile/diabetes_stats.json
  llm-synth dp       --stats processed_data/diabetes/stat_profile/diabetes_stats.json --epsilon 1.0
  llm-synth generate --stats data/diab_stats_dp.json --rows 5000 --output processed_data/diabetes/synthetic_output.csv
  llm-synth validate --synthetic processed_data/diabetes/synthetic_output.csv
  llm-synth tstr
  llm-synth build-prompt heart_failure
  llm-synth build-prompt diabetes --top-rules 10 --exemplar-rows 6
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def _cmd_run(args: argparse.Namespace) -> None:
    from llm_synth.config import set_domain, load_config, get_root
    from pathlib import Path

    root = get_root()

    # ── Merge: config.yaml run: section → CLI args (CLI wins) ────────────────
    try:
        cfg = load_config()
    except FileNotFoundError:
        cfg = {}
    run_cfg = cfg.get("run", {})

    seed_arg      = args.seed      or run_cfg.get("seed")
    condition_arg = args.condition or run_cfg.get("condition")
    domain_arg    = args.domain    or run_cfg.get("domain") or None
    output_dir_arg = (
        args.output_dir
        or run_cfg.get("output_dir")
        or None
    )
    # For flags that default to False in argparse, only override from config
    # when the CLI flag was NOT explicitly passed.
    skip_tstr = args.skip_tstr or bool(run_cfg.get("skip_tstr", False))
    skip_llm  = args.skip_llm  or bool(run_cfg.get("skip_llm",  False))

    # epsilon: CLI arg (None when not passed) > run_cfg > dp.epsilon > 1.0
    epsilon = (
        args.epsilon
        if args.epsilon is not None
        else run_cfg.get("epsilon")
        or cfg.get("dp", {}).get("epsilon", 1.0)
    )

    # rows: CLI arg > run_cfg > generation.total_rows > 1 000
    total_rows = (
        args.rows
        or run_cfg.get("rows")
        or cfg.get("generation", {}).get("total_rows", 1_000)
    )

    # Validate required values are now resolved
    if not seed_arg:
        logger.error(
            "--seed is required (or set run.seed in config.yaml)"
        )
        sys.exit(1)
    if not condition_arg:
        logger.error(
            "--condition is required (or set run.condition in config.yaml)"
        )
        sys.exit(1)

    # ── Resolve domain ────────────────────────────────────────────────────────
    if not domain_arg:
        slug = condition_arg.lower().replace(" ", "_").replace("-", "_")
        candidate = root / "domains" / f"{slug}.yaml"
        if candidate.exists():
            domain_arg = str(candidate)
            logger.info("  Domain config : %s", candidate)
    if domain_arg:
        set_domain(domain_arg)

    # Import pipeline AFTER domain is set — module-level config loads in
    # generate.py / validate.py / tstr.py will pick up the merged config
    from llm_synth.pipeline import run_pipeline

    seed_path = Path(seed_arg)
    if not seed_path.exists():
        logger.error("seed file not found: %s", seed_path)
        sys.exit(1)

    output_dir = Path(output_dir_arg) if output_dir_arg else root / "processed_data"

    run_pipeline(
        seed_file=str(seed_path),
        condition=condition_arg,
        output_dir=output_dir,
        epsilon=float(epsilon),
        total_rows=int(total_rows),
        skip_tstr=skip_tstr,
        skip_llm=skip_llm,
    )


def _cmd_profile(args: argparse.Namespace) -> None:
    from llm_synth.seed_statistics import extract_seed_stats
    from pathlib import Path

    out = args.output or str(Path(args.seed).with_suffix("").name + "_stats.json")
    extract_seed_stats(csv_path=args.seed, output_json=out)
    logger.info("Stats saved → %s", out)


def _cmd_dp(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path
    from llm_synth.dp_statistics import apply_dp

    stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    out_path = Path(args.output or args.stats.replace(".json", "_dp.json"))
    dp_stats = apply_dp(stats, epsilon=args.epsilon, seed=args.seed_val)
    out_path.write_text(json.dumps(dp_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("DP stats (ε=%s) saved → %s", args.epsilon, out_path)


def _cmd_generate(args: argparse.Namespace) -> None:
    from llm_synth.config import set_domain, load_config, get_root
    from pathlib import Path

    # Explicit domain flag takes priority; otherwise try to infer from stats / seed filename.
    # e.g. diab_stats_dp.json → tokens ["diab"] → exact match domains/diab.yaml (fails)
    #                         → prefix match  domains/diabetes.yaml (succeeds)
    domain_path = args.domain
    if not domain_path:
        root = get_root()
        domains_dir = root / "domains"

        def _find_domain(filename: str | None) -> str | None:
            if not filename or not domains_dir.exists():
                return None
            stem = Path(filename).stem
            clean = (stem
                     .replace("_dp_stats", "")
                     .replace("_stats_dp", "")
                     .replace("_stats",    "")
                     .replace("_seed",     ""))
            for token in [t for t in clean.split("_") if t]:
                # Exact match
                exact = domains_dir / f"{token}.yaml"
                if exact.exists():
                    return str(exact)
                # Prefix match (e.g. "diab" → "diabetes.yaml")
                hits = sorted(domains_dir.glob(f"{token}*.yaml"), key=lambda f: len(f.stem))
                if hits:
                    return str(hits[0])
            return None

        domain_path = _find_domain(args.stats) or _find_domain(args.seed)
        if domain_path:
            logger.info("  Domain config : %s", domain_path)
    if domain_path:
        set_domain(domain_path)

    from llm_synth.generate import generate_synthetic_data

    cfg     = load_config()
    gen_cfg = cfg.get("generation", {})
    val_cfg = cfg.get("validation", {})
    root    = get_root()

    # Resolve seed CSV: CLI --seed > domain validation.seed_csv > generation.seed_csv
    seed_csv = (
        args.seed
        or (str(root / val_cfg["seed_csv"]) if val_cfg.get("seed_csv") else None)
        or str(root / gen_cfg.get("seed_csv", "data/diab_seed.csv"))
    )

    generate_synthetic_data(
        total_rows=args.rows or gen_cfg.get("total_rows", 1_000),
        stats_file=args.stats or str(root / gen_cfg.get("stats_file", "data/diab_stats_dp.json")),
        output_csv=args.output or str(root / gen_cfg.get("output_csv", "processed_data/synthetic_output.csv")),
        seed_csv=seed_csv,
    )


def _cmd_validate(args: argparse.Namespace) -> None:
    from llm_synth.config import set_domain, load_config, get_root
    if args.domain:
        set_domain(args.domain)
    from llm_synth.validate import validate_synthetic_data, _print_report

    cfg     = load_config()
    val_cfg = cfg.get("validation", {})
    root    = get_root()

    report = validate_synthetic_data(
        synthetic_csv=args.synthetic or str(root / val_cfg.get("synthetic_csv", "processed_data/synthetic_output.csv")),
        seed_stats_json=args.stats or str(root / val_cfg.get("seed_stats_json", "data/diab_stats.json")),
        seed_csv=args.seed or str(root / val_cfg.get("seed_csv", "data/diab_seed.csv")),
        output_report=args.output or str(root / val_cfg.get("output_report", "diab_validation_report.json")),
        run_llm=not args.skip_llm,
    )
    _print_report(report)


def _cmd_tstr(args: argparse.Namespace) -> None:
    from llm_synth.config import set_domain
    if args.domain:
        set_domain(args.domain)
    from llm_synth.tstr import main as tstr_main
    tstr_main()


def _cmd_build_prompt(args: argparse.Namespace) -> None:
    from llm_synth.config import get_root
    from llm_synth.build_system_prompt import build_system_prompt
    from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-synth",
        description="Statistics-first, privacy-preserving synthetic data generation using LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--domain",
        default=None,
        metavar="PATH",
        help="Path to a domain config YAML (e.g. domains/kidney.yaml). "
             "Merged on top of config.yaml. 'run' auto-discovers domains/<condition>.yaml.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Full end-to-end pipeline")
    p_run.add_argument("--seed",       default=None,
                       help="Path to seed CSV (or set run.seed in config.yaml)")
    p_run.add_argument("--condition",  default=None,
                       help="Condition label — used as output folder name and domain auto-discovery key "
                            "(or set run.condition in config.yaml)")
    p_run.add_argument("--output-dir", dest="output_dir", default=None,
                       help="Base output directory (default: run.output_dir in config.yaml, "
                            "then processed_data/)")
    p_run.add_argument("--epsilon",    type=float, default=None,
                       help="Differential privacy budget ε (default: run.epsilon → dp.epsilon → 1.0)")
    p_run.add_argument("--rows",       type=int, default=None,
                       help="Total synthetic rows to generate "
                            "(default: run.rows → generation.total_rows in config.yaml)")
    p_run.add_argument("--skip-tstr",  dest="skip_tstr", action="store_true",
                       help="Skip the TSTR evaluation step (or set run.skip_tstr: true in config.yaml)")
    p_run.add_argument("--skip-llm",   dest="skip_llm",  action="store_true",
                       help="Skip LLM-based validation analysis — faster, no API call "
                            "(or set run.skip_llm: true in config.yaml)")

    # ── profile ───────────────────────────────────────────────────────────────
    p_prof = sub.add_parser("profile", help="Extract statistical profile from seed CSV")
    p_prof.add_argument("--seed",   required=True, help="Path to seed CSV")
    p_prof.add_argument("--output", default=None,  help="Output JSON path (default: <seed>_stats.json)")

    # ── dp ────────────────────────────────────────────────────────────────────
    p_dp = sub.add_parser("dp", help="Apply Laplace differential privacy to stats JSON")
    p_dp.add_argument("--stats",    required=True, help="Input stats JSON produced by 'profile'")
    p_dp.add_argument("--epsilon",  type=float, default=1.0,
                      help="Privacy budget ε (0.1=strong, 1.0=moderate, 10.0=weak; default: 1.0)")
    p_dp.add_argument("--seed-val", dest="seed_val", type=int, default=42,
                      help="RNG seed for reproducible noise (default: 42)")
    p_dp.add_argument("--output",   default=None,
                      help="Output JSON path (default: <stats>_dp.json)")

    # ── generate ──────────────────────────────────────────────────────────────
    p_gen = sub.add_parser("generate", help="Generate synthetic rows")
    p_gen.add_argument("--stats",  default=None, help="DP stats JSON (overrides config)")
    p_gen.add_argument("--rows",   type=int, default=None,
                       help="Total synthetic rows to generate (default: from config)")
    p_gen.add_argument("--output", default=None, help="Output CSV path (default: from config)")
    p_gen.add_argument("--seed",   default=None,
                       help="Seed CSV for stratified categorical pre-assignment "
                            "(default: data/diab_seed.csv)")

    # ── validate ──────────────────────────────────────────────────────────────
    p_val = sub.add_parser("validate", help="Validate synthetic CSV")
    p_val.add_argument("--synthetic", default=None,
                       help="Path to synthetic CSV (default: from config)")
    p_val.add_argument("--stats",     default=None, help="Seed stats JSON (default: from config)")
    p_val.add_argument("--seed",      default=None, help="Seed CSV for privacy checks (default: from config)")
    p_val.add_argument("--output",    default=None, help="Validation report JSON (default: from config)")
    p_val.add_argument("--skip-llm",  dest="skip_llm", action="store_true",
                       help="Skip LLM-based deep analysis (faster, no API call)")

    # ── tstr ──────────────────────────────────────────────────────────────────
    p_tstr = sub.add_parser("tstr", help="Train-on-Synthetic / Test-on-Real evaluation")
    p_tstr.add_argument("--extended", action="store_true",
                        help="Run extended multi-model analysis (LR, MLP, HGB) with AUROC, "
                             "distribution-shift robustness, and subgroup parity")

    # ── build-prompt ──────────────────────────────────────────────────────────
    p_bp = sub.add_parser(
        "build-prompt",
        help="Generate system prompt markdown from domain config, seed data, and soft rules",
    )
    p_bp.add_argument("condition", help="Domain slug matching domains/<condition>.yaml")
    p_bp.add_argument("--output", "-o", help="Output path (default: prompts/system_prompt_<condition>.md)")
    p_bp.add_argument("--top-rules", type=int, default=15,
                      help="Number of top soft rules to include (default: 15)")
    p_bp.add_argument("--exemplar-rows", type=int, default=7,
                      help="Number of exemplar rows to include (default: 7)")

    args = parser.parse_args()
    dispatch = {
        "run":          _cmd_run,
        "profile":      _cmd_profile,
        "dp":           _cmd_dp,
        "generate":     _cmd_generate,
        "validate":     _cmd_validate,
        "tstr":         _cmd_tstr,
        "build-prompt": _cmd_build_prompt,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
