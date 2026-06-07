"""
generate.py
===========
Generate synthetic patient records using the OpenAI API.

The system prompt (instructions, schema, exemplars) is loaded from
prompts/system_prompt.md. Per-batch statistical profiles are built from
data/diab_stats_dp.json by build_user_message.py.

Usage:
    python scripts/generate.py
    python scripts/generate.py --total 5000 --batch 100 --output out.csv

Requirements:
    pip install openai python-dotenv pyyaml
    OPENAI_API_KEY set in .env at project root
"""

import argparse
import concurrent.futures
import csv
import io
import json
import logging
import re
import time
from pathlib import Path

log = logging.getLogger(__name__)

import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

from llm_synth.build_user_message import build_user_message, build_stratified_user_message, _init_build_cfg
from llm_synth.utils.stratified_sampler import sample_strata, _init_strata_cfg
from llm_synth.utils.postprocess_utils import (
    _enforce_numeric_distributions,
    _enforce_correlations,
    _enforce_strata,
    _init_postprocess_cfg,
)
from llm_synth.config import load_config, get_root

_ROOT = get_root()
load_dotenv(_ROOT / ".env")

def _make_client() -> OpenAI:
    cfg = load_config()
    backend = cfg.get("llm", {}).get("backend", "openai")
    if backend == "ollama":
        base_url = cfg.get("llm", {}).get("ollama_base_url", "http://localhost:11434/v1")
        return OpenAI(base_url=base_url, api_key="ollama")
    return OpenAI()

client = _make_client()


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

#  Loading the system prompt.md from the prompt folder.
def load_system_prompt(prompt_path: str) -> str:
    """Load and clean the system prompt (instructions only — no stats)."""
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path.resolve()}")
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return text.strip()

# loading the  statistical profile to build the user pprompt
def load_stats(stats_path: str) -> dict:
    """Load the seed statistics JSON used to build per-batch user messages."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path.resolve()}")
    return json.loads(path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Row parser: parse and clean rows returned by the LLM.
# ─────────────────────────────────────────────────────────────────────────────

# Matches hex patient-ID hashes the LLM occasionally emits as an extra column
_PATIENT_ID_RE = re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE)

def parse_csv_rows(raw_text: str, expected_cols: list[str]) -> tuple[list[list[str]], list[str]]:
    """
    Parse raw CSV text returned by the LLM.
    Returns (valid_rows, malformed_lines).

    The LLM occasionally prefixes rows with a patient-ID hash despite the
    system prompt saying not to. Two cases are handled:
    - 14-col row where col 0 is a hex ID: strip col 0, recover the 13 data cols
      (race is still present in col 1 of the raw row).
    - 13-col row where col 0 is a hex ID: race is irrecoverable — reject.
    """
    valid, malformed = [], []
    reader = csv.reader(io.StringIO(raw_text.strip()))
    for line in reader:
        if not line:
            continue
        col0 = line[0].strip().lower()
        # Skip header rows the LLM may have accidentally emitted
        if col0 in ("patient_id", "race"):
            continue

        cells = [c.strip() for c in line]

        # LLM added patient_id as an extra leading column — strip it
        if len(cells) == len(expected_cols) + 1 and _PATIENT_ID_RE.match(cells[0]):
            cells = cells[1:]

        # LLM stripped race/gender/age from the front (stratified mode failure):
        # pad with empty strings so _enforce_strata can fill correct values.
        if len(cells) == len(expected_cols) - 3:
            cells = ["", "", ""] + cells

        if len(cells) == len(expected_cols):
            # Reject rows where race slot still contains a patient-ID hash
            if _PATIENT_ID_RE.match(cells[0]):
                malformed.append(",".join(cells))
            else:
                valid.append(cells)
        elif any(cells):
            malformed.append(",".join(cells))

    return valid, malformed



# ─────────────────────────────────────────────────────────────────────────────
# Per-batch worker, generate 1 batch of rows (runs in a thread)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_batch(
    batch_num: int,
    rows_this_batch: int,
    n_batches: int,
    system_prompt: str,
    stats: dict,
    model: str,
    max_completion_tokens: int,
    max_retries: int,
    retry_delay: float,
    temperature: float,
    min_batch_yield: float,
    batch_categoricals: list[dict] | None = None,
    expected_cols: list[str] | None = None,
) -> tuple[int, list[list[str]], int]:
    """
    Call the API for one batch with retries.
    Returns (batch_num, valid_rows, n_malformed).
    Thread-safe: system_prompt, stats, and batch_categoricals are read-only.

    When batch_categoricals is provided, stratified generation is used:
    categorical values are pre-assigned and the LLM only fills numeric slots.
    """
    if batch_categoricals is not None:
        user_msg = build_stratified_user_message(
            stats=stats,
            batch_categoricals=batch_categoricals,
            batch_num=batch_num + 1,
            n_batches=n_batches,
        )
    else:
        user_msg = build_user_message(
            stats=stats,
            batch_size=rows_this_batch,
            batch_num=batch_num + 1,
            n_batches=n_batches,
        )

    rows: list[list[str]] = []
    n_malformed = 0

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw_text = response.choices[0].message.content
            rows, malformed = parse_csv_rows(raw_text, expected_cols)
            n_malformed = len(malformed)

            if malformed:
                log.warning("[batch %d] %d malformed lines skipped", batch_num + 1, len(malformed))

            # Pin categorical strata immediately so LLM drift cannot change values.
            if batch_categoricals is not None and rows:
                col_idx = {col: i for i, col in enumerate(expected_cols)}
                drift_cols = [c for c in batch_categoricals[0] if c in col_idx]
                drift_counts = {c: 0 for c in drift_cols}
                for i, row in enumerate(rows[:len(batch_categoricals)]):
                    for c in drift_cols:
                        cidx = col_idx.get(c)
                        if cidx is not None and c in batch_categoricals[i]:
                            if row[cidx].strip() != batch_categoricals[i][c]:
                                drift_counts[c] += 1
                n = len(rows)
                for c, cnt in drift_counts.items():
                    if cnt > 0:
                        log.warning("[batch %d] LLM changed %s in %d/%d rows (%.0f%%) — pinning back to strata",
                                    batch_num + 1, c, cnt, n, cnt / n * 100)
                rows = _enforce_strata(rows, batch_categoricals[:len(rows)], col_idx)

            if len(rows) < rows_this_batch * min_batch_yield:
                log.warning("[batch %d] Only %d/%d rows returned (attempt %d). Retrying...",
                            batch_num + 1, len(rows), rows_this_batch, attempt)
                time.sleep(retry_delay)
                continue

            # Canary: warn if any readmitted class drifts from expected seed proportions.
            if "readmitted" in expected_cols:
                readmitted_idx = expected_cols.index("readmitted")
                readmitted_vals = [r[readmitted_idx] for r in rows if len(r) > readmitted_idx]
                n_obs = max(len(readmitted_vals), 1)
                gt30_rate = readmitted_vals.count(">30") / n_obs
                lt30_rate = readmitted_vals.count("<30") / n_obs
                if gt30_rate < 0.25:
                    log.warning("[batch %d] >30 rate=%.1f%% (expected ~35%%) — possible class drift",
                                batch_num + 1, gt30_rate * 100)
                if lt30_rate < 0.07:
                    log.warning("[batch %d] <30 rate=%.1f%% (expected ~11%%) — minority class under-generated",
                                batch_num + 1, lt30_rate * 100)

            break  # success

        except Exception as e:
            log.error("[batch %d] API error (attempt %d): %s", batch_num + 1, attempt, e)
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                log.error("[batch %d] All retries exhausted. Skipping.", batch_num + 1)

    return batch_num, rows, n_malformed


# ─────────────────────────────────────────────────────────────────────────────
# Core generation loop (fires batches concurrently, collects results writes CSV)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(
    total_rows:            int   | None = None,
    batch_size:            int   | None = None,
    output_csv:            str   | None = None,
    system_prompt_file:    str   | None = None,
    stats_file:            str   | None = None,
    model:                 str   | None = None,
    max_completion_tokens: int   | None = None,
    max_retries:           int   | None = None,
    retry_delay:           float | None = None,
    max_workers:           int   | None = None,
    temperature:           float | None = None,
    min_batch_yield:       float | None = None,
    seed_csv:              str   | None = None,
    progress_callback=None,
) -> None:
    """
    Generate `total_rows` synthetic patient records and write to `synthetic_output.csv`.

    Parameters
    ----------
    total_rows          : Total synthetic rows to generate.
    batch_size          : Rows per API call. Keep ≤ 500.
    output_csv          : Destination CSV file path.
    system_prompt_file  : Path to system prompt (instructions, schema, exemplars).
    stats_file          : Path to seed stats JSON (injected into each user message).
    model               : OpenAI model.
    max_retries         : Retries on API error or malformed output.
    retry_delay         : Seconds between retries.
    max_workers         : Concurrent API calls.
    """
    # ── Resolve config (deferred so domain config is available at call time) ─
    _cfg     = load_config()
    _gen_cfg = _cfg.get("generation", {})
    root     = get_root()

    _init_strata_cfg()
    _init_postprocess_cfg()
    _init_build_cfg()

    expected_cols: list[str] = _cfg.get("schema", {}).get("expected_columns", [])

    if total_rows            is None: total_rows            = _gen_cfg.get("total_rows",            10_000)
    if batch_size            is None: batch_size            = _gen_cfg.get("batch_size",            100)
    if output_csv            is None: output_csv            = str(root / _gen_cfg.get("output_csv",            "processed_data/synthetic_output.csv"))
    if system_prompt_file    is None: system_prompt_file    = str(root / _gen_cfg.get("system_prompt_file",    "prompts/system_prompt.md"))
    if stats_file            is None: stats_file            = str(root / _gen_cfg.get("stats_file",            "data/diab_stats_dp.json"))
    if model                 is None: model                 = _gen_cfg.get("model",                 "gpt-4o-mini")
    if max_completion_tokens is None: max_completion_tokens = _gen_cfg.get("max_completion_tokens", 15000)
    if max_retries           is None: max_retries           = _gen_cfg.get("max_retries",           3)
    if retry_delay           is None: retry_delay           = _gen_cfg.get("retry_delay",           2.0)
    if max_workers           is None: max_workers           = _gen_cfg.get("max_workers",           5)
    if temperature           is None: temperature           = _gen_cfg.get("temperature",           0.9)
    if min_batch_yield       is None: min_batch_yield       = _gen_cfg.get("min_batch_yield",       0.9)
    if seed_csv              is None: seed_csv              = str(root / _gen_cfg.get("seed_csv",   "data/diab_seed.csv"))

    # ── Load system prompt (static — cached across all batches) ───────────

    log.info("Loading system prompt : %s", system_prompt_file)
    system_prompt = load_system_prompt(system_prompt_file)

    # ── Load seed stats (injected as JSON into every user message) ────────

    log.info("Loading seed stats    : %s", stats_file)
    stats = load_stats(stats_file)
    log.info("  %d columns profiled", len(stats.get("columns", {})))

    # ── Load seed CSV and pre-assign strata ───────────────────────────────
    seed_path = Path(seed_csv)
    seed_df: pd.DataFrame | None = None
    all_strata: list[dict] | None = None

    if seed_path.exists():
        seed_df = pd.read_csv(seed_path)
        log.info("Pre-assigning categorical strata from seed (%d seed rows)...", len(seed_df))
        all_strata = sample_strata(total_rows, seed_df, seed=42)
        log.info("  Strata assigned for %d rows (stratified generation enabled)", len(all_strata))
    else:
        raise FileNotFoundError(
            f"Seed CSV not found: {seed_csv!r}\n"
            f"Stratified generation requires the seed CSV to preserve the 'readmitted' "
            f"class distribution. Pass --seed <path> or set generation.seed_csv in config.yaml."
        )

    # ── Setup ─────────────────────────────────────────────────────────────
    n_batches = (total_rows + batch_size - 1) // batch_size
    # pre-compute exact row count per batch so workers don't share mutable state
    batch_sizes = [
        min(batch_size, total_rows - i * batch_size) for i in range(n_batches)
    ]

    # Slice strata into per-batch lists (mirrors batch_sizes slicing)
    strata_by_batch: list[list[dict] | None] = []
    if all_strata is not None:
        offset = 0
        for bs in batch_sizes:
            strata_by_batch.append(all_strata[offset: offset + bs])
            offset += bs
    else:
        strata_by_batch = [None] * n_batches

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "Generation plan: total=%d  batch=%d  batches=%d  workers=%d  model=%s  mode=%s  output=%s",
        total_rows, batch_size, n_batches, max_workers, model,
        "stratified" if all_strata else "non-stratified",
        output_path.resolve(),
    )

    # ── Fire all batches concurrently ─────────────────────────────────────
    results: dict[int, tuple[list[list[str]], int]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_batch,
                batch_num, batch_sizes[batch_num], n_batches,
                system_prompt, stats, model, max_completion_tokens, max_retries, retry_delay, temperature, min_batch_yield,
                strata_by_batch[batch_num], expected_cols,
            ): batch_num
            for batch_num in range(n_batches)
        }
        batches_done = 0
        rows_done    = 0
        for future in concurrent.futures.as_completed(futures):
            batch_num, rows, n_malformed = future.result()
            results[batch_num] = (rows, n_malformed)
            batches_done += 1
            rows_done    += len(rows)
            log.info("[batch %4d/%d] done — %d rows", batch_num + 1, n_batches, len(rows))
            if progress_callback is not None:
                progress_callback(batches_done, n_batches, rows_done, total_rows)

    # ── Collect rows ──────────────────────────────────────────────────────
    total_malformed = 0
    all_rows: list[list[str]] = []

    for batch_num in range(n_batches):
        rows, n_malformed = results.get(batch_num, ([], 0))
        total_malformed += n_malformed
        all_rows.extend(rows[:batch_sizes[batch_num]])

    col_indices = {col: i for i, col in enumerate(expected_cols)}

    # ── Numeric distributions and correlations ────────────────────────────
    log.info("Enforcing numeric distributions (quantile normalization)...")
    all_rows = _enforce_numeric_distributions(all_rows, col_indices, seed_df)
    log.info("Enforcing correlation structure (Iman-Conover)...")
    all_rows = _enforce_correlations(
        all_rows, col_indices, seed_df,
        stratified=all_strata is not None,
    )

    # Append mode: write header only if the file doesn't exist yet
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with open(output_path, "a", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        if not file_exists:
            writer.writerow(expected_cols)
        writer.writerows(all_rows)

    total_written = len(all_rows)

    # ── Post-write distribution check on the combined output file ─────────
    # Verifies that accumulated rows (across all append runs) still match the
    # seed marginal for `readmitted` within ±3 pp. Catches drift from prior
    # non-stratified or mis-configured runs that were appended silently.
    if seed_df is not None and "readmitted" in expected_cols:
        try:
            combined_df = pd.read_csv(output_path)
            if "readmitted" in combined_df.columns:
                combined_counts = combined_df["readmitted"].value_counts(normalize=True)
                seed_counts     = seed_df["readmitted"].value_counts(normalize=True)
                _DRIFT_THRESHOLD = 0.03
                log.info("Post-write distribution check (combined file, n=%d):", len(combined_df))
                any_drift = False
                for cls in seed_counts.index:
                    seed_p     = seed_counts.get(cls, 0.0)
                    combined_p = combined_counts.get(cls, 0.0)
                    delta = combined_p - seed_p
                    if abs(delta) > _DRIFT_THRESHOLD:
                        log.warning("  %s: combined=%.1f%%  seed=%.1f%%  delta=%+.1f%%  *** DRIFT ***",
                                    cls, combined_p * 100, seed_p * 100, delta * 100)
                        any_drift = True
                    else:
                        log.info("  %s: combined=%.1f%%  seed=%.1f%%  delta=%+.1f%%",
                                 cls, combined_p * 100, seed_p * 100, delta * 100)
                if any_drift:
                    log.warning("Combined output drifts >±%.0f%% from seed marginal "
                                "— consider clearing the output file and re-generating.",
                                _DRIFT_THRESHOLD * 100)
        except Exception as e:
            log.error("[post-write check] Could not read combined file: %s", e)

    # ── Final summary ─────────────────────────────────────────────────────
    log.info("Generation complete — rows_written=%d  malformed=%d  output=%s%s",
             total_written, total_malformed, output_path.resolve(),
             "  (appended)" if file_exists else "")
    if total_written < total_rows:
        log.warning("Expected %d rows but only wrote %d. Re-run to top up.",
                    total_rows, total_written)


# ─────────────────────────────────────────────────────────────────────────────
# CLI, parse argument and call main generation function
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments, falling back to config.yaml values for any unspecified option."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient benefit records using the OpenAI API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _cfg = load_config()
    _gen = _cfg.get("generation", {})
    root = get_root()
    parser.add_argument("--total",        type=int, default=_gen.get("total_rows",  10_000),
                        help="Total rows to generate")
    parser.add_argument("--batch",        type=int, default=_gen.get("batch_size",  100),
                        help="Rows per API call (keep ≤ 100)")
    parser.add_argument("--output",       type=str, default=_gen.get("output_csv",  "synthetic_output.csv"),
                        help="Output CSV file path")
    parser.add_argument("--system-prompt",type=str, default=str(root / _gen.get("system_prompt_file", "prompts/system_prompt.md")),
                        help="Path to system prompt markdown file")
    parser.add_argument("--stats",        type=str, default=str(root / _gen.get("stats_file", "data/diab_stats_dp.json")),
                        help="Path to seed statistics JSON file")
    parser.add_argument("--model",        type=str, default=_gen.get("model",       "gpt-4o-mini"),
                        help="OpenAI model for generation")
    parser.add_argument("--max-completion-tokens",   type=int, default=_gen.get("max_completion_tokens",  15000),
                        help="Max output tokens per API call")
    parser.add_argument("--retries",      type=int,   default=_gen.get("max_retries",  3),
                        help="Max retries per batch on failure")
    parser.add_argument("--retry-delay",  type=float, default=_gen.get("retry_delay",  2.0),
                        help="Seconds to wait between retries")
    parser.add_argument("--workers",      type=int,   default=_gen.get("max_workers",  5),
                        help="Concurrent API calls (raise to 10 for large runs)")
    parser.add_argument("--temperature",  type=float, default=_gen.get("temperature", 0.9),
                        help="Diversity control (0=deterministic, 1=max variety)")
    parser.add_argument("--min-batch-yield", type=float, default=_gen.get("min_batch_yield", 0.9),
                        help="Retry a batch if fewer than this fraction of rows were returned")
    parser.add_argument("--seed", type=str, default=str(root / "data" / "diab_seed.csv"),
                        help="Path to seed CSV for numeric distribution and correlation enforcement")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    generate_synthetic_data(
        total_rows         = args.total,
        batch_size         = args.batch,
        output_csv         = args.output,
        system_prompt_file = args.system_prompt,
        stats_file         = args.stats,
        model              = args.model,
        max_completion_tokens = args.max_completion_tokens,
        max_retries        = args.retries,
        retry_delay        = args.retry_delay,
        max_workers        = args.workers,
        temperature        = args.temperature,
        min_batch_yield    = args.min_batch_yield,
        seed_csv           = args.seed,
    )
