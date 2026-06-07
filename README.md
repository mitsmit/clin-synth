![llm-synth](assets/llm_synth_logo_grey_crimson.svg)

# llm-synth : Synthetic Clinical Data with LLM

A statistics-first pipeline that generates realistic synthetic clinical records
using an LLM as the sampler and a multi-layer validation framework to verify quality.

---

## Quick Start

### Install

```bash
pip install -e .
```

### API key

Add your OpenAI key to a `.env` file in the project root (git-ignored):

```bash
OPENAI_API_KEY=sk-...
```

### Run on the included sample

```bash
# Full pipeline — uses the 500-row diabetes sample, no download needed
llm-synth run \
    --seed data/samples/diabetes_sample.csv \
    --condition diabetes \
    --rows 500
```

See [`data/samples/README.md`](data/samples/README.md) for download links to other datasets (CHF, CKD).

### Config-file driven run

All `run` parameters can be set in `config.yaml` under the `run:` key so the pipeline
is fully reproducible with no CLI flags:

```yaml
# config.yaml
run:
  seed:       data/diab_seed.csv
  condition:  diabetes
  output_dir: processed_data
  domain:     ~              # ~ = auto-discover domains/<condition>.yaml
  epsilon:    1.0
  rows:       500
  skip_tstr:  false
  skip_llm:   false
```

Then run with no arguments:

```bash
llm-synth run
```

CLI flags always override config values, so you can mix both:

```bash
# Use all config.yaml defaults, but override rows and epsilon for this run
llm-synth run --rows 2000 --epsilon 0.5
```

**Precedence order** (highest → lowest) for each parameter:

| Parameter | 1st | 2nd | 3rd |
|---|---|---|---|
| `seed` | `--seed` | `run.seed` in config | — (error) |
| `condition` | `--condition` | `run.condition` in config | — (error) |
| `epsilon` | `--epsilon` | `run.epsilon` in config | `dp.epsilon` in config |
| `rows` | `--rows` | `run.rows` in config | `generation.total_rows` in config |
| `output_dir` | `--output-dir` | `run.output_dir` in config | `processed_data/` |
| `domain` | `--domain` | `run.domain` in config | auto-discover |
| `skip_tstr` | `--skip-tstr` | `run.skip_tstr` in config | `false` |
| `skip_llm` | `--skip-llm` | `run.skip_llm` in config | `false` |

### Step-by-step

```bash
llm-synth profile --seed data/diab_seed.csv

llm-synth --domain domains/diabetes.yaml dp \
    --stats data/diab_stats.json --epsilon 1.0

llm-synth --domain domains/diabetes.yaml generate \
    --stats data/diab_stats_dp.json --rows 10000

llm-synth --domain domains/diabetes.yaml validate \
    --synthetic processed_data/synthetic_output.csv

llm-synth --domain domains/diabetes.yaml tstr   # optional
```

---

## Pipeline

```mermaid
flowchart LR
    A([Seed CSV]) -->|"_stats.json"| B["1 · profile"]
    B -->|"_stats_dp.json"| C["2 · dp"]
```

```mermaid
flowchart LR
    C["2 · dp"] -->|"_synthetic.csv"| D["3 · generate"]
    D -->|"_validation_report.json"| E["4 · validate"]
    E -->|"_tstr_comparison.json"| F["5 · tstr"]
    F --> G([Results])
```

| Stage | What it does |
|---|---|
| **profile** | Extracts per-column distributions, correlations, and missingness from the seed CSV |
| **dp** | Applies Laplace differential privacy (ε configurable) before statistics reach the model |
| **generate** | Injects DP statistics into an LLM prompt and generates synthetic rows in batches |
| **validate** | Checks schema, distribution fidelity, correlation fidelity, privacy, and clinical plausibility |
| **tstr** | Train-on-Synthetic / Test-on-Real predictive evaluation; reports an F1-macro fidelity ratio |

---

## Configuration

| File | Purpose |
|---|---|
| `config.yaml` | Pipeline-wide settings: model, batch size, DP epsilon, validation thresholds |
| `domains/<condition>.yaml` | Disease-specific settings: schema, file paths, TSTR target, clinical rules |

Pipeline settings are shared across all conditions. Domain configs are swapped per run.

---

## Prompt Generation

The `generate` stage is driven by two distinct prompts, built in very different ways:

| | System prompt | Per-batch user message |
|---|---|---|
| **Built by** | `build_system_prompt.py`, run via `llm-synth build-prompt <condition>` | `build_user_message.py`, called automatically inside `generate`'s batch loop |
| **When** | Explicitly, ahead of time — a one-off step you (re-)run when inputs change | Freshly, for every single batch of every `generate` run — no manual step |
| **Driven by** | `condition` → `domains/<condition>.yaml` + stats JSON (from `profile`) + seed CSV exemplar rows + mined soft-rule CSVs | DP-stats JSON (categorical/numeric distributions, correlations) + a rotating archetype + the batch counter |
| **Output** | Written to disk: `prompts/system_prompt_<condition>.md` | Ephemeral — composed in memory and sent straight to the LLM; never saved |
| **Format** | Fixed sections: schema & missingness, clinical rules, exemplar rows, etc. | Fixed sections: categorical/numeric distributions → critical reminders → correlations → archetype instruction → diversity instruction → "Generate N rows. Batch X of Y." |

Generate (or regenerate) the system prompt for a condition with:

```bash
llm-synth build-prompt <condition> [--top-rules 15] [--exemplar-rows 7]
```

This gives you a consistent, **condition- and seed-driven** baseline assembled
from your domain config, seed statistics, and mined rules — in the fixed
section format above. Hand-editing the resulting `.md` afterwards to encode
domain expertise the generator can't infer from raw statistics (clinical
nuance, edge cases) is expected and supported: `build-prompt` produces a
reproducible *starting point*, not a one-way, no-touch pipeline.

---

## Running for a New Disease Area

Only three things are needed to run the pipeline on a new dataset.

### 1 — Add a seed CSV

```
data/chf_seed.csv
```

The seed is only used to extract statistics and is never passed to the model.

### 2 — Create a domain config

```bash
cp domains/diabetes.yaml domains/chf.yaml
```

Edit `domains/chf.yaml` and update:

- **`schema`** — column names, categorical columns, missingness columns
- **File paths** — `dp`, `generation`, `validation` output paths
- **`tstr`** — target column, target class order, ordinal column encodings
- **`clinical_plausibility`** — numeric bounds, group comparison rules, correlation pairs

### 3 — Generate the system prompt, then refine it

```bash
llm-synth profile --seed data/chf_seed.csv --output processed_data/chf_stats.json
llm-synth build-prompt chf
```

This builds `prompts/system_prompt_chf.md` from `domains/chf.yaml`, the seed
statistics, and exemplar rows — see [Prompt Generation](#prompt-generation)
for the fixed format it follows. Review the result and hand-edit it for
domain-specific nuance (clinical logic, edge cases, valid value ranges) the
generator can't infer from statistics alone.

### Run

```bash
llm-synth run \
    --seed data/chf_seed.csv \
    --condition chf \
    --rows 5000
```

### Files to create or change

| File | Action |
|---|---|
| `data/chf_seed.csv` | **Add** your seed dataset |
| `domains/chf.yaml` | **Add** domain config (copy from diabetes) |
| `prompts/system_prompt_chf.md` | **Generate** via `llm-synth build-prompt chf`, then refine |
| `config.yaml` | No change needed |

---

## CLI Reference

```bash
llm-synth run [--seed <path>] [--condition <name>] [options]
```

| Argument | Default | Description |
|---|---|---|
| `--seed` | `run.seed` in config | Path to seed CSV |
| `--condition` | `run.condition` in config | Condition name; auto-discovers `domains/<condition>.yaml` |
| `--domain` | `run.domain` → auto-discover | Explicit path to domain config YAML |
| `--epsilon` | `run.epsilon` → `dp.epsilon` → `1.0` | Differential privacy budget |
| `--rows` | `run.rows` → `generation.total_rows` | Number of synthetic rows to generate |
| `--output-dir` | `run.output_dir` → `processed_data/` | Base output directory |
| `--skip-tstr` | `run.skip_tstr` → `false` | Skip TSTR evaluation |
| `--skip-llm` | `run.skip_llm` → `false` | Skip LLM deep-analysis in validation |

`--seed` and `--condition` are required if not set in `config.yaml`.

