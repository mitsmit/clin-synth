"""clin-synth: statistics-first, privacy-preserving synthetic data generation using LLMs."""

from __future__ import annotations

__version__ = "0.1.0"

from clin_synth.config import load_config, set_domain, get_root
from clin_synth.pipeline import run_pipeline
from clin_synth.seed_statistics import extract_seed_stats
from clin_synth.dp_statistics import apply_dp
from clin_synth.generate import generate_synthetic_data
from clin_synth.validate import validate_synthetic_data
from clin_synth.clinical_plausibility import (
    check_clinical_plausibility,
    check_rule_based_clinicalplausibility,
)

__all__ = [
    "__version__",
    # config
    "load_config",
    "set_domain",
    "get_root",
    # pipeline
    "run_pipeline",
    # pipeline stages
    "extract_seed_stats",
    "apply_dp",
    "generate_synthetic_data",
    "validate_synthetic_data",
    # plausibility
    "check_clinical_plausibility",
    "check_rule_based_clinicalplausibility",
]
