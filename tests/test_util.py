"""Deterministic tests for clin_synth.utils.util — PSI drift metrics and hash-column detection."""

import pandas as pd
import pytest

from clin_synth.utils.util import (
    _is_hash_column,
    _psi_categorical,
    _psi_numeric,
    _psi_status,
)


# ── _psi_status ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("psi, expected", [
    (0.0,  "stable"),
    (0.05, "stable"),
    (0.10, "monitor"),   # boundary is exclusive on the lower bucket
    (0.15, "monitor"),
    (0.20, "unstable"),  # boundary is exclusive on the monitor bucket
    (0.50, "unstable"),
])
def test_psi_status_buckets(psi, expected):
    assert _psi_status(psi) == expected


# ── _psi_categorical ──────────────────────────────────────────────────────────

def test_psi_categorical_is_zero_for_identical_distribution():
    seed_freq = {
        "Female": {"count": 60, "proportion": 0.6},
        "Male":   {"count": 40, "proportion": 0.4},
    }
    synth = pd.Series(["Female"] * 60 + ["Male"] * 40)
    psi, n_bins = _psi_categorical(seed_freq, synth)
    assert psi == pytest.approx(0.0, abs=1e-6)
    assert n_bins == 2


def test_psi_categorical_is_positive_for_shifted_distribution():
    seed_freq = {
        "Female": {"count": 60, "proportion": 0.6},
        "Male":   {"count": 40, "proportion": 0.4},
    }
    shifted = pd.Series(["Female"] * 10 + ["Male"] * 90)
    psi, _ = _psi_categorical(seed_freq, shifted)
    assert psi > 0.10


def test_psi_categorical_accounts_for_novel_categories_via_other_bucket():
    seed_freq = {"Female": {"count": 100, "proportion": 1.0}}
    synth = pd.Series(["Female"] * 50 + ["Nonbinary"] * 50)
    psi, n_bins = _psi_categorical(seed_freq, synth)
    assert psi > 0
    assert n_bins == 2  # the declared category + an implicit "other" bucket


# ── _psi_numeric ──────────────────────────────────────────────────────────────

def test_psi_numeric_is_near_zero_when_synthetic_matches_seed_quantiles():
    seed_col_stats = {
        "quantiles": {"p1": 0.0, "p25": 25.0, "p50": 50.0, "p75": 75.0, "p99": 100.0},
    }
    # Uniform sample over [0, 100] roughly matches uniform quantile spacing
    synth = pd.Series(range(0, 101))
    psi, n_bins = _psi_numeric(seed_col_stats, synth)
    assert n_bins > 0
    assert psi < 0.10


def test_psi_numeric_returns_zero_when_fewer_than_two_quantiles_known():
    seed_col_stats = {"quantiles": {"p50": 50.0}}
    synth = pd.Series([10, 20, 30])
    psi, n_bins = _psi_numeric(seed_col_stats, synth)
    assert (psi, n_bins) == (0.0, 0)


def test_psi_numeric_returns_zero_when_synthetic_series_is_empty_after_coercion():
    seed_col_stats = {"quantiles": {"p1": 0.0, "p99": 100.0}}
    synth = pd.Series(["not", "numeric", "values"])
    psi, n_bins = _psi_numeric(seed_col_stats, synth)
    assert (psi, n_bins) == (0.0, 0)


# ── _is_hash_column ───────────────────────────────────────────────────────────

def test_is_hash_column_detects_hex_hash_ids():
    series = pd.Series(["a" * 32, "b" * 32, "c" * 32, "1234567890abcdef1234567890abcdef"])
    assert _is_hash_column(series) is True


def test_is_hash_column_rejects_normal_categorical_values():
    series = pd.Series(["Caucasian", "AfricanAmerican", "Asian", "Other"])
    assert _is_hash_column(series) is False
