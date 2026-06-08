"""Deterministic tests for clin_synth.dp_statistics — Laplace mechanism and DP plumbing.

These exercise the noise/projection primitives directly with seeded RNGs, so
results are reproducible, and check the structural invariants the DP mechanism
must preserve (valid proportions, valid correlation matrices, determinism).
"""

import numpy as np
import pytest

from clin_synth.dp_statistics import (
    _laplace,
    _dp_proportions,
    _project_to_correlation_matrix,
    apply_dp,
)


# ── _laplace ──────────────────────────────────────────────────────────────────

def test_laplace_returns_zero_for_non_positive_scale():
    rng = np.random.default_rng(0)
    assert _laplace(0.0, rng) == 0.0
    assert _laplace(-1.0, rng) == 0.0


def test_laplace_is_reproducible_with_seeded_rng():
    a = _laplace(1.0, np.random.default_rng(42))
    b = _laplace(1.0, np.random.default_rng(42))
    assert a == b


# ── _dp_proportions ───────────────────────────────────────────────────────────

def test_dp_proportions_sum_to_one_and_are_non_negative():
    freq_table = {
        "Caucasian":       {"count": 765, "proportion": 0.765},
        "AfricanAmerican": {"count": 193, "proportion": 0.193},
        "Other":           {"count": 42,  "proportion": 0.042},
    }
    rng = np.random.default_rng(7)
    noised = _dp_proportions(freq_table, n=1000, epsilon=1.0, rng=rng)

    assert set(noised.keys()) == set(freq_table.keys())
    total = sum(v["proportion"] for v in noised.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    for v in noised.values():
        assert v["count"] >= 0
        assert v["proportion"] >= 0


def test_dp_proportions_handles_degenerate_zero_count_table():
    freq_table = {"OnlyCategory": {"count": 0, "proportion": 1.0}}
    rng = np.random.default_rng(1)
    noised = _dp_proportions(freq_table, n=1, epsilon=1.0, rng=rng)
    assert noised["OnlyCategory"]["proportion"] >= 0


# ── _project_to_correlation_matrix ────────────────────────────────────────────

def test_project_to_correlation_matrix_is_symmetric_psd_with_unit_diagonal():
    # A symmetric but indefinite matrix (negative eigenvalue), as DP noise produces
    noisy = np.array([
        [1.0,  0.95, -0.95],
        [0.95, 1.0,   0.95],
        [-0.95, 0.95, 1.0],
    ])
    corr = _project_to_correlation_matrix(noisy)

    assert np.allclose(corr, corr.T)
    assert np.allclose(np.diag(corr), 1.0)
    eigvals = np.linalg.eigvalsh(corr)
    assert (eigvals >= -1e-8).all()
    assert (corr >= -1.0 - 1e-9).all() and (corr <= 1.0 + 1e-9).all()


def test_project_to_correlation_matrix_is_idempotent_on_valid_matrix():
    valid = np.array([[1.0, 0.3], [0.3, 1.0]])
    projected = _project_to_correlation_matrix(valid)
    assert np.allclose(projected, valid, atol=1e-9)


# ── apply_dp ──────────────────────────────────────────────────────────────────

def _toy_stats() -> dict:
    return {
        "meta": {"n_rows": 500, "n_columns": 2},
        "columns": {
            "gender": {
                "semantic_type": "categorical",
                "n_total": 500,
                "n_non_null": 500,
                "frequency_table": {
                    "Female": {"count": 300, "proportion": 0.6},
                    "Male":   {"count": 200, "proportion": 0.4},
                },
                "mode": "Female",
                "mode_frequency": 0.6,
            },
            "age_years": {
                "semantic_type": "integer",
                "n_total": 500,
                "n_non_null": 480,
                "min": 0,
                "max": 100,
                "mean": 55.0,
                "median": 56.0,
                "std": 12.0,
                "quantiles": {"p1": 5.0, "p50": 56.0, "p99": 95.0},
            },
        },
    }


def test_apply_dp_is_deterministic_given_a_seed():
    stats = _toy_stats()
    out_a = apply_dp(stats, epsilon=1.0, seed=123)
    out_b = apply_dp(stats, epsilon=1.0, seed=123)
    assert out_a == out_b


def test_apply_dp_preserves_structure_and_records_provenance():
    stats = _toy_stats()
    dp_stats = apply_dp(stats, epsilon=0.5, seed=1)

    assert set(dp_stats["columns"].keys()) == set(stats["columns"].keys())
    assert dp_stats["dp"]["mechanism"] == "Laplace"
    assert dp_stats["dp"]["epsilon"] == 0.5
    assert dp_stats["dp"]["seed"] == 1

    # Categorical proportions remain a valid distribution after noising
    gender_table = dp_stats["columns"]["gender"]["frequency_table"]
    assert sum(v["proportion"] for v in gender_table.values()) == pytest.approx(1.0, abs=1e-6)

    # Numeric values stay within the declared [min, max] range
    age = dp_stats["columns"]["age_years"]
    assert stats["columns"]["age_years"]["min"] <= age["mean"] <= stats["columns"]["age_years"]["max"]
    assert age["std"] >= 0
