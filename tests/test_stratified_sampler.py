"""Deterministic tests for clin_synth.utils.stratified_sampler — strata sampling for generation.

compute_conditional_distributions / sample_strata read module-level config
(_CAT_COLS, _COLS_WITH_MISSINGNESS, _CONDITIONING_COL), normally populated by
_init_strata_cfg() from the active domain config. Tests monkeypatch these
directly so behaviour is independent of whichever domain config happens to be
active in the environment.
"""

import pandas as pd
import pytest

import clin_synth.utils.stratified_sampler as ss
from clin_synth.utils.stratified_sampler import (
    _normalise,
    compute_conditional_distributions,
    sample_strata,
)


# ── _normalise ────────────────────────────────────────────────────────────────

def test_normalise_produces_probabilities_summing_to_one():
    out = _normalise({"a": 3, "b": 1})
    assert out == {"a": 0.75, "b": 0.25}
    assert sum(out.values()) == pytest.approx(1.0)


def test_normalise_returns_input_unchanged_for_zero_total():
    counts = {"a": 0, "b": 0}
    assert _normalise(counts) == counts


def test_normalise_single_key_is_certain():
    assert _normalise({"only": 7}) == {"only": 1.0}


# ── compute_conditional_distributions ─────────────────────────────────────────

@pytest.fixture
def toy_seed_df():
    return pd.DataFrame({
        "readmitted": ["NO", "NO", "NO", ">30", ">30", "<30"],
        "gender":     ["Female", "Female", "Male", "Male", "Female", "Male"],
        "race":       ["Caucasian", "Caucasian", "Other", "Caucasian", None, "Caucasian"],
    })


@pytest.fixture
def patched_strata_cfg(monkeypatch, toy_seed_df):
    monkeypatch.setattr(ss, "_CONDITIONING_COL", "readmitted")
    monkeypatch.setattr(ss, "_CAT_COLS", ["gender"])
    monkeypatch.setattr(ss, "_COLS_WITH_MISSINGNESS", ["race"])
    return toy_seed_df


def test_conditional_distributions_normalise_per_conditioning_value(patched_strata_cfg):
    dist = compute_conditional_distributions(patched_strata_cfg)

    # Simple categorical: P(gender | readmitted=NO) sums to 1
    no_gender = dist["gender"]["NO"]
    assert sum(no_gender.values()) == pytest.approx(1.0)
    assert no_gender["Female"] == pytest.approx(2 / 3)
    assert no_gender["Male"] == pytest.approx(1 / 3)


def test_conditional_distributions_compute_missingness_rate(patched_strata_cfg):
    dist = compute_conditional_distributions(patched_strata_cfg)

    # race is missing for exactly 1 of the 2 rows where readmitted == ">30"
    race_gt30 = dist["race"][">30"]
    assert race_gt30["_missing_rate"] == pytest.approx(0.5)
    assert sum(race_gt30["_proportions"].values()) == pytest.approx(1.0)


# ── sample_strata ─────────────────────────────────────────────────────────────

def test_sample_strata_returns_exactly_n_rows(patched_strata_cfg):
    strata = sample_strata(n_rows=25, seed_df=patched_strata_cfg, seed=42)
    assert len(strata) == 25
    assert all(set(row.keys()) == {"readmitted", "gender", "race"} for row in strata)


def test_sample_strata_is_reproducible_with_same_seed(patched_strata_cfg):
    a = sample_strata(n_rows=20, seed_df=patched_strata_cfg, seed=99)
    b = sample_strata(n_rows=20, seed_df=patched_strata_cfg, seed=99)
    assert a == b


def test_sample_strata_differs_across_seeds(patched_strata_cfg):
    a = sample_strata(n_rows=20, seed_df=patched_strata_cfg, seed=1)
    b = sample_strata(n_rows=20, seed_df=patched_strata_cfg, seed=2)
    assert a != b
