"""Deterministic tests for clin_synth.config._deep_merge — pipeline/domain config merge order.

This is the function behind load_config()'s "config.yaml ← domain config"
merge, so getting its precedence rules right is load-bearing for every domain.
"""

from clin_synth.config import _deep_merge


def test_deep_merge_overrides_scalar_values():
    base = {"epsilon": 1.0, "name": "diabetes"}
    override = {"epsilon": 0.5}
    assert _deep_merge(base, override) == {"epsilon": 0.5, "name": "diabetes"}


def test_deep_merge_recursively_merges_nested_dicts():
    base = {"generation": {"total_rows": 1000, "model": "gpt-4o-mini"}}
    override = {"generation": {"total_rows": 5000}}

    merged = _deep_merge(base, override)

    assert merged == {"generation": {"total_rows": 5000, "model": "gpt-4o-mini"}}


def test_deep_merge_replaces_lists_wholesale_not_elementwise():
    base = {"schema": {"expected_columns": ["a", "b", "c"]}}
    override = {"schema": {"expected_columns": ["x", "y"]}}

    merged = _deep_merge(base, override)

    assert merged["schema"]["expected_columns"] == ["x", "y"]


def test_deep_merge_adds_new_keys_from_override():
    base = {"dp": {"epsilon": 1.0}}
    override = {"tstr": {"target_column": "ckd_stage"}}

    merged = _deep_merge(base, override)

    assert merged == {"dp": {"epsilon": 1.0}, "tstr": {"target_column": "ckd_stage"}}


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}

    _deep_merge(base, override)

    assert base == {"a": {"x": 1}}
    assert override == {"a": {"y": 2}}
