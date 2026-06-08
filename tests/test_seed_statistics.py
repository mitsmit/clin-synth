"""Deterministic tests for clin_synth.seed_statistics._infer_col_type — pure type inference."""

import pandas as pd

from clin_synth.seed_statistics import _infer_col_type


def test_infers_integer():
    assert _infer_col_type(pd.Series([1, 2, 3, 4, 5], dtype="int64")) == "integer"


def test_infers_float():
    assert _infer_col_type(pd.Series([1.5, 2.25, 3.75], dtype="float64")) == "float"


def test_infers_boolean():
    assert _infer_col_type(pd.Series([True, False, True], dtype="bool")) == "boolean"


def test_infers_datetime():
    series = pd.Series(["2024-01-01", "2024-02-15", "2024-03-30"])
    assert _infer_col_type(series) == "datetime"


def test_infers_categorical_for_low_cardinality_object_column():
    series = pd.Series(["Female", "Male", "Female", "Male", "Female"] * 10)
    assert _infer_col_type(series) == "categorical"


def test_infers_text_for_high_cardinality_object_column():
    series = pd.Series([f"unique free-text note number {i}" for i in range(100)])
    assert _infer_col_type(series) == "text"
