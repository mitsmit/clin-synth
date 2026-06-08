"""Deterministic tests for clin_synth.utils.postprocess_utils._enforce_strata.

This is the guard that pins LLM-generated categorical values back to the
pre-assigned strata, so generated batches cannot drift from the seed's
categorical/target joint distribution.
"""

from clin_synth.utils.postprocess_utils import _enforce_strata

COL_INDICES = {"race": 0, "gender": 1, "readmitted": 12}


def test_enforce_strata_overwrites_drifted_values():
    rows = [
        ["WrongRace", "WrongGender", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "WrongTarget"],
    ]
    batch_categoricals = [{"race": "Caucasian", "gender": "Female", "readmitted": "NO"}]

    out = _enforce_strata(rows, batch_categoricals, COL_INDICES)

    assert out[0][COL_INDICES["race"]] == "Caucasian"
    assert out[0][COL_INDICES["gender"]] == "Female"
    assert out[0][COL_INDICES["readmitted"]] == "NO"


def test_enforce_strata_leaves_non_strata_columns_untouched():
    rows = [["Caucasian", "Female"] + ["unchanged"] * 10 + ["NO"]]
    batch_categoricals = [{"race": "Caucasian", "gender": "Female", "readmitted": "NO"}]

    out = _enforce_strata(rows, batch_categoricals, COL_INDICES)

    assert out[0][2] == "unchanged"


def test_enforce_strata_only_overwrites_positionally_matched_rows():
    # Two parsed rows but only one strata assignment — the second row
    # (no positional match) must be left alone, not raise.
    rows = [
        ["Caucasian", "Female"] + ["x"] * 11,
        ["AfricanAmerican", "Male"] + ["x"] * 11,
    ]
    batch_categoricals = [{"race": "Caucasian", "gender": "Female"}]

    out = _enforce_strata(rows, batch_categoricals, COL_INDICES)

    assert out[0][0] == "Caucasian"
    assert out[1][0] == "AfricanAmerican"   # untouched — no strata for index 1


def test_enforce_strata_ignores_keys_absent_from_col_indices():
    # Strata dicts may carry keys (e.g. domain-specific cols) that don't
    # exist in this batch's column layout — must be skipped, not KeyError.
    rows = [["Caucasian"] + ["x"] * 12]
    batch_categoricals = [{"race": "Caucasian", "unknown_col": "ignored"}]

    out = _enforce_strata(rows, batch_categoricals, COL_INDICES)

    assert out[0][0] == "Caucasian"
