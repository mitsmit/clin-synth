"""Deterministic tests for clin_synth.generate.parse_csv_rows — pure CSV parsing/repair logic."""

from clin_synth.generate import parse_csv_rows

EXPECTED_COLS = [
    "race", "gender", "age", "time_in_hospital", "num_lab_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "A1Cresult", "metformin", "insulin", "readmitted",
]


def _row(*, race="Caucasian", id_prefix=None):
    """Build one well-formed data row (13 cols), optionally prefixed with a hex ID."""
    cells = [
        race, "Female", "[70-80)", "3", "45", "12", "0", "0",
        "0", "None", "No", "Steady", "NO",
    ]
    if id_prefix is not None:
        cells = [id_prefix] + cells
    return cells


def test_well_formed_rows_pass_through():
    raw = ",".join(_row()) + "\n" + ",".join(_row(race="AfricanAmerican"))
    valid, malformed = parse_csv_rows(raw, EXPECTED_COLS)
    assert len(valid) == 2
    assert malformed == []
    assert valid[0][0] == "Caucasian"
    assert valid[1][0] == "AfricanAmerican"


def test_strips_leading_hex_id_when_row_has_one_extra_column():
    # 14 cols, col 0 is a hex hash → strip it, race is recoverable in col 1
    raw = ",".join(_row(id_prefix="0123456789abcdef"))
    valid, malformed = parse_csv_rows(raw, EXPECTED_COLS)
    assert len(valid) == 1
    assert len(valid[0]) == len(EXPECTED_COLS)
    assert valid[0][0] == "Caucasian"
    assert malformed == []


def test_rejects_13_col_row_whose_first_cell_is_a_hex_id():
    # Exactly len(expected_cols) cols but col 0 is a hex hash — race is
    # irrecoverable, so the row must be rejected as malformed, not accepted.
    cells = ["0123456789abcdef"] + _row()[1:]
    raw = ",".join(cells)
    valid, malformed = parse_csv_rows(raw, EXPECTED_COLS)
    assert valid == []
    assert len(malformed) == 1


def test_pads_rows_missing_leading_three_columns():
    # len(expected_cols) - 3 cols → LLM dropped race/gender/age; pad with "".
    short_row = _row()[3:]
    raw = ",".join(short_row)
    valid, malformed = parse_csv_rows(raw, EXPECTED_COLS)
    assert len(valid) == 1
    assert valid[0][:3] == ["", "", ""]
    assert valid[0][3:] == short_row
    assert malformed == []


def test_skips_accidental_header_rows():
    header = ",".join(EXPECTED_COLS)
    raw = header + "\n" + ",".join(_row())
    valid, malformed = parse_csv_rows(raw, EXPECTED_COLS)
    assert len(valid) == 1
    assert malformed == []
