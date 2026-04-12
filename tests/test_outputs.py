"""Tests for output discovery (list_outputs / OutputIndex)."""
from __future__ import annotations

import pytest

from dendros import open_outputs


def test_list_outputs_returns_two_rows(single_file):
    with open_outputs(single_file) as c:
        tbl = c.list_outputs()
    assert len(tbl) == 2


def test_list_outputs_columns(single_file):
    with open_outputs(single_file) as c:
        tbl = c.list_outputs()
    for col in ("index", "name", "time", "scale_factor", "redshift"):
        assert col in tbl.colnames


def test_output1_redshift_zero(single_file):
    """Output1 has a=1.0, so z should be 0."""
    with open_outputs(single_file) as c:
        tbl = c.list_outputs()
    row = tbl[tbl["name"] == "Output1"][0]
    assert abs(row["redshift"]) < 1e-9
    assert abs(row["scale_factor"] - 1.0) < 1e-9
    assert abs(row["time"] - 13.8) < 1e-9


def test_output2_redshift_one(single_file):
    """Output2 has a=0.5, so z should be 1."""
    with open_outputs(single_file) as c:
        tbl = c.list_outputs()
    row = tbl[tbl["name"] == "Output2"][0]
    assert abs(row["redshift"] - 1.0) < 1e-9


def test_outputs_index_len(single_file):
    with open_outputs(single_file) as c:
        assert len(c.outputs) == 2


def test_outputs_index_getitem_by_name(single_file):
    with open_outputs(single_file) as c:
        meta = c.outputs["Output1"]
    assert meta.name == "Output1"
    assert abs(meta.redshift) < 1e-9


def test_outputs_index_getitem_by_position(single_file):
    with open_outputs(single_file) as c:
        meta = c.outputs[0]
    assert meta.name == "Output1"


def test_outputs_index_missing_key(single_file):
    with open_outputs(single_file) as c:
        with pytest.raises(KeyError):
            _ = c.outputs["Output99"]


def test_list_outputs_pandas(single_file):
    pd = pytest.importorskip("pandas")
    with open_outputs(single_file) as c:
        df = c.list_outputs(format="pandas")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_list_outputs_invalid_format(single_file):
    with open_outputs(single_file) as c:
        with pytest.raises(ValueError, match="format"):
            c.list_outputs(format="csv")
