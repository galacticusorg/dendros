"""Tests for properties listing (list_properties)."""
from __future__ import annotations

import pytest

from dendros import open_outputs


def test_list_properties_columns(single_file):
    with open_outputs(single_file) as c:
        tbl = c.list_properties("Output1")
    for col in ("name", "dtype", "shape", "description", "unitsInSI"):
        assert col in tbl.colnames


def test_list_properties_contains_halo_mass(single_file):
    with open_outputs(single_file) as c:
        tbl = c.list_properties("Output1")
    names = list(tbl["name"])
    assert "haloMass" in names


def test_list_properties_by_integer_index(single_file):
    """Passing an integer should resolve to Output<n>."""
    with open_outputs(single_file) as c:
        tbl_str = c.list_properties("Output1")
        tbl_int = c.list_properties(1)
    assert list(tbl_str["name"]) == list(tbl_int["name"])


def test_list_properties_description(single_file):
    """Each dataset should carry a non-empty description."""
    with open_outputs(single_file) as c:
        tbl = c.list_properties("Output1")
    for row in tbl:
        assert row["description"] != ""


def test_list_properties_units_numeric(single_file):
    """unitsInSI should be a numeric value for datasets that have the attr."""
    with open_outputs(single_file) as c:
        tbl = c.list_properties("Output1")
    for row in tbl:
        assert row["unitsInSI"] is not None
        assert float(row["unitsInSI"]) > 0


def test_list_properties_pandas(single_file):
    pd = pytest.importorskip("pandas")
    with open_outputs(single_file) as c:
        df = c.list_properties("Output1", format="pandas")
    assert isinstance(df, pd.DataFrame)


def test_list_properties_missing_output(single_file):
    with open_outputs(single_file) as c:
        with pytest.raises(KeyError):
            c.list_properties("Output99")
