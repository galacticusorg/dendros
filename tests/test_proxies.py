"""Tests for the h5py-like group and dataset proxies."""

from __future__ import annotations

import numpy as np
import pytest

from dendros import open_outputs

from .conftest import _make_file


@pytest.fixture()
def array_file(tmp_path):
    """A single file holding a 1-D property and a 2-D (array-valued) property, as Galacticus writes spectra."""
    p = tmp_path / "arrays.hdf5"
    _make_file(
        p,
        outputs=[
            {
                "time": 13.8,
                "a": 1.0,
                "data": {
                    "basicMass": np.array([1e12, 2e12, 3e12]),
                    "stellarSED": np.arange(12.0).reshape(3, 4),
                },
            }
        ],
    )
    return str(p)


def test_dataset_ndim_size_len_1d(array_file):
    with open_outputs(array_file) as c:
        ds = c["Outputs/Output1/nodeData/basicMass"]
        assert ds.ndim == 1
        assert ds.size == 3
        assert len(ds) == 3


def test_dataset_ndim_size_len_2d(array_file):
    with open_outputs(array_file) as c:
        ds = c["Outputs/Output1/nodeData/stellarSED"]
        assert ds.shape == (3, 4)
        assert ds.ndim == 2
        assert ds.size == 12
        assert len(ds) == 3


def test_dataset_ndim_size_len_multifile(mpi_files):
    with open_outputs(mpi_files[0]) as c:
        ds = c["Outputs/Output1/nodeData/basicMass"]
        assert ds.ndim == len(ds.shape) == 1
        assert ds.size == ds.shape[0] == len(ds)


def test_dataset_compound_attribute_is_dict(single_file):
    with open_outputs(single_file) as c:
        units = c["Outputs/Output1/nodeData/basicMass"].attrs["units"]
        assert isinstance(units, dict)
        assert set(units) == {"unitsInSI", "description", "quantity", "isComoving"}
        assert isinstance(units["unitsInSI"], float)
        assert units["unitsInSI"] == pytest.approx(1.98892e30)
        assert units["description"] == "Solar masses"
        assert units["quantity"] == "solMass"
        assert units["isComoving"] == 0


def test_dataset_string_attribute_is_str(single_file):
    with open_outputs(single_file) as c:
        assert (
            c["Outputs/Output1/nodeData/basicMass"].attrs["comment"]
            == "Test dataset basicMass"
        )


def test_group_numeric_attribute_is_number(single_file):
    with open_outputs(single_file) as c:
        attrs = c["Outputs/Output1"].attrs
        assert attrs["outputTime"] == pytest.approx(13.8)
        assert isinstance(attrs["outputTime"], float)


def test_group_contains_and_iter(array_file):
    with open_outputs(array_file) as c:
        node = c["Outputs/Output1/nodeData"]
        assert "stellarSED" in node
        assert "missingProperty" not in node
        assert sorted(node) == sorted(node.keys())
