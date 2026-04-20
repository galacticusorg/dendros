"""Tests for dataset reading (Collection.read)."""
from __future__ import annotations

import numpy as np
import pytest

from dendros import open_outputs


def test_read_single_dataset(single_file):
    with open_outputs(single_file) as c:
        data = c.read("Output1", ["nodeData/basicMass"])
    assert "nodeData/basicMass" in data
    arr = data["nodeData/basicMass"]
    assert isinstance(arr, np.ndarray)
    assert len(arr) == 3


def test_read_returns_dict(single_file):
    with open_outputs(single_file) as c:
        data = c.read("Output1", ["nodeData/basicMass", "nodeData/diskMassStellar"])
    assert set(data.keys()) == {"nodeData/basicMass", "nodeData/diskMassStellar"}


def test_read_dict_datasets(single_file):
    """When datasets is a dict, keys should appear in the result."""
    with open_outputs(single_file) as c:
        data = c.read(
            "Output1",
            {"Mhalo": "nodeData/basicMass", "Mstar": "nodeData/diskMassStellar"},
        )
    assert set(data.keys()) == {"Mhalo", "Mstar"}
    np.testing.assert_allclose(data["Mhalo"], [1e12, 2e12, 3e12])


def test_read_by_integer_output(single_file):
    """Passing integer output index should work the same as a string."""
    with open_outputs(single_file) as c:
        data_str = c.read("Output1", ["nodeData/basicMass"])
        data_int = c.read(1, ["nodeData/basicMass"])
    np.testing.assert_array_equal(data_str["nodeData/basicMass"],
                                  data_int["nodeData/basicMass"])


def test_read_with_bool_mask(single_file):
    with open_outputs(single_file) as c:
        data_all = c.read("Output1", ["nodeData/basicMass"])
        arr_all = data_all["nodeData/basicMass"]
        mask = arr_all > 1.5e12
        data_sel = c.read("Output1", ["nodeData/basicMass"], where=mask)
    assert len(data_sel["nodeData/basicMass"]) == int(mask.sum())


def test_read_with_int_index_array(single_file):
    with open_outputs(single_file) as c:
        data = c.read("Output1", ["nodeData/basicMass"], where=[0, 2])
    np.testing.assert_allclose(data["nodeData/basicMass"], [1e12, 3e12])


def test_read_multifile_concat(mpi_files):
    """Arrays from MPI-split files should be concatenated along axis 0."""
    path0, path1 = mpi_files
    with open_outputs([path0, path1]) as c:
        data = c.read("Output1", ["nodeData/basicMass"])
    arr = data["nodeData/basicMass"]
    assert len(arr) == 4
    np.testing.assert_allclose(arr[:2], [1e12, 2e12])
    np.testing.assert_allclose(arr[2:], [3e12, 4e12])


def test_read_missing_dataset(single_file):
    with open_outputs(single_file) as c:
        with pytest.raises(KeyError):
            c.read("Output1", ["nodeData/nonExistent"])


def test_open_outputs_glob(mpi_files):
    """open_outputs with a single MPI file should auto-detect peers."""
    path0, _path1 = mpi_files
    with open_outputs(path0) as c:
        assert len(c.files) == 2
        data = c.read("Output1", ["nodeData/basicMass"])
    assert len(data["nodeData/basicMass"]) == 4


def test_open_outputs_file_not_found():
    with pytest.raises(FileNotFoundError):
        open_outputs("/nonexistent/path/to/file.hdf5")


def test_open_outputs_type_error():
    with pytest.raises(TypeError):
        open_outputs(12345)  # type: ignore[arg-type]
