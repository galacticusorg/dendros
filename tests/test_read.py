"""Tests for dataset reading (Collection.read)."""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

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
    np.testing.assert_allclose(data["Mhalo"].value, [1e12, 2e12, 3e12])


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
        arr_all = data_all["nodeData/basicMass"].value
        mask = arr_all > 1.5e12
        data_sel = c.read("Output1", ["nodeData/basicMass"], where=mask)
    assert len(data_sel["nodeData/basicMass"]) == int(mask.sum())


def test_read_with_int_index_array(single_file):
    with open_outputs(single_file) as c:
        data = c.read("Output1", ["nodeData/basicMass"], where=[0, 2])
    np.testing.assert_allclose(data["nodeData/basicMass"].value, [1e12, 3e12])


def test_read_multifile_concat(mpi_files):
    """Arrays from MPI-split files should be concatenated along axis 0."""
    path0, path1 = mpi_files
    with open_outputs([path0, path1]) as c:
        data = c.read("Output1", ["nodeData/basicMass"])
    arr = data["nodeData/basicMass"]
    assert len(arr) == 4
    np.testing.assert_allclose(arr[:2].value, [1e12, 2e12])
    np.testing.assert_allclose(arr[2:].value, [3e12, 4e12])


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


# ---------------------------------------------------------------------------
# Units / astropy.units.Quantity behaviour
# ---------------------------------------------------------------------------


def test_read_returns_quantity_by_default(single_file):
    """Datasets with a 'quantity' string are returned as Quantity objects."""
    with open_outputs(single_file) as c:
        data = c.read("Output1", ["nodeData/basicMass"])
    arr = data["nodeData/basicMass"]
    assert isinstance(arr, u.Quantity)
    assert arr.unit == u.solMass
    np.testing.assert_allclose(arr.value, [1e12, 2e12, 3e12])


def test_read_quantity_converts_units(single_file):
    """The returned Quantity should be convertible to other mass units."""
    with open_outputs(single_file) as c:
        data = c.read("Output1", ["nodeData/basicMass"])
    in_kg = data["nodeData/basicMass"].to(u.kg)
    expected = np.array([1e12, 2e12, 3e12]) * u.solMass.to(u.kg)
    np.testing.assert_allclose(in_kg.value, expected)


def test_read_as_quantity_false_returns_plain_array(single_file):
    """as_quantity=False returns plain numpy arrays even for dimensioned data."""
    with open_outputs(single_file) as c:
        data = c.read("Output1", ["nodeData/basicMass"], as_quantity=False)
    arr = data["nodeData/basicMass"]
    assert isinstance(arr, np.ndarray)
    assert not isinstance(arr, u.Quantity)
    np.testing.assert_allclose(arr, [1e12, 2e12, 3e12])


def test_read_dimensionless_stays_numpy(single_file):
    """Datasets with an empty 'quantity' are left as plain numpy arrays."""
    with open_outputs(single_file) as c:
        data = c.read("Output1", ["nodeData/spin"])
    arr = data["nodeData/spin"]
    assert isinstance(arr, np.ndarray)
    assert not isinstance(arr, u.Quantity)
    np.testing.assert_allclose(arr, [0.1, 0.2, 0.3])


def test_read_legacy_unitsinsi_dimensionless(history_file):
    """Legacy scalar unitsInSI carries no quantity, so stays a numpy array."""
    with open_outputs(history_file) as c:
        data = c.read("Output1", ["nodeData/basicMass"])
    arr = data["nodeData/basicMass"]
    assert isinstance(arr, np.ndarray)
    assert not isinstance(arr, u.Quantity)


def test_read_multifile_quantity(mpi_files):
    """Quantity wrapping works across concatenated MPI files."""
    path0, path1 = mpi_files
    with open_outputs([path0, path1]) as c:
        data = c.read("Output1", ["nodeData/basicMass"])
    arr = data["nodeData/basicMass"]
    assert isinstance(arr, u.Quantity)
    assert arr.unit == u.solMass
    np.testing.assert_allclose(arr.value, [1e12, 2e12, 3e12, 4e12])
