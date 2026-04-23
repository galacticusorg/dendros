"""Tests for :func:`dendros.trace_galaxy_history` and ``Collection.trace_history``."""
from __future__ import annotations

import numpy as np
import pytest

from dendros import open_outputs, trace_galaxy_history


def test_id_present_in_all_outputs(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history([104], ["nodeData/basicMass"])
    assert hist["present"].all()
    np.testing.assert_allclose(
        hist["nodeData/basicMass"][0],
        [4.0, 44.0, 4444.0],
    )


def test_id_present_only_some_outputs(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history([101], ["nodeData/basicMass"])
    present = hist["present"][0]
    np.testing.assert_array_equal(present, [True, False, False])
    masses = hist["nodeData/basicMass"][0]
    assert masses[0] == 1.0
    assert np.isnan(masses[1])
    assert np.isnan(masses[2])


def test_id_never_found_warns(history_file):
    with open_outputs(history_file) as c:
        with pytest.warns(UserWarning, match="never found"):
            hist = c.trace_history([9999], ["nodeData/basicMass"])
    assert not hist["present"].any()
    assert np.isnan(hist["nodeData/basicMass"]).all()


def test_2d_property_shape_and_values(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history([104, 105], ["nodeData/spectrum"])
    arr = hist["nodeData/spectrum"]
    assert arr.shape == (2, 4, 3)
    # Galaxy 104: present at all three outputs.
    np.testing.assert_allclose(
        arr[0, :, 0], [104 * 10 + b for b in range(4)]
    )
    np.testing.assert_allclose(
        arr[0, :, 1], [104 * 10 + b for b in range(4)]
    )
    np.testing.assert_allclose(
        arr[0, :, 2], [104 * 10 + b for b in range(4)]
    )
    # Galaxy 105: absent at Output1 → NaN; present at Output2 and Output3.
    assert np.isnan(arr[1, :, 0]).all()
    np.testing.assert_allclose(
        arr[1, :, 1], [105 * 10 + b for b in range(4)]
    )


def test_integer_property_sentinel_default(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history([101, 104], ["nodeData/nodeIndex"])
    arr = hist["nodeData/nodeIndex"]
    assert np.issubdtype(arr.dtype, np.integer)
    # Galaxy 101 present only at Output1.
    assert arr[0, 0] == 10
    assert arr[0, 1] == -1
    assert arr[0, 2] == -1
    # Galaxy 104 present at all outputs.
    np.testing.assert_array_equal(arr[1], [40, 400, 4000])


def test_integer_property_sentinel_custom(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history(
            [101], ["nodeData/nodeIndex"], int_sentinel=-999
        )
    arr = hist["nodeData/nodeIndex"]
    assert arr[0, 0] == 10
    assert arr[0, 1] == -999
    assert arr[0, 2] == -999


def test_time_and_expansion_factor_arrays(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history(
            [101, 104, 105], ["nodeData/basicMass"]
        )
    expected_times = np.array([2.0, 6.0, 13.8])
    expected_a = np.array([0.2, 0.5, 1.0])
    present = hist["present"]
    # Where present, time/expansion_factor match the per-output attribute.
    for row in range(3):
        for o in range(3):
            if present[row, o]:
                assert hist["time"][row, o] == expected_times[o]
                assert hist["expansion_factor"][row, o] == expected_a[o]
            else:
                assert np.isnan(hist["time"][row, o])
                assert np.isnan(hist["expansion_factor"][row, o])


def test_present_mask_consistency(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history(
            [101, 104, 105],
            {"M": "nodeData/basicMass", "i": "nodeData/nodeIndex"},
        )
    present = hist["present"]
    # Float property: NaN exactly where not present.
    np.testing.assert_array_equal(np.isnan(hist["M"]), ~present)
    # Integer property: sentinel -1 exactly where not present.
    np.testing.assert_array_equal(hist["i"] == -1, ~present)


def test_multifile_per_file_search(history_mpi_files):
    path0, path1 = history_mpi_files
    with open_outputs([path0, path1]) as c:
        # IDs unique to each file resolve without collision.
        hist = c.trace_history([10, 13], ["nodeData/basicMass"])
    np.testing.assert_array_equal(hist["present"], [[True], [True]])
    np.testing.assert_allclose(
        hist["nodeData/basicMass"][:, 0], [100.0, 130.0]
    )


def test_multifile_duplicate_error(history_mpi_files):
    path0, path1 = history_mpi_files
    with open_outputs([path0, path1]) as c:
        with pytest.raises(ValueError, match="appear in both"):
            c.trace_history([12], ["nodeData/basicMass"])


def test_multifile_duplicate_warn(history_mpi_files):
    path0, path1 = history_mpi_files
    with open_outputs([path0, path1]) as c:
        with pytest.warns(UserWarning, match="appear in both"):
            hist = c.trace_history(
                [12],
                ["nodeData/basicMass"],
                on_duplicate_file_match="warn",
            )
    # First file's match is retained.
    assert hist["nodeData/basicMass"][0, 0] == 120.0


def test_multifile_duplicate_first_silent(history_mpi_files):
    import warnings as _warnings

    path0, path1 = history_mpi_files
    with open_outputs([path0, path1]) as c:
        with _warnings.catch_warnings(record=True) as record:
            _warnings.simplefilter("always")
            hist = c.trace_history(
                [12],
                ["nodeData/basicMass"],
                on_duplicate_file_match="first",
            )
    assert not any("appear in both" in str(w.message) for w in record)
    assert hist["nodeData/basicMass"][0, 0] == 120.0


def test_missing_id_dataset_raises(history_file_no_id):
    with open_outputs(history_file_no_id) as c:
        with pytest.raises(KeyError, match="nodeUniqueIDBranchTip"):
            c.trace_history([1], ["nodeData/basicMass"])


def test_missing_property_raises(history_file):
    with open_outputs(history_file) as c:
        with pytest.raises(KeyError, match="notAThing"):
            c.trace_history([104], ["nodeData/notAThing"])


def test_outputs_integer_range(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history(
            [104], ["nodeData/basicMass"], outputs=range(1, 3)
        )
    assert hist["output_names"].tolist() == ["Output1", "Output2"]
    assert hist["nodeData/basicMass"].shape == (1, 2)


def test_outputs_mixed_names_and_ints(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history(
            [104], ["nodeData/basicMass"], outputs=["Output1", 3]
        )
    assert hist["output_names"].tolist() == ["Output1", "Output3"]
    np.testing.assert_allclose(
        hist["nodeData/basicMass"][0], [4.0, 4444.0]
    )


def test_dict_label_input(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history(
            [104], {"Mhalo": "nodeData/basicMass"}
        )
    assert "Mhalo" in hist
    assert "nodeData/basicMass" not in hist


def test_reserved_label_raises(history_file):
    with open_outputs(history_file) as c:
        with pytest.raises(ValueError, match="[Rr]eserved"):
            c.trace_history([104], {"time": "nodeData/basicMass"})


def test_empty_ids(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history([], ["nodeData/basicMass"])
    assert hist["nodeData/basicMass"].shape == (0, 3)
    assert hist["present"].shape == (0, 3)
    assert hist["time"].shape == (0, 3)


def test_empty_outputs_raises(history_file):
    with open_outputs(history_file) as c:
        with pytest.raises(ValueError, match="empty"):
            c.trace_history([104], ["nodeData/basicMass"], outputs=[])


def test_varying_tail_shape_raises(history_file_varying_width):
    with open_outputs(history_file_varying_width) as c:
        with pytest.raises(ValueError, match="tail shape"):
            c.trace_history([1, 2], ["nodeData/spectrum"])


def test_convenience_method_matches_function(history_file):
    with open_outputs(history_file) as c:
        from_method = c.trace_history([104], ["nodeData/basicMass"])
        from_func = trace_galaxy_history(c, [104], ["nodeData/basicMass"])
    np.testing.assert_array_equal(
        from_method["nodeData/basicMass"], from_func["nodeData/basicMass"]
    )
    np.testing.assert_array_equal(from_method["present"], from_func["present"])


def test_history_output_order(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history([104], ["nodeData/basicMass"])
    # Output names are in temporal (early → late) order.
    assert hist["output_names"].tolist() == ["Output1", "Output2", "Output3"]
    # Times are monotonically increasing across the output axis.
    times = hist["time"][0]
    assert np.all(np.diff(times) > 0)


def test_ids_array_int32_and_list(history_file):
    with open_outputs(history_file) as c:
        hist_list = c.trace_history([104], ["nodeData/basicMass"])
        hist_arr = c.trace_history(
            np.array([104], dtype=np.int32), ["nodeData/basicMass"]
        )
    np.testing.assert_array_equal(
        hist_list["nodeData/basicMass"], hist_arr["nodeData/basicMass"]
    )
    assert hist_list["ids"].dtype == np.int64
    assert hist_arr["ids"].dtype == np.int64


def test_output_not_found_raises(history_file):
    with open_outputs(history_file) as c:
        with pytest.raises(KeyError, match="Output9"):
            c.trace_history([104], ["nodeData/basicMass"], outputs=[9])


def test_returned_keys(history_file):
    with open_outputs(history_file) as c:
        hist = c.trace_history([104], ["nodeData/basicMass"])
    for key in ("time", "expansion_factor", "redshift", "present", "output_names", "ids"):
        assert key in hist
    assert hist["ids"].dtype == np.int64
    assert hist["present"].dtype == np.bool_
