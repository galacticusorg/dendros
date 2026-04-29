"""Shared pytest fixtures that build minimal Galacticus-like HDF5 files."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


def _make_file(
    path: Path,
    outputs=None,
    complete: bool = True,
    output_root: str = "Outputs",
):
    """Write a minimal Galacticus-like HDF5 file to *path*.

    Parameters
    ----------
    outputs:
        List of dicts with keys ``"time"``, ``"a"``, and ``"data"`` (a dict
        mapping dataset name → numpy array).  Defaults to two snapshot outputs.
    complete:
        If ``True``, writes ``statusCompletion = 0``; otherwise
        writes ``1``.
    output_root:
        Name of the top-level group that holds the ``Output*`` groups.
    """
    if outputs is None:
        outputs = [
            {
                "time": 13.8,
                "a": 1.0,
                "data": {
                    "basicMass": np.array([1e12, 2e12, 3e12]),
                    "diskMassStellar": np.array([1e10, 2e10, 3e10]),
                },
            },
            {
                "time": 6.0,
                "a": 0.5,
                "data": {
                    "basicMass": np.array([5e11, 1e12]),
                    "diskMassStellar": np.array([5e9, 1e10]),
                },
            },
        ]

    with h5py.File(path, "w") as f:
        f.attrs["statusCompletion"] = 0 if complete else 1
        root = f.create_group(output_root)
        for i, out in enumerate(outputs, 1):
            grp = root.create_group(f"Output{i}")
            grp.attrs["outputTime"] = out["time"]
            grp.attrs["outputExpansionFactor"] = out["a"]
            node = grp.create_group("nodeData")
            for name, arr in out["data"].items():
                ds = node.create_dataset(name, data=arr)
                ds.attrs["comment"] = f"Test dataset {name}"
                ds.attrs["unitsInSI"] = 1.989e30  # solar mass in kg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def single_file(tmp_path):
    """A single complete Galacticus HDF5 file with two outputs."""
    p = tmp_path / "galacticus.hdf5"
    _make_file(p)
    return str(p)


@pytest.fixture()
def incomplete_file(tmp_path):
    """A single file whose statusCompletion is not 'complete'."""
    p = tmp_path / "galacticus_incomplete.hdf5"
    _make_file(p, complete=False)
    return str(p)


@pytest.fixture()
def history_file(tmp_path):
    """A file with three outputs and a ``nodeUniqueIDBranchTip`` dataset.

    Galaxy IDs are arranged so that tracing exercises per-output presence:

    * ``101`` appears in Output1 only.
    * ``102`` appears in Output1 and Output2.
    * ``103`` appears in Output1 only.
    * ``104`` appears in all three outputs.
    * ``105`` appears in Output2 and Output3 (formed later).

    Row order differs between outputs so row-index matching is exercised.
    Each output carries a 2D ``spectrum`` dataset (4 wavelength bins) and an
    integer ``nodeIndex`` property so absent-slot sentinels can be tested.
    """
    p = tmp_path / "history.hdf5"
    with h5py.File(p, "w") as f:
        f.attrs["statusCompletion"] = 0
        root = f.create_group("Outputs")

        def _write_output(index, time, a, ids, masses, node_indices):
            grp = root.create_group(f"Output{index}")
            grp.attrs["outputTime"] = time
            grp.attrs["outputExpansionFactor"] = a
            nd = grp.create_group("nodeData")
            nd.create_dataset(
                "nodeUniqueIDBranchTip", data=np.asarray(ids, dtype=np.int64)
            )
            ds = nd.create_dataset("basicMass", data=np.asarray(masses, dtype=float))
            ds.attrs["comment"] = "halo mass"
            ds.attrs["unitsInSI"] = 1.989e30
            ds = nd.create_dataset(
                "nodeIndex", data=np.asarray(node_indices, dtype=np.int64)
            )
            ds.attrs["comment"] = "node index"
            # 2D spectrum: one row per galaxy, 4 wavelength bins encoding (id, band).
            spec = np.array(
                [[float(i) * 10 + b for b in range(4)] for i in ids], dtype=float
            )
            ds = nd.create_dataset("spectrum", data=spec)
            ds.attrs["comment"] = "mock spectrum"

        # Output1 (earliest): 101, 102, 103, 104
        _write_output(1, 2.0, 0.2,
                      ids=[103, 101, 104, 102],
                      masses=[3.0, 1.0, 4.0, 2.0],
                      node_indices=[30, 10, 40, 20])
        # Output2 (mid): 102, 104, 105
        _write_output(2, 6.0, 0.5,
                      ids=[104, 105, 102],
                      masses=[44.0, 55.0, 22.0],
                      node_indices=[400, 500, 200])
        # Output3 (latest): 104, 105
        _write_output(3, 13.8, 1.0,
                      ids=[105, 104],
                      masses=[5555.0, 4444.0],
                      node_indices=[5000, 4000])
    return str(p)


@pytest.fixture()
def history_file_no_id(tmp_path):
    """A two-output file that omits ``nodeUniqueIDBranchTip`` entirely."""
    p = tmp_path / "history_no_id.hdf5"
    with h5py.File(p, "w") as f:
        f.attrs["statusCompletion"] = 0
        root = f.create_group("Outputs")
        for i, (t, a) in enumerate(((2.0, 0.2), (13.8, 1.0)), start=1):
            grp = root.create_group(f"Output{i}")
            grp.attrs["outputTime"] = t
            grp.attrs["outputExpansionFactor"] = a
            nd = grp.create_group("nodeData")
            nd.create_dataset("basicMass", data=np.array([1.0, 2.0], dtype=float))
    return str(p)


@pytest.fixture()
def history_mpi_files(tmp_path):
    """Two MPI-split files sharing an output, each with its own IDs.

    File 0 has IDs ``[10, 11, 12]`` at Output1; file 1 has ``[12, 13, 14]``.
    ID ``12`` therefore appears in both files at Output1 to exercise the
    cross-file duplicate policy.  IDs ``10`` and ``13`` are unique to their
    respective files and exercise the per-file search.
    """
    p0 = tmp_path / "hist:MPI0000.hdf5"
    p1 = tmp_path / "hist:MPI0001.hdf5"

    def _write(path, ids, masses):
        with h5py.File(path, "w") as f:
            f.attrs["statusCompletion"] = 0
            root = f.create_group("Outputs")
            grp = root.create_group("Output1")
            grp.attrs["outputTime"] = 13.8
            grp.attrs["outputExpansionFactor"] = 1.0
            nd = grp.create_group("nodeData")
            nd.create_dataset(
                "nodeUniqueIDBranchTip", data=np.asarray(ids, dtype=np.int64)
            )
            nd.create_dataset("basicMass", data=np.asarray(masses, dtype=float))

    _write(p0, ids=[10, 11, 12], masses=[100.0, 110.0, 120.0])
    _write(p1, ids=[12, 13, 14], masses=[9999.0, 130.0, 140.0])
    return str(p0), str(p1)


@pytest.fixture()
def history_file_varying_width(tmp_path):
    """File whose 2D property has different widths in two outputs."""
    p = tmp_path / "history_varying.hdf5"
    with h5py.File(p, "w") as f:
        f.attrs["statusCompletion"] = 0
        root = f.create_group("Outputs")
        for i, (t, a, width) in enumerate(((2.0, 0.2, 3), (13.8, 1.0, 5)), start=1):
            grp = root.create_group(f"Output{i}")
            grp.attrs["outputTime"] = t
            grp.attrs["outputExpansionFactor"] = a
            nd = grp.create_group("nodeData")
            nd.create_dataset(
                "nodeUniqueIDBranchTip", data=np.array([1, 2], dtype=np.int64)
            )
            nd.create_dataset("spectrum", data=np.zeros((2, width), dtype=float))
    return str(p)


def _write_analysis_function1d(
    parent: h5py.Group,
    name: str,
    *,
    x: np.ndarray,
    y: np.ndarray,
    description: str = "",
    x_label: str = "x",
    y_label: str = "y",
    x_log: int = 0,
    y_log: int = 0,
    y_target: np.ndarray = None,
    target_label: str = "Target",
    y_err_lower: np.ndarray = None,
    y_err_upper: np.ndarray = None,
    y_covariance: np.ndarray = None,
    y_target_err_lower: np.ndarray = None,
    y_target_err_upper: np.ndarray = None,
    y_target_covariance: np.ndarray = None,
    type_value: str = "function1D",
) -> None:
    """Helper that writes one analysis subgroup with the standard Galacticus
    attribute → dataset-name redirection pattern."""
    grp = parent.create_group(name)
    grp.attrs["type"] = np.bytes_(type_value)
    grp.attrs["description"] = np.bytes_(description)
    grp.attrs["xAxisLabel"] = np.bytes_(x_label)
    grp.attrs["yAxisLabel"] = np.bytes_(y_label)
    grp.attrs["xAxisIsLog"] = np.int32(x_log)
    grp.attrs["yAxisIsLog"] = np.int32(y_log)
    grp.attrs["targetLabel"] = np.bytes_(target_label)

    grp.create_dataset("x", data=np.asarray(x, dtype=float))
    grp.attrs["xDataset"] = np.bytes_("x")
    grp.create_dataset("y", data=np.asarray(y, dtype=float))
    grp.attrs["yDataset"] = np.bytes_("y")

    if y_err_lower is not None and y_err_upper is not None:
        grp.create_dataset("yErrorLower", data=np.asarray(y_err_lower, dtype=float))
        grp.attrs["yErrorLower"] = np.bytes_("yErrorLower")
        grp.create_dataset("yErrorUpper", data=np.asarray(y_err_upper, dtype=float))
        grp.attrs["yErrorUpper"] = np.bytes_("yErrorUpper")
    if y_covariance is not None:
        grp.create_dataset("yCovariance", data=np.asarray(y_covariance, dtype=float))
        grp.attrs["yCovariance"] = np.bytes_("yCovariance")

    if y_target is not None:
        grp.create_dataset("yTarget", data=np.asarray(y_target, dtype=float))
        grp.attrs["yDatasetTarget"] = np.bytes_("yTarget")
        if y_target_err_lower is not None and y_target_err_upper is not None:
            grp.create_dataset(
                "yErrorLowerTarget", data=np.asarray(y_target_err_lower, dtype=float)
            )
            grp.attrs["yErrorLowerTarget"] = np.bytes_("yErrorLowerTarget")
            grp.create_dataset(
                "yErrorUpperTarget", data=np.asarray(y_target_err_upper, dtype=float)
            )
            grp.attrs["yErrorUpperTarget"] = np.bytes_("yErrorUpperTarget")
        if y_target_covariance is not None:
            grp.create_dataset(
                "yCovarianceTarget", data=np.asarray(y_target_covariance, dtype=float)
            )
            grp.attrs["yCovarianceTarget"] = np.bytes_("yCovarianceTarget")


@pytest.fixture()
def analyses_file(tmp_path):
    """Galacticus-like file containing a populated ``/analyses`` group."""
    p = tmp_path / "analyses.hdf5"
    with h5py.File(p, "w") as f:
        f.attrs["statusCompletion"] = 0
        # nodeData (for completeness; not required by analyses code)
        out = f.create_group("Outputs/Output1")
        out.attrs["outputTime"] = 13.8
        out.attrs["outputExpansionFactor"] = 1.0
        out.create_group("nodeData")

        analyses = f.create_group("analyses")

        # Plain function1D, log axes, no target.
        _write_analysis_function1d(
            analyses, "simpleSMF",
            x=[1e9, 1e10, 1e11],
            y=[1e-2, 1e-3, 1e-4],
            description=r"Stellar mass function $\hbox{at }z=0$",
            x_label=r"$M_\star/M_\odot$",
            y_label=r"$\phi\,/\,\hbox{Mpc}^{-3}\,\hbox{dex}^{-1}$",
            x_log=1, y_log=1,
        )

        # function1D with target overlay + asymmetric errors on both.
        _write_analysis_function1d(
            analyses, "withTarget",
            x=[1.0, 2.0, 3.0, 4.0],
            y=[10.0, 20.0, 30.0, 40.0],
            description=r"With target overlay $x \le 4$",
            x_label="x",
            y_label="y",
            x_log=0, y_log=0,
            y_err_lower=[9.0, 18.0, 27.0, 36.0],
            y_err_upper=[11.5, 22.0, 33.0, 44.0],
            y_target=[12.0, 19.0, 31.0, 39.0],
            target_label="Foo+24",
            y_target_err_lower=[10.0, 17.0, 28.0, 35.0],
            y_target_err_upper=[14.0, 21.0, 34.0, 43.0],
        )

        # function1D with covariance only (no asymmetric errors).
        cov = np.diag([0.1, 0.2, 0.3]) ** 2
        _write_analysis_function1d(
            analyses, "withCov",
            x=[1.0, 2.0, 3.0],
            y=[5.0, 6.0, 7.0],
            description="Cov errors",
            y_covariance=cov,
        )

        # Non-function1D — must be filtered out.
        _write_analysis_function1d(
            analyses, "notFunction1D",
            x=[1.0, 2.0],
            y=[1.0, 2.0],
            type_value="function2D",
        )

        # Nested under stepN:chainM.
        nested = analyses.create_group("step1:chain1")
        _write_analysis_function1d(
            nested, "inner",
            x=[1.0, 2.0],
            y=[10.0, 20.0],
            description="Nested analysis",
        )
    return str(p)


@pytest.fixture()
def mpi_files(tmp_path):
    """Two MPI-split files that together cover one output."""
    rank0_outputs = [
        {
            "time": 13.8,
            "a": 1.0,
            "data": {
                "basicMass": np.array([1e12, 2e12]),
                "diskMassStellar": np.array([1e10, 2e10]),
            },
        }
    ]
    rank1_outputs = [
        {
            "time": 13.8,
            "a": 1.0,
            "data": {
                "basicMass": np.array([3e12, 4e12]),
                "diskMassStellar": np.array([3e10, 4e10]),
            },
        }
    ]
    p0 = tmp_path / "galacticus:MPI0000.hdf5"
    p1 = tmp_path / "galacticus:MPI0001.hdf5"
    _make_file(p0, outputs=rank0_outputs)
    _make_file(p1, outputs=rank1_outputs)
    return str(p0), str(p1)


@pytest.fixture()
def analyses_mpi_files(tmp_path):
    """Two MPI-split files; only rank 0 carries the (already-reduced) /analyses."""
    p0 = tmp_path / "ana:MPI0000.hdf5"
    p1 = tmp_path / "ana:MPI0001.hdf5"
    _make_file(p0)
    _make_file(p1)
    with h5py.File(p0, "a") as f:
        analyses = f.create_group("analyses")
        _write_analysis_function1d(
            analyses, "simple",
            x=[1.0, 2.0, 3.0],
            y=[10.0, 20.0, 30.0],
            description="MPI test",
        )
    return str(p0), str(p1)
