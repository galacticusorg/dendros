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
        If ``True``, writes ``statusCompletion = "complete"``; otherwise
        writes ``"incomplete"``.
    output_root:
        Name of the top-level group that holds the ``Output*`` groups.
    """
    if outputs is None:
        outputs = [
            {
                "time": 13.8,
                "a": 1.0,
                "data": {
                    "haloMass": np.array([1e12, 2e12, 3e12]),
                    "stellarMass": np.array([1e10, 2e10, 3e10]),
                },
            },
            {
                "time": 6.0,
                "a": 0.5,
                "data": {
                    "haloMass": np.array([5e11, 1e12]),
                    "stellarMass": np.array([5e9, 1e10]),
                },
            },
        ]

    with h5py.File(path, "w") as f:
        f.attrs["statusCompletion"] = "complete" if complete else "incomplete"
        root = f.create_group(output_root)
        for i, out in enumerate(outputs, 1):
            grp = root.create_group(f"Output{i}")
            grp.attrs["outputTime"] = out["time"]
            grp.attrs["outputExpansionFactor"] = out["a"]
            node = grp.create_group("nodeData")
            for name, arr in out["data"].items():
                ds = node.create_dataset(name, data=arr)
                ds.attrs["description"] = f"Test dataset {name}"
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
def mpi_files(tmp_path):
    """Two MPI-split files that together cover one output."""
    rank0_outputs = [
        {
            "time": 13.8,
            "a": 1.0,
            "data": {
                "haloMass": np.array([1e12, 2e12]),
                "stellarMass": np.array([1e10, 2e10]),
            },
        }
    ]
    rank1_outputs = [
        {
            "time": 13.8,
            "a": 1.0,
            "data": {
                "haloMass": np.array([3e12, 4e12]),
                "stellarMass": np.array([3e10, 4e10]),
            },
        }
    ]
    p0 = tmp_path / "galacticus_MPI:0000.hdf5"
    p1 = tmp_path / "galacticus_MPI:0001.hdf5"
    _make_file(p0, outputs=rank0_outputs)
    _make_file(p1, outputs=rank1_outputs)
    return str(p0), str(p1)
