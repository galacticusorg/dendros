"""Synthetic galaxy merger tree data shared by the dendros.viz tests (fixtures in conftest.py)."""

from __future__ import annotations

import h5py
import numpy as np

TIME_FINAL = 13.8

# Branches of the central galaxy: (id, start time, end time, final stellar mass, merges into).
CENTRAL = [
    (100, 0.5, TIME_FINAL, 5.0e10, None),
    (101, 1.0, 4.0, 1.0e10, 100),
    (102, 1.5, 3.0, 2.0e9, 101),
    (103, 2.0, 8.0, 5.0e9, 100),
    (104, 0.8, 2.5, 1.0e9, 100),
]
SATELLITE = [(200, 1.0, TIME_FINAL, 3.0e9, None)]


def _samples(start, end, mass):
    t = np.linspace(start, end, 40)
    m = mass * (t / end) ** 2
    sfr = 0.1 * m / t  # [M☉/Gyr]: sSFR = 0.1/t Gyr⁻¹
    return t, m, sfr


def _vlen(group, name, rows, dtype):
    ds = group.create_dataset(
        name, (len(rows),), dtype=h5py.vlen_dtype(np.dtype(dtype))
    )
    for k, r in enumerate(rows):
        ds[k] = np.asarray(r, dtype=dtype)


def write_tree_file(path):
    """Write a model file holding a galaxy merger tree (index 7): a central with branches CENTRAL and a surviving
    satellite with branch SATELLITE."""
    rows = {
        "Count": [],
        "NodeIndex": [],
        "Time": [],
        "MassStellarTotal": [],
        "TotalStarFormationRate": [],
        "MergeCentralIndex": [],
        "MergeSatelliteIndex": [],
        "MergeTime": [],
    }
    for branches in (CENTRAL, SATELLITE):
        count, ids, time, mass, sfr, cen, sat, tm = [], [], [], [], [], [], [], []
        for b, start, end, m_final, parent in branches:
            t, m, s = _samples(start, end, m_final)
            # A sample at the instant of each merger into this branch.
            for c, _, end_c, _, parent_c in branches:
                if parent_c == b:
                    t[np.argmin(np.abs(t - end_c))] = end_c
            count.append(t.size)
            ids.append(b)
            time.append(t)
            mass.append(m)
            sfr.append(s)
            if parent is not None:
                cen.append(parent)
                sat.append(b)
                tm.append(end)
        for key, value in (
            ("Count", count),
            ("NodeIndex", ids),
            ("Time", np.concatenate(time)),
            ("MassStellarTotal", np.concatenate(mass)),
            ("TotalStarFormationRate", np.concatenate(sfr)),
            ("MergeCentralIndex", cen),
            ("MergeSatelliteIndex", sat),
            ("MergeTime", tm),
        ):
            rows[key].append(value)
    with h5py.File(path, "w") as f:
        f.attrs["statusCompletion"] = 0
        out = f.create_group("Outputs/Output1")
        out.attrs["outputTime"] = TIME_FINAL
        out.attrs["outputExpansionFactor"] = 1.0
        out["mergerTreeIndex"] = np.array([7])
        out["mergerTreeStartIndex"] = np.array([0])
        out["mergerTreeCount"] = np.array([2])
        nodes = out.create_group("nodeData")
        nodes["nodeIndex"] = np.array([1, 5])
        nodes["nodeIsIsolated"] = np.array([1, 0])
        for key in ("Count", "NodeIndex", "MergeCentralIndex", "MergeSatelliteIndex"):
            _vlen(nodes, "galaxyMergerTree" + key, rows[key], np.int64)
        for key in ("Time", "MassStellarTotal", "TotalStarFormationRate", "MergeTime"):
            _vlen(nodes, "galaxyMergerTree" + key, rows[key], np.float64)
    return path
