"""Manipulate star formation histories in Galacticus HDF5 collections."""
from __future__ import annotations

from ._collection import DatasetProxy

import re
import numpy as np


def sfh_collapse_metallicities(dataset: "DatasetProxy"):
    """Collapse a formation history over metallicity.

    Collapses (sums) star formation histories over the metallicity axis.

    The return type depends on how the history was tabulated:

    * If a shared ``time`` attribute is present (fixed tabulation times common
      to every galaxy), a fixed-length 2D :class:`numpy.ndarray` of shape
      ``(n_galaxies, n_times)`` is returned, and any empty entries are filled
      with zeros.
    * If the history was tabulated with the ``fixedAges`` method (typically
      used for lightcone outputs), each galaxy is tabulated at a fixed set of
      ages relative to its lightcone-crossing time.  Ages that precede the Big
      Bang are dropped, so galaxies crossing earlier retain fewer bins.  Since
      the ages form a fixed, nested set, the collapsed histories are
      *right-aligned* (the crossing-time bin is last) into a non-ragged 2D
      :class:`numpy.ndarray` of shape ``(n_galaxies, n_ages)``, front-padded
      with zeros where bins were dropped.  Column ``j`` corresponds to the same
      tabulation age across all galaxies; use :func:`sfh_times` to recover the
      per-galaxy times for each column.
    * Otherwise (variable per-galaxy times with no fixed-age structure), a list
      of 1D :class:`numpy.ndarray` objects is returned.

    Parameters
    ----------
    dataset:
        The dataset containing the star formation history data.

    """
    sfh = dataset.read()
    if 'time' in dataset.attrs:
        # The 'time' attribute exists - we can convert to a fixed length 2D array.
        times         = sfh_times(dataset)
        count_times   = len(times)
        sfh_collapsed = np.array(list(map(lambda x: sum(x) if len(x) > 0 else np.zeros(count_times), sfh)))
    elif _fixed_ages_parameters(dataset._collection) is not None:
        # The 'fixedAges' method was used - tabulation ages form a fixed, nested
        # set, so collapse over metallicity and right-align into a 2D array.
        collapsed     = [np.asarray(sum(x), dtype=float) if len(x) > 0 else np.zeros(0) for x in sfh]
        sfh_collapsed = _right_align(collapsed, fill=0.0)
    else:
        # The 'time' attribute does not exist - collapse over metallicity, but leave as a list of arrays.
        sfh_collapsed =          list(map(lambda x: sum(x) if len(x) > 0 else np.zeros(0          ), sfh))
    return sfh_collapsed


def sfh_times(dataset: "DatasetProxy"):
    """Return times associated with a star formation history.

    Returns `None` if no times are associated with this star formation history.

    The return type depends on how the history was tabulated:

    * If a shared ``time`` attribute is present, a 1D :class:`numpy.ndarray` of
      the tabulation times (common to every galaxy) is returned.
    * If the history was tabulated with the ``fixedAges`` method (typically
      used for lightcone outputs), the times differ from galaxy to galaxy and
      are stored in a companion ``...Times`` dataset.  These per-galaxy times
      are *right-aligned* (the crossing-time bin is last) into a non-ragged 2D
      :class:`numpy.ndarray` of shape ``(n_galaxies, n_ages)``, front-padded
      with ``NaN`` where bins were dropped.  This matches the alignment of the
      array returned by :func:`sfh_collapse_metallicities`, so column ``j`` of
      both arrays refer to the same tabulation bin.  Note that, because each
      galaxy crosses the lightcone at a different cosmic time, a given column
      holds a different absolute time for each galaxy (but the same lookback
      age relative to crossing).
    * Otherwise, ``None`` is returned.

    Parameters
    ----------
    dataset:
        The dataset containing the star formation history data.

    """
    if 'time' in dataset.attrs:
        # The 'time' attribute exists - a fixed-length set of times shared by
        # every galaxy.
        times = np.array(re.sub(r'[\[\],]','',dataset.attrs['time']).split(),dtype=float)
    elif _fixed_ages_parameters(dataset._collection) is not None:
        # The 'fixedAges' method was used - per-galaxy times live in a companion
        # '...Times' dataset; right-align them to match the collapsed masses.
        times_dataset = _companion_times_dataset(dataset)
        if times_dataset is None:
            times = None
        else:
            per_galaxy = [np.asarray(t, dtype=float) for t in times_dataset.read()]
            times      = _right_align(per_galaxy, fill=np.nan)
    else:
        times = None
    return times


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fixed_ages_parameters(collection):
    """Return the ``starFormationHistory`` parameter group if the ``fixedAges``
    method was used, otherwise ``None``.

    Galacticus records the run configuration in the ``Parameters`` group.  The
    ``fixedAges`` star formation history class is uniquely identified by its
    ``countAges`` and ``ageMinimum`` parameters, which no other class defines.
    """
    try:
        params = collection._primary['Parameters']
    except (KeyError, TypeError):
        return None
    group = params.get('starFormationHistory') if hasattr(params, 'get') else None
    if group is not None and 'countAges' in group.attrs and 'ageMinimum' in group.attrs:
        return group
    return None


def _companion_times_dataset(dataset: "DatasetProxy"):
    """Return a :class:`DatasetProxy` for the companion ``...Times`` dataset of a
    ``...Mass`` star formation history dataset, or ``None`` if absent."""
    if not dataset.name.endswith('Mass'):
        return None
    times_path = dataset.name[:-len('Mass')] + 'Times'
    try:
        dataset._collection._primary[times_path]
    except KeyError:
        return None
    return DatasetProxy(dataset._collection, times_path)


def _right_align(arrays, fill):
    """Stack variable-length 1D arrays into a 2D array, aligned to the right.

    Each input array is placed in the final columns of its row, with the
    leading columns filled with *fill*.  Returns an ``(n, width)`` float array,
    where ``width`` is the longest input length.
    """
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    width  = max((a.shape[0] for a in arrays), default=0)
    out    = np.full((len(arrays), width), fill, dtype=float)
    for i, a in enumerate(arrays):
        n = a.shape[0]
        if n:
            out[i, width - n:] = a
    return out
