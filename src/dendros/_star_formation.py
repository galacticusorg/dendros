"""Manipulate star formation histories in Galacticus HDF5 collections."""
from __future__ import annotations

from ._collection import DatasetProxy

import re
import numpy as np

def sfh_collapse_metallicities(dataset: "DatasetProxy"):
    """Collapse a formation history over metallicity.

    Collapses (sums) star formation histories over the metallicity axis. If fixed times were used a 2D :class:`numpy.ndarray` is
    returned, and any empty entries are filled with zeros. Otherwise, a list of 1D :class:`numpy.ndarray`s is returned.
  
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
    else:
        # The 'time' attribute does not exist - collapse over metallicity, but leave as a list of arrays.
        sfh_collapsed =          list(map(lambda x: sum(x) if len(x) > 0 else np.zeros(0          ), sfh))
    return sfh_collapsed

def sfh_times(dataset: "DatasetProxy"):
    """Return times associated with a star formation history.

    Returns `None` if no fixed times are associated with this star formation history.
    
    Parameters
    ----------
    dataset:
        The dataset containing the star formation history data.

    """
    if 'time' in dataset.attrs:
        # The 'time' attribute exists - we can convert to a fixed length 2D array.
        times = np.array(re.sub(r'[\[\],]','',dataset.attrs['time']).split(),dtype=float)
    else:
        times = None
    return times
