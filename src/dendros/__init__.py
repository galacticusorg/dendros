"""Dendros: a Python toolkit for analyzing Galacticus semi-analytic model outputs."""
from __future__ import annotations

from ._collection import Collection, open_outputs
from ._galaxy_history import trace_galaxy_history
from ._outputs import OutputIndex, OutputMeta
from ._star_formation import sfh_collapse_metallicities, sfh_times

__version__ = "0.2.0"

__all__ = [
    "Collection",
    "open_outputs",
    "OutputIndex",
    "OutputMeta",
    "sfh_collapse_metallicities",
    "sfh_times",
    "trace_galaxy_history",
]
