"""Dendros: a Python toolkit for analyzing Galacticus semi-analytic model outputs."""
from __future__ import annotations

from ._collection import Collection, open_outputs
from ._galaxy_history import trace_galaxy_history
from ._mcmc import (
    Chain,
    ChainSet,
    Likelihood,
    MCMCConfig,
    MCMCRun,
    ModelParameter,
    PerturberSpec,
    PriorSpec,
    RhatResult,
    convergence_step,
    gelman_rubin,
    geweke,
    open_mcmc,
    outlier_chains,
    parse_mcmc_config,
    read_chains,
)
from ._outputs import OutputIndex, OutputMeta
from ._star_formation import sfh_collapse_metallicities, sfh_times

__version__ = "0.2.0"

__all__ = [
    "Chain",
    "ChainSet",
    "Collection",
    "Likelihood",
    "MCMCConfig",
    "MCMCRun",
    "ModelParameter",
    "OutputIndex",
    "OutputMeta",
    "PerturberSpec",
    "PriorSpec",
    "RhatResult",
    "convergence_step",
    "gelman_rubin",
    "geweke",
    "open_mcmc",
    "open_outputs",
    "outlier_chains",
    "parse_mcmc_config",
    "read_chains",
    "sfh_collapse_metallicities",
    "sfh_times",
    "trace_galaxy_history",
]
