"""MCMC support for Dendros: read Galacticus posterior-sample chain files and analyze them."""
from __future__ import annotations

from ._chains import Chain, ChainSet, read_chains
from ._config import (
    Likelihood,
    MCMCConfig,
    ModelParameter,
    PerturberSpec,
    PriorSpec,
    parse_mcmc_config,
)
from ._convergence import (
    RhatResult,
    convergence_step,
    gelman_rubin,
    geweke,
    outlier_chains,
)
from ._run import MCMCRun, open_mcmc

__all__ = [
    "Chain",
    "ChainSet",
    "Likelihood",
    "MCMCConfig",
    "MCMCRun",
    "ModelParameter",
    "PerturberSpec",
    "PriorSpec",
    "RhatResult",
    "convergence_step",
    "gelman_rubin",
    "geweke",
    "open_mcmc",
    "outlier_chains",
    "parse_mcmc_config",
    "read_chains",
]
