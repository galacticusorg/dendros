"""MCMC support for Dendros: read Galacticus posterior-sample chain files and analyze them."""
from __future__ import annotations

from ._analysis import acceptance_rate, acceptance_rate_trace
from ._autocorr import (
    autocorrelation_function,
    autocorrelation_time,
    effective_sample_size,
)
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
    "acceptance_rate",
    "acceptance_rate_trace",
    "autocorrelation_function",
    "autocorrelation_time",
    "convergence_step",
    "effective_sample_size",
    "gelman_rubin",
    "geweke",
    "open_mcmc",
    "outlier_chains",
    "parse_mcmc_config",
    "read_chains",
]
