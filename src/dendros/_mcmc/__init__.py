"""MCMC support for Dendros: read Galacticus posterior-sample chain files and analyze them."""
from __future__ import annotations

from ._analysis import (
    MaxResult,
    PosteriorSamples,
    acceptance_rate,
    acceptance_rate_trace,
    maximum_likelihood,
    maximum_posterior,
    posterior_samples,
)
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
from ._mvn_reparam import MVNFit, multivariate_normal_fit
from ._projection import ProjectionPursuitResult, projection_pursuit
from ._run import MCMCRun, open_mcmc

__all__ = [
    "Chain",
    "ChainSet",
    "Likelihood",
    "MCMCConfig",
    "MCMCRun",
    "MVNFit",
    "MaxResult",
    "ModelParameter",
    "PerturberSpec",
    "PosteriorSamples",
    "PriorSpec",
    "ProjectionPursuitResult",
    "RhatResult",
    "acceptance_rate",
    "acceptance_rate_trace",
    "autocorrelation_function",
    "autocorrelation_time",
    "convergence_step",
    "effective_sample_size",
    "gelman_rubin",
    "geweke",
    "maximum_likelihood",
    "maximum_posterior",
    "multivariate_normal_fit",
    "open_mcmc",
    "outlier_chains",
    "parse_mcmc_config",
    "posterior_samples",
    "projection_pursuit",
    "read_chains",
]
