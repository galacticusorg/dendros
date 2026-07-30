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
from ._counts import (
    LOG_IMPROBABLE,
    fisher_weights,
    log_likelihood_bins,
    negative_binomial_shape,
    count_variance,
)
from ._covariance import (
    dodelson_schneider_factor,
    hartlap_factor,
    nearest_positive_definite,
    rescale_correlation,
    shrink_to_diagonal,
    to_correlation,
)
from ._fisher import (
    InflationResult,
    JacobianFit,
    fisher_matrix,
    information_shares,
    jacobian_from_samples,
    parameter_covariances,
)
from ._config import (
    Likelihood,
    MCMCConfig,
    ModelParameter,
    PerturberSpec,
    PriorSpec,
    parse_mcmc_config,
)
from ._convergence import (
    EnsembleDriftResult,
    RhatResult,
    convergence_step,
    ensemble_drift,
    gelman_rubin,
    geweke,
    outlier_chains,
)
from ._mvn_reparam import MVNFit, multivariate_normal_fit
from ._params import (
    apply_state,
    emit_parameter_files,
    read_parameter_file,
    resolve_parameter_path,
    write_parameter_file_to,
)
from ._plots import corner_plot
from ._proposals import (
    PROPOSAL_INFIX,
    ProposalSeries,
    ProposalSet,
    discover_proposal_files,
    read_proposals,
)
from ._predictions import (
    LEGACY_SAMPLE_STEP_OFFSET,
    PairedPredictions,
    PredictionSeries,
    PredictionSet,
    SAMPLE_STEP_OFFSETS,
    discover_prediction_labels,
    index_prediction_files,
    read_predictions,
)
from ._projection import ProjectionPursuitResult, projection_pursuit
from ._run import MCMCRun, open_mcmc

__all__ = [
    "Chain",
    "ChainSet",
    "EnsembleDriftResult",
    "InflationResult",
    "JacobianFit",
    "LOG_IMPROBABLE",
    "Likelihood",
    "MCMCConfig",
    "MCMCRun",
    "MVNFit",
    "MaxResult",
    "ModelParameter",
    "PairedPredictions",
    "PerturberSpec",
    "PosteriorSamples",
    "PredictionSeries",
    "PredictionSet",
    "ProposalSeries",
    "ProposalSet",
    "PriorSpec",
    "ProjectionPursuitResult",
    "RhatResult",
    "SAMPLE_STEP_OFFSETS",
    "LEGACY_SAMPLE_STEP_OFFSET",
    "acceptance_rate",
    "acceptance_rate_trace",
    "apply_state",
    "autocorrelation_function",
    "autocorrelation_time",
    "convergence_step",
    "corner_plot",
    "discover_prediction_labels",
    "discover_proposal_files",
    "index_prediction_files",
    "dodelson_schneider_factor",
    "effective_sample_size",
    "emit_parameter_files",
    "ensemble_drift",
    "fisher_matrix",
    "fisher_weights",
    "gelman_rubin",
    "geweke",
    "hartlap_factor",
    "information_shares",
    "jacobian_from_samples",
    "log_likelihood_bins",
    "maximum_likelihood",
    "maximum_posterior",
    "multivariate_normal_fit",
    "nearest_positive_definite",
    "negative_binomial_shape",
    "open_mcmc",
    "outlier_chains",
    "parameter_covariances",
    "parse_mcmc_config",
    "posterior_samples",
    "projection_pursuit",
    "read_chains",
    "read_parameter_file",
    "read_predictions",
    "read_proposals",
    "rescale_correlation",
    "resolve_parameter_path",
    "shrink_to_diagonal",
    "to_correlation",
    "count_variance",
    "write_parameter_file_to",
]
