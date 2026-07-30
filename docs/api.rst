API Reference
=============

.. autofunction:: dendros.open_outputs

.. autofunction:: dendros.open_models

.. autoclass:: dendros.Collection
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dendros.ModelCollection
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dendros.OutputIndex
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dendros.OutputMeta
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: dendros.sfh_collapse_metallicities

.. autofunction:: dendros.sfh_times

.. autofunction:: dendros.trace_galaxy_history

.. autofunction:: dendros.list_analyses

.. autofunction:: dendros.plot_analyses

MCMC
----

Entry point
~~~~~~~~~~~

.. autofunction:: dendros.open_mcmc

.. autoclass:: dendros.MCMCRun
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

.. autofunction:: dendros.parse_mcmc_config

.. autoclass:: dendros.MCMCConfig
   :members:

.. autoclass:: dendros.ModelParameter
   :members:

.. autoclass:: dendros.Likelihood
   :members:

.. autoclass:: dendros.PriorSpec
   :members:

.. autoclass:: dendros.PerturberSpec
   :members:

Chains
~~~~~~

.. autofunction:: dendros.read_chains

.. autoclass:: dendros.Chain
   :members:

.. autoclass:: dendros.ChainSet
   :members:

Convergence
~~~~~~~~~~~

.. autofunction:: dendros.gelman_rubin

.. autoclass:: dendros.RhatResult
   :members:

.. autofunction:: dendros.convergence_step

.. autofunction:: dendros.geweke

.. autofunction:: dendros.outlier_chains

Mixing diagnostics
~~~~~~~~~~~~~~~~~~

.. autofunction:: dendros.autocorrelation_function

.. autofunction:: dendros.autocorrelation_time

.. autofunction:: dendros.effective_sample_size

.. autofunction:: dendros.acceptance_rate

.. autofunction:: dendros.acceptance_rate_trace

Posterior analyses
~~~~~~~~~~~~~~~~~~

.. autofunction:: dendros.maximum_posterior

.. autofunction:: dendros.maximum_likelihood

.. autoclass:: dendros.MaxResult
   :members:

.. autofunction:: dendros.posterior_samples

.. autoclass:: dendros.PosteriorSamples
   :members:

.. autofunction:: dendros.projection_pursuit

.. autoclass:: dendros.ProjectionPursuitResult
   :members:

.. autofunction:: dendros.multivariate_normal_fit

.. autoclass:: dendros.MVNFit
   :members:

Parameter-file emission
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: dendros.read_parameter_file

.. autofunction:: dendros.resolve_parameter_path

.. autofunction:: dendros.apply_state

.. autofunction:: dendros.emit_parameter_files

.. autofunction:: dendros.write_parameter_file_to

Corner plots
~~~~~~~~~~~~

.. autofunction:: dendros.corner_plot

Model prediction vectors
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: dendros.index_prediction_files

.. autofunction:: dendros.discover_prediction_labels

.. autofunction:: dendros.read_predictions

.. autoclass:: dendros.PredictionSet
   :members:

.. automethod:: dendros.PredictionSet.records_by_chain

.. autoclass:: dendros.PredictionSeries
   :members:

.. autoclass:: dendros.PairedPredictions
   :members:

.. autodata:: dendros.SAMPLE_STEP_OFFSETS

.. autodata:: dendros.LEGACY_SAMPLE_STEP_OFFSET

Proposed states
~~~~~~~~~~~~~~~

.. autofunction:: dendros.read_proposals

.. autofunction:: dendros.discover_proposal_files

.. autoclass:: dendros.ProposalSet
   :members:

.. autoclass:: dendros.ProposalSeries
   :members:

Count-data likelihoods
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: dendros.log_likelihood_bins

.. autofunction:: dendros.count_variance

.. autofunction:: dendros.fisher_weights

.. autofunction:: dendros.negative_binomial_shape

.. autodata:: dendros.LOG_IMPROBABLE

Linearized error propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: dendros.jacobian_from_samples

.. autoclass:: dendros.JacobianFit
   :members:

.. autofunction:: dendros.parameter_covariances

.. autoclass:: dendros.InflationResult
   :members:

.. autofunction:: dendros.fisher_matrix

.. autofunction:: dendros.information_shares

Covariance utilities
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: dendros.to_correlation

.. autofunction:: dendros.rescale_correlation

.. autofunction:: dendros.shrink_to_diagonal

.. autofunction:: dendros.nearest_positive_definite

.. autofunction:: dendros.hartlap_factor

.. autofunction:: dendros.dodelson_schneider_factor

Internal helpers
----------------

.. autoclass:: dendros._collection.GroupProxy
   :members:
   :undoc-members:

.. autoclass:: dendros._collection.DatasetProxy
   :members:
   :undoc-members:
