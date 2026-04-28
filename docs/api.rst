API Reference
=============

.. autofunction:: dendros.open_outputs

.. autoclass:: dendros.Collection
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

Internal helpers
----------------

.. autoclass:: dendros._collection.GroupProxy
   :members:
   :undoc-members:

.. autoclass:: dendros._collection.DatasetProxy
   :members:
   :undoc-members:
