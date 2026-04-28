MCMC analysis
=============

Dendros reads Galacticus posterior-sample ("MCMC") chain logs given the
``<parameters>`` config XML used to drive the run, and provides convergence
diagnostics, post-burn analyses, parameter-file emission, and corner plots.

Opening a run
-------------

:func:`~dendros.open_mcmc` parses the config and returns an
:class:`~dendros.MCMCRun`.  Per-rank ``<root>_NNNN.log`` chain files are
discovered automatically from the ``<logFileRoot>`` entry and loaded lazily
on first access of ``run.chains``.

.. code-block:: python

   from dendros import open_mcmc

   with open_mcmc("mcmcConfig.xml") as run:
       print(run.parameters)            # active model parameters
       chains = run.chains              # ChainSet, one Chain per MPI rank

Both headerless chain files (the standard Galacticus output) and
``#``-prefixed headered chain files (as on the ``dmConstraintPipeline``
branch) are supported.  When a header is present, its parameter columns are
validated against the config.  ``simulation_kind="particleSwarm"``
configurations include trailing per-row velocity columns; these are split off
into ``Chain.velocity`` automatically.

Convergence
-----------

Brooks-Gelman corrected :math:`\hat R` and the non-parametric
:math:`R_{\mathrm{interval}}` are returned as functions of step:

.. code-block:: python

   result = run.gelman_rubin()
   step = run.convergence_step(threshold=1.1)

For chains started from an under-dispersed state (where Gelman-Rubin can
appear converged before mixing is achieved), Geweke z-scores are a useful
secondary diagnostic:

.. code-block:: python

   z = run.geweke()                      # (n_chains, n_params)

Outlier-chain detection iteratively applies a two-sided Grubbs test to each
chain's most recent state:

.. code-block:: python

   outliers = run.outlier_chains()       # tuple of chain_index values
   step = run.convergence_step(drop_chains=outliers)

Grubbs requires the inverse Student-t quantile from :mod:`scipy.stats`, which
ships with the optional ``mcmc`` extra alongside ``corner`` and
``matplotlib``::

    pip install 'dendros[mcmc]'

A clear :class:`ImportError` is raised if ``outlier_chains`` is called
without ``scipy`` installed.

All post-burn methods accept ``post_burn=None`` (the default), which runs
:meth:`~dendros.MCMCRun.gelman_rubin` and :meth:`~dendros.MCMCRun.convergence_step`
internally to pick a burn point.  Pass an explicit integer for full control.

Mixing diagnostics
------------------

.. code-block:: python

   tau = run.autocorrelation_time(post_burn=step)        # per parameter
   ess = run.effective_sample_size(post_burn=step)       # per parameter
   rate = run.acceptance_rate(post_burn=step)            # per chain
   trace = run.acceptance_rate_trace(window=30, post_burn=step)

Maximum posterior, sampling, PCA, and MVN fits
----------------------------------------------

.. code-block:: python

   res = run.maximum_posterior()
   print(res.state, res.log_posterior, res.chain_index, res.step)

   import numpy as np
   samples = run.posterior_samples(
       n=1000, post_burn=step, rng=np.random.default_rng(42),
   )

   pca = run.projection_pursuit(post_burn=step)
   print(pca.eigenvalues)            # ascending — smallest = best constrained
   print(pca.latex_summary(0))

   fit = run.multivariate_normal_fit(post_burn=step)
   fit.write_reparameterization_config("reparam.xml")

The reparameterization config declares ``metaParameter{i}`` as active
unit-normal parameters truncated to :math:`\pm n_\sigma` (default 5),
together with the original parameters as derived expressions of those
metas.  Re-running the MCMC against this config samples in coordinates where
the posterior is approximately spherical.

Emitting parameter files
------------------------

For likelihoods that derive from
``posteriorSampleLikelihoodBaseParameters``, a state vector can be written
into a Galacticus parameter file by reusing the leaf's
``<baseParametersFileName>``:

.. code-block:: python

   res = run.maximum_posterior()
   run.write_parameter_file(res.state, "max_post.xml")

For ``independentLikelihoods`` configs, each leaf has its own base file and
``<parameterMap>``; one file is written per leaf:

.. code-block:: python

   run.write_parameter_files(res.state, "out_dir")

Chain values are stored in physical (model) space — Galacticus applies the
inverse of ``operatorUnaryMapper`` before writing each row — so no mapper
inversion is performed at emission time.

Galacticus's parameter selectors are supported in active-parameter
``<name>`` paths:

* ``a/b`` and ``a::b`` — element navigation (both separators are accepted).
* ``a[2]`` — 1-based integer instance selector.
* ``a[@value='x']`` — element-with-matching-value-attribute selector.

Corner plots
------------

:meth:`~dendros.MCMCRun.corner_plot` is a thin wrapper around
:func:`corner.corner` that defaults to plotting every active parameter with
LaTeX labels derived from the config.

.. code-block:: python

   fig = run.corner_plot(post_burn=step)
   fig = run.corner_plot(parameters=["alpha", "beta"], post_burn=step)

The optional ``mcmc`` extra (``scipy``, ``corner``, ``matplotlib``) is
required for both :meth:`~dendros.MCMCRun.outlier_chains` and
:meth:`~dendros.MCMCRun.corner_plot`::

    pip install 'dendros[mcmc]'

End-to-end example
------------------

.. code-block:: python

   import numpy as np
   from dendros import open_mcmc

   with open_mcmc("mcmcConfig.xml") as run:
       outliers = run.outlier_chains()
       step = run.convergence_step(threshold=1.1, drop_chains=outliers)
       if step is None:
           raise RuntimeError("MCMC did not converge on the default grid.")

       ess = run.effective_sample_size(post_burn=step)
       print(f"ESS per parameter: {dict(zip(run.config.parameter_names, ess))}")

       fit = run.multivariate_normal_fit(
           post_burn=step, drop_chains=outliers,
       )
       fit.write_reparameterization_config("reparam.xml")

       map_ = run.maximum_posterior(drop_chains=outliers)
       run.write_parameter_files(map_.state, "max_posterior")

       samples = run.posterior_samples(
           n=200, post_burn=step, drop_chains=outliers,
           rng=np.random.default_rng(0),
       )
       for i, state in enumerate(samples.state):
           run.write_parameter_files(state, f"samples/{i:04d}")

       fig = run.corner_plot(post_burn=step, drop_chains=outliers)
       fig.savefig("corner.png")
