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

Model prediction vectors
------------------------

Some likelihood classes record the model vector they evaluated at each
likelihood call.  ``posteriorSampleLikelihoodHaloMassFunction`` does so when its
``pathSamples`` option is set, writing one file per constraint per MPI rank.
These make it possible to ask what the posterior *would* have been under a
different data weighting, without re-running anything::

    from dendros import index_prediction_files, read_predictions

    index = index_prediction_files("samples")        # scan the directory once
    preds = read_predictions("samples", label, files=index[label])
    paired = preds.paired(run.chains)

Three properties of these files are easy to get wrong, and
:meth:`~dendros.PredictionSet.paired` handles all three:

* **One record per likelihood evaluation, not per accepted state.**  Every
  proposal reaching the likelihood is recorded.  At a *rejected* step the chain
  log holds the retained state, not the proposed one, so on its own such a record
  cannot be attributed to any parameter vector and is dropped.  See
  :ref:`proposal-logs` to recover them.
* **Step indices are not contiguous and may be offset.**  Proposals rejected on
  the prior never reach the likelihood, so records must be *joined* on the step
  index rather than assumed to align positionally.  Runs predating the
  labelling fix in ``differential_evolution.F90`` label proposals one step
  behind the chain log; the convention is detected automatically, and
  :meth:`~dendros.PredictionSet.step_offset` warns rather than guessing in the
  rare case where records cover every step and the two are indistinguishable.
* **Posterior weighting.**  Accepted states persist for a variable number of
  steps, so use ``paired.multiplicity`` as sample weights for any
  posterior-averaged quantity.
* **A file's name identifies the evaluating process, not the chain.**  Under
  Galacticus' ``[loadBalance]=true`` (its default) any process may evaluate any
  chain's proposal.  Newer runs record the chain index in each row and
  :meth:`~dendros.PredictionSet.records_by_chain` attributes records by it, so
  this is handled transparently.  Files predating that column can only be
  attributed to the process that wrote them, which is valid only if load
  balancing was off; the symptom of it not having been is accepted steps of a
  chain having no record in that chain's own file.

Always pass ``files=`` from :func:`~dendros.index_prediction_files` when reading
many constraints: a production run's samples directory holds hundreds of
thousands of entries, and globbing per label re-walks all of them each time.

.. _proposal-logs:

Recovering the rejected proposals
---------------------------------

Dropping rejected-step records discards most of the evaluations — at a 15%
acceptance rate, roughly six in seven — and with them the wider coverage of
parameter space that rejected proposals explore.  Setting ``logProposals`` on a
differential-evolution simulation writes the proposed state at every step to
``<logFileRoot>Proposals_<rank>.log``, which makes all of them usable::

    from dendros import read_proposals

    proposals = read_proposals(run.config, log_file_root="chains")
    paired    = preds.paired(run.chains, proposals=proposals)

``paired.multiplicity`` is zero for rejected proposals: they are valid samples of
the model's response but carry no posterior weight.  So weight by multiplicity
only for posterior-averaged quantities, and pass ``weights=None`` when fitting a
Jacobian or surrogate — weighting there would throw away precisely the extra
coverage the proposals provide.  Because that coverage is centred on the
proposal distribution rather than the posterior, it is usually worth pinning the
expansion point explicitly::

    fit = jacobian_from_samples(
        paired.state, paired.prediction, degree=2,
        center=posterior_mean,        # derivative where it is wanted...
    )                                 # ...but fitted over the wider sample

Error propagation under a different data covariance
---------------------------------------------------

Given the paired predictions, the model can be linearized over the posterior
volume and parameter uncertainties re-derived under a data covariance the
original fit did not use::

    from dendros import (
        jacobian_from_samples, fisher_weights, parameter_covariances,
        rescale_correlation,
    )

    fit = jacobian_from_samples(
        paired.state, paired.prediction,
        weights=paired.multiplicity, active=mapped_parameters, degree=2,
    )
    weights = fisher_weights(mu, variance_fractional=f)   # negative binomial
    result  = parameter_covariances(
        fit.jacobian * scale[:, None], weights, covariance,
    )
    print(result.factors("sandwich"), result.volume_factor("sandwich"))

:class:`~dendros.InflationResult` carries three covariances: ``assumed``
reproduces the fit that was run, ``optimal`` is what a correctly-weighted fit
would have given, and ``sandwich`` is the true uncertainty of the mis-weighted
estimator that *was* used.

Several points decide whether the answer means anything:

* **Validate first.**  ``assumed`` should reproduce the chain's own covariance.
  If it does not, the linearization or the prior treatment is inadequate and
  nothing downstream is quantitative.
* **Prefer** ``degree=2``.  A model that curves over the posterior volume has
  its linear-fit slopes biased, because the fit trades curvature against the
  slopes of correlated parameters.  Where a parameter is known not to enter the
  model prediction — one entering only the variance, say — its fitted Jacobian
  column should vanish, which makes a free check on the fit's quality.
* **Inflation is not guaranteed.**  A strongly correlated mode the model cannot
  produce is effectively marginalized away, which can leave the remaining
  information *sharper*.  Compute it; do not assume the sign.
* **Regularize in correlation space.**  A rank-deficient covariance must be
  conditioned before inversion, but :func:`~dendros.nearest_positive_definite`
  floors eigenvalues relative to the largest.  Applied to a covariance whose
  variances span orders of magnitude that swamps the small-variance entries
  entirely, so convert to correlation form first
  (:func:`~dendros.to_correlation`), regularize, and restore the scale with
  :func:`~dendros.rescale_correlation`.
* **Bracket the correlation strength.**  :func:`~dendros.shrink_to_diagonal`
  scales off-diagonal terms by ``alpha``, leaving variances untouched.
  ``alpha=0`` must return an inflation of exactly one, which is a useful null
  check on the whole pipeline; if the answer is flat across a range of
  ``alpha``, a better covariance estimate is not worth obtaining.  Note
  ``alpha>1`` is no longer a valid correlation matrix, so only the ``sandwich``
  branch — which never inverts the covariance — stays interpretable there.

Fisher matrices are additive over independent constraints, so
:func:`~dendros.information_shares` ranks constraints by how much each tightens
the posterior, using only the diagonal weights already in hand.  Run it before
any covariance work: a constraint contributing negligibly cannot move the result
however its covariance is treated.

Finite-sample corrections for an estimated covariance are available as
:func:`~dendros.hartlap_factor` and
:func:`~dendros.dodelson_schneider_factor`.
