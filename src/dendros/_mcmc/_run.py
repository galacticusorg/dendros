"""Top-level :class:`MCMCRun` container and :func:`open_mcmc` entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from ._analysis import (
    MaxResult,
    PosteriorSamples,
    acceptance_rate,
    acceptance_rate_trace,
    maximum_likelihood,
    maximum_posterior,
    posterior_samples,
)
from ._autocorr import autocorrelation_time, effective_sample_size
from ._chains import ChainSet, read_chains
from ._config import MCMCConfig, ModelParameter, parse_mcmc_config
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
    write_parameter_file_to,
)
from ._plots import corner_plot
from ._projection import ProjectionPursuitResult, projection_pursuit


class MCMCRun:
    """An MCMC run, parsed from its config file and lazily backed by chain data.

    Construct via :func:`open_mcmc`.  The chain files are not read until
    :attr:`chains` is first accessed; subsequent accesses return the cached
    :class:`ChainSet`.

    Parameters
    ----------
    config:
        Parsed :class:`MCMCConfig`.
    max_steps:
        When given, retain only the last ``max_steps`` recorded steps of each
        chain when the chain data are first read.  Intended for run-time
        monitoring of a live chain: it bounds the per-row parse cost and speeds
        up every downstream diagnostic, at the cost of discarding older history.
        ``None`` (the default) reads the entire history.  Assigning to the
        :attr:`max_steps` property later invalidates any cached chains.
    """

    def __init__(self, config: MCMCConfig, *, max_steps: Optional[int] = None) -> None:
        self._config = config
        self._chains: Optional[ChainSet] = None
        self._max_steps: Optional[int] = None
        self.max_steps = max_steps  # validates via the setter

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> MCMCConfig:
        """The parsed :class:`MCMCConfig`."""
        return self._config

    @property
    def parameters(self) -> Tuple[ModelParameter, ...]:
        """Active model parameters, in chain-file column order."""
        return self._config.parameters

    @property
    def max_steps(self) -> Optional[int]:
        """Retain only the last this-many steps per chain when reading (``None`` = all).

        Assigning a new value invalidates the cached :attr:`chains` so the next
        access re-reads with the new window.
        """
        return self._max_steps

    @max_steps.setter
    def max_steps(self, value: Optional[int]) -> None:
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(
                f"max_steps must be a positive integer or None; got {value!r}"
            )
        if value != self._max_steps:
            self._chains = None  # invalidate cache; re-read on next access
        self._max_steps = value

    @property
    def chains(self) -> ChainSet:
        """Lazily-loaded :class:`ChainSet` for this run.

        Honors :attr:`max_steps`: when set, only the last ``max_steps`` steps of
        each chain are read and cached.
        """
        if self._chains is None:
            self._chains = read_chains(self._config, max_steps=self._max_steps)
        return self._chains

    # ------------------------------------------------------------------
    # Convergence diagnostics
    # ------------------------------------------------------------------

    def gelman_rubin(
        self,
        *,
        drop_chains: Sequence[int] = (),
        step_grid: Optional[Sequence[int]] = None,
        n_grid: int = 200,
        min_steps: int = 10,
        alpha_interval: float = 0.15,
    ) -> RhatResult:
        """Convenience wrapper around :func:`dendros.gelman_rubin`."""
        return gelman_rubin(
            self.chains,
            drop_chains=drop_chains,
            step_grid=step_grid,
            n_grid=n_grid,
            min_steps=min_steps,
            alpha_interval=alpha_interval,
        )

    def convergence_step(
        self,
        *,
        threshold: float = 1.1,
        sustained_for: int = 1,
        drop_chains: Sequence[int] = (),
        step_grid: Optional[Sequence[int]] = None,
        n_grid: int = 200,
        min_steps: int = 10,
    ) -> Optional[int]:
        """First simulation-step count at which max-Rhat is sustained below *threshold*.

        Computes a Gelman-Rubin trace via :meth:`gelman_rubin` and returns the
        ``RhatResult.steps`` value at which convergence is first declared.
        Returns ``None`` if convergence is never reached on the chosen grid.
        """
        result = self.gelman_rubin(
            drop_chains=drop_chains,
            step_grid=step_grid,
            n_grid=n_grid,
            min_steps=min_steps,
        )
        idx = convergence_step(
            result.Rhat_c_max(),
            threshold=threshold,
            sustained_for=sustained_for,
        )
        if idx is None:
            return None
        return int(result.steps[idx])

    def geweke(
        self,
        *,
        first: float = 0.1,
        last: float = 0.5,
    ) -> np.ndarray:
        """Convenience wrapper around :func:`dendros.geweke`."""
        return geweke(self.chains, first=first, last=last)

    def ensemble_drift(
        self,
        *,
        drop_chains: Sequence[int] = (),
        burn: int = 0,
        first: float = 0.5,
        last: float = 0.5,
    ) -> "EnsembleDriftResult":
        """Convenience wrapper around :func:`dendros.ensemble_drift`.

        The effect-size stationarity check appropriate for the ``differentialEvolution``
        ensemble: gate on ``result.max_drift() < threshold`` rather than on
        Gelman-Rubin / Geweke, which over-reject for large ensembles.
        """
        return ensemble_drift(
            self.chains,
            drop_chains=drop_chains,
            burn=burn,
            first=first,
            last=last,
        )

    def outlier_chains(
        self,
        *,
        alpha: float = 0.05,
        max_outliers: int = 10,
        parameters: Optional[Iterable[str]] = None,
    ) -> Tuple[int, ...]:
        """Convenience wrapper around :func:`dendros.outlier_chains`."""
        return outlier_chains(
            self.chains,
            alpha=alpha,
            max_outliers=max_outliers,
            parameters=parameters,
        )

    # ------------------------------------------------------------------
    # Mixing diagnostics
    # ------------------------------------------------------------------

    def acceptance_rate(
        self,
        *,
        post_burn: Optional[int] = None,
    ) -> np.ndarray:
        """Convenience wrapper around :func:`dendros.acceptance_rate`."""
        return acceptance_rate(self.chains, post_burn=post_burn)

    def acceptance_rate_trace(
        self,
        *,
        window: int = 30,
        post_burn: int = 0,
    ):
        """Convenience wrapper around :func:`dendros.acceptance_rate_trace`."""
        return acceptance_rate_trace(self.chains, window=window, post_burn=post_burn)

    def autocorrelation_time(
        self,
        *,
        post_burn: Optional[int] = None,
        c: float = 5.0,
    ) -> np.ndarray:
        """Convenience wrapper around :func:`dendros.autocorrelation_time`."""
        return autocorrelation_time(self.chains, post_burn=post_burn, c=c)

    def effective_sample_size(
        self,
        *,
        post_burn: Optional[int] = None,
        c: float = 5.0,
    ) -> np.ndarray:
        """Convenience wrapper around :func:`dendros.effective_sample_size`."""
        return effective_sample_size(self.chains, post_burn=post_burn, c=c)

    # ------------------------------------------------------------------
    # Max-posterior / sampling / projection / MVN fit
    # ------------------------------------------------------------------

    def maximum_posterior(
        self,
        *,
        drop_chains: Sequence[int] = (),
    ) -> MaxResult:
        """Convenience wrapper around :func:`dendros.maximum_posterior`."""
        return maximum_posterior(self.chains, drop_chains=drop_chains)

    def maximum_likelihood(
        self,
        *,
        drop_chains: Sequence[int] = (),
    ) -> MaxResult:
        """Convenience wrapper around :func:`dendros.maximum_likelihood`."""
        return maximum_likelihood(self.chains, drop_chains=drop_chains)

    def posterior_samples(
        self,
        n: int,
        *,
        post_burn: Optional[int] = None,
        drop_chains: Sequence[int] = (),
        rng: Optional[np.random.Generator] = None,
        replace: Optional[bool] = None,
    ) -> PosteriorSamples:
        """Convenience wrapper around :func:`dendros.posterior_samples`."""
        return posterior_samples(
            self.chains,
            n,
            post_burn=post_burn,
            drop_chains=drop_chains,
            rng=rng,
            replace=replace,
        )

    def projection_pursuit(
        self,
        *,
        post_burn: Optional[int] = None,
        drop_chains: Sequence[int] = (),
    ) -> ProjectionPursuitResult:
        """Convenience wrapper around :func:`dendros.projection_pursuit`."""
        return projection_pursuit(
            self.chains, post_burn=post_burn, drop_chains=drop_chains
        )

    def multivariate_normal_fit(
        self,
        *,
        post_burn: Optional[int] = None,
        drop_chains: Sequence[int] = (),
    ) -> MVNFit:
        """Convenience wrapper around :func:`dendros.multivariate_normal_fit`."""
        return multivariate_normal_fit(
            self.chains, post_burn=post_burn, drop_chains=drop_chains
        )

    # ------------------------------------------------------------------
    # Parameter-file emission
    # ------------------------------------------------------------------

    def write_parameter_file(
        self,
        state,
        out_path,
        *,
        likelihood_index: int = 0,
    ):
        """Emit a single Galacticus parameter file for one likelihood leaf.

        Reads ``leaves[likelihood_index].base_parameters_file``, applies the
        leaf's ``parameter_map`` (or the full state when no map is set), and
        writes the result to *out_path*.

        Parameters
        ----------
        state:
            ``(n_params,)`` state vector in physical (model) space.  Galacticus
            stores chain rows in physical space, so a row from
            :meth:`maximum_posterior` / :meth:`posterior_samples` can be passed
            directly.
        out_path:
            Output path.  Parent directory is created if missing.
        likelihood_index:
            Which leaf of the likelihood tree to use.  Defaults to ``0``.

        Returns
        -------
        pathlib.Path
            Resolved output path.
        """
        if self._config.likelihood is None:
            raise ValueError(
                "Config has no <posteriorSampleLikelihood>; cannot emit a "
                "parameter file."
            )
        leaves = self._config.likelihood.leaves()
        if likelihood_index < 0 or likelihood_index >= len(leaves):
            raise IndexError(
                f"likelihood_index={likelihood_index!r} out of range; "
                f"this run has {len(leaves)} leaf likelihoods."
            )
        leaf = leaves[likelihood_index]
        if leaf.base_parameters_file is None:
            raise ValueError(
                f"Likelihood leaf {likelihood_index!r} has no "
                "<baseParametersFileName>; cannot emit a parameter file."
            )

        state_arr = np.asarray(state, dtype=float)
        tree = read_parameter_file(leaf.base_parameters_file)
        apply_state(
            tree,
            self._config.parameters,
            state_arr,
            parameter_map=leaf.parameter_map,
        )
        return write_parameter_file_to(tree, out_path)

    def corner_plot(
        self,
        *,
        parameters: Optional[Iterable[str]] = None,
        post_burn: Optional[int] = None,
        drop_chains: Sequence[int] = (),
        labels: Optional[Sequence[str]] = None,
        **corner_kwargs,
    ):
        """Convenience wrapper around :func:`dendros.corner_plot`."""
        return corner_plot(
            self.chains,
            parameters=parameters,
            post_burn=post_burn,
            drop_chains=drop_chains,
            labels=labels,
            **corner_kwargs,
        )

    def write_parameter_files(
        self,
        state,
        out_dir,
        *,
        name_format: Optional[str] = None,
    ):
        """Emit one parameter file per likelihood leaf into *out_dir*.

        For ``independentLikelihoods`` configs each leaf has its own
        ``baseParametersFileName`` and ``parameterMap``; this writes one
        file per leaf, with each file's filename derived from the base file's
        stem.

        Parameters
        ----------
        state:
            ``(n_params,)`` state vector in physical space.
        out_dir:
            Output directory (created if missing).
        name_format:
            Output-filename format string accepting ``leaf_index`` and
            ``stem``.  Defaults to ``"{stem}.xml"`` for a single leaf and
            ``"{leaf_index:02d}_{stem}.xml"`` for multiple.

        Returns
        -------
        list of (leaf_index, pathlib.Path)
            One entry per leaf, in document order.
        """
        return emit_parameter_files(
            np.asarray(state, dtype=float),
            self._config,
            out_dir,
            name_format=name_format,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<MCMCRun config={self._config.config_path.name!r} "
            f"n_params={len(self._config.parameters)} "
            f"simulation_kind={self._config.simulation_kind!r}>"
        )

    def __enter__(self) -> "MCMCRun":
        return self

    def __exit__(self, *args) -> None:
        # Nothing to release: chain files are read fully into memory.
        pass


def open_mcmc(
    config_path: Union[str, "Path"], *, max_steps: Optional[int] = None
) -> MCMCRun:
    """Open an MCMC run by parsing its config XML.

    Parameters
    ----------
    config_path:
        Path to the Galacticus MCMC ``<parameters>`` XML file.
    max_steps:
        When given, retain only the last ``max_steps`` recorded steps of each
        chain.  Intended for run-time monitoring of a live chain, where only the
        recent window is of interest: it bounds the per-row parse cost and
        speeds up every downstream diagnostic (R-hat, ESS, autocorrelation).
        ``None`` (the default) reads the full history — use that for definitive,
        post-hoc analysis (corner plots, final convergence, posterior sampling)
        where truncation would bias the result.

    Returns
    -------
    MCMCRun

    Examples
    --------
    >>> from dendros import open_mcmc
    >>> with open_mcmc("mcmcConfig.xml") as run:
    ...     print(run.parameters)
    ...     chains = run.chains

    Fast run-time monitoring on the last 1000 steps::

    >>> with open_mcmc("mcmcConfig.xml", max_steps=1000) as run:
    ...     rhat = run.gelman_rubin()
    """
    return MCMCRun(parse_mcmc_config(config_path), max_steps=max_steps)
