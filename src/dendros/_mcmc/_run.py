"""Top-level :class:`MCMCRun` container and :func:`open_mcmc` entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from ._chains import ChainSet, read_chains
from ._config import MCMCConfig, ModelParameter, parse_mcmc_config


class MCMCRun:
    """An MCMC run, parsed from its config file and lazily backed by chain data.

    Construct via :func:`open_mcmc`.  The chain files are not read until
    :attr:`chains` is first accessed; subsequent accesses return the cached
    :class:`ChainSet`.

    Parameters
    ----------
    config:
        Parsed :class:`MCMCConfig`.
    """

    def __init__(self, config: MCMCConfig) -> None:
        self._config = config
        self._chains: Optional[ChainSet] = None

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
    def chains(self) -> ChainSet:
        """Lazily-loaded :class:`ChainSet` for this run."""
        if self._chains is None:
            self._chains = read_chains(self._config)
        return self._chains

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


def open_mcmc(config_path: Union[str, "Path"]) -> MCMCRun:
    """Open an MCMC run by parsing its config XML.

    Parameters
    ----------
    config_path:
        Path to the Galacticus MCMC ``<parameters>`` XML file.

    Returns
    -------
    MCMCRun

    Examples
    --------
    >>> from dendros import open_mcmc
    >>> with open_mcmc("mcmcConfig.xml") as run:
    ...     print(run.parameters)
    ...     chains = run.chains
    """
    return MCMCRun(parse_mcmc_config(config_path))
