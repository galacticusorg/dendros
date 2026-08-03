"""Read the proposed-state logs written by a Galacticus posterior-sample run.

A chain log records only the state *retained* at each step, so the parameters of a
rejected proposal appear nowhere in it.  Setting ``logProposals`` on a
differential-evolution simulation additionally writes
``<logFileRoot>Proposals_<rank>.log``, one row per evaluated proposal, whether or
not it was accepted.

This matters for anything that pairs recorded model outputs with the parameters
that produced them — see :mod:`dendros._mcmc._predictions`.  Without a proposal
log only accepted steps can be used, which for a typical acceptance rate discards
the large majority of the evaluations, and with them the wider coverage of
parameter space that rejected proposals provide.  That coverage is exactly what a
Jacobian or surrogate fit benefits from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._config import MCMCConfig

# Matches the per-rank proposal log filename: `<root>Proposals_<NNNN>.log`.
_RANK_SUFFIX = re.compile(r"Proposals_(\d{4})\.log$")

#: Filename infix distinguishing a proposal log from a chain log.
PROPOSAL_INFIX = "Proposals"


@dataclass
class ProposalSeries:
    """One rank's proposed states.

    Attributes
    ----------
    chain_index:
        MPI rank, parsed from the filename suffix.
    path:
        Source log-file path.
    step:
        ``(n_proposals,)`` simulation step at which each proposal was evaluated.
        Directly comparable with :attr:`dendros.Chain.step`.
    accepted:
        ``(n_proposals,)`` whether the proposal was accepted.
    log_posterior, log_likelihood:
        ``(n_proposals,)`` values *of the proposal*, not of the retained state.
    state:
        ``(n_proposals, n_params)`` proposed parameter vectors, in
        :attr:`MCMCConfig.parameters` order and in physical (unmapped) space.
    """

    chain_index: int
    path: Path
    step: np.ndarray
    accepted: np.ndarray
    log_posterior: np.ndarray
    log_likelihood: np.ndarray
    state: np.ndarray

    @property
    def n_proposals(self) -> int:
        return int(self.step.size)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of evaluated proposals that were accepted.

        Note this counts only proposals that reached the likelihood; those
        rejected on the prior are never evaluated and never logged, so this runs
        higher than the chain's overall step-acceptance rate.
        """
        return float(self.accepted.mean()) if self.accepted.size else float("nan")


class ProposalSet(Sequence[ProposalSeries]):
    """An ordered collection of :class:`ProposalSeries`, one per MPI rank."""

    def __init__(self, config: MCMCConfig, series: Sequence[ProposalSeries]) -> None:
        self._config = config
        self._series: Tuple[ProposalSeries, ...] = tuple(
            sorted(series, key=lambda s: s.chain_index)
        )

    def __len__(self) -> int:
        return len(self._series)

    def __iter__(self) -> Iterator[ProposalSeries]:
        return iter(self._series)

    def __getitem__(self, key):
        return self._series[key]

    def __repr__(self) -> str:
        return (
            f"<ProposalSet n_chains={len(self._series)} "
            f"n_proposals={self.n_proposals} "
            f"n_params={len(self._config.parameters)}>"
        )

    @property
    def config(self) -> MCMCConfig:
        return self._config

    @property
    def n_params(self) -> int:
        return len(self._config.parameters)

    @property
    def n_proposals(self) -> int:
        return sum(s.n_proposals for s in self._series)

    def by_chain(self) -> dict:
        """Return ``{chain_index: ProposalSeries}``."""
        return {s.chain_index: s for s in self._series}


def discover_proposal_files(log_file_root: Union[str, Path]) -> List[Path]:
    """Return all per-rank proposal logs matching ``<root>Proposals_NNNN.log``."""
    root = Path(log_file_root)
    parent = root.parent if str(root.parent) else Path(".")
    candidates = sorted(parent.glob(f"{root.name}{PROPOSAL_INFIX}_[0-9][0-9][0-9][0-9].log"))
    return [p for p in candidates if _RANK_SUFFIX.search(p.name)]


def read_proposals(
    config: MCMCConfig,
    *,
    log_file_root: Optional[Union[str, Path]] = None,
) -> ProposalSet:
    """Read every rank's proposal log for *config*.

    Parameters
    ----------
    config:
        Parsed :class:`MCMCConfig`.
    log_file_root:
        Override for ``config.log_file_root``, for a run analysed away from the
        machine it executed on.

    Returns
    -------
    ProposalSet

    Raises
    ------
    FileNotFoundError
        If no proposal logs are found.  A run without ``logProposals`` set writes
        none; in that case only accepted steps can be paired with predictions.
    """
    root = config.log_file_root if log_file_root is None else Path(log_file_root)
    files = discover_proposal_files(root)
    if not files:
        raise FileNotFoundError(
            f"No proposal logs found matching "
            f"'{root}{PROPOSAL_INFIX}_[0-9][0-9][0-9][0-9].log'. Set "
            f"[logProposals]=true on the simulation to write them."
        )
    return ProposalSet(config, [_read_proposal_file(p, config) for p in files])


def _read_proposal_file(path: Path, config: MCMCConfig) -> ProposalSeries:
    """Parse a single ``<root>Proposals_NNNN.log`` file."""
    m = _RANK_SUFFIX.search(path.name)
    chain_index = int(m.group(1)) if m else -1
    n_params = len(config.parameters)

    steps: List[int] = []
    accepted: List[bool] = []
    log_posterior: List[float] = []
    log_likelihood: List[float] = []
    states: List[np.ndarray] = []

    with open(path) as fh:
        for line in fh:
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if len(tokens) < 5 + n_params:
                raise ValueError(
                    f"Proposal log {path} line has {len(tokens)} columns; expected "
                    f"at least {5 + n_params} (= 5 + {n_params} parameters)."
                )
            steps.append(int(float(tokens[0])))
            accepted.append(tokens[2].strip() in ("T", ".true."))
            log_posterior.append(float(tokens[3]))
            log_likelihood.append(float(tokens[4]))
            states.append(np.array(tokens[5 : 5 + n_params], dtype=float))

    return ProposalSeries(
        chain_index=chain_index,
        path=path,
        step=np.array(steps, dtype=np.int64),
        accepted=np.array(accepted, dtype=bool),
        log_posterior=np.array(log_posterior, dtype=float),
        log_likelihood=np.array(log_likelihood, dtype=float),
        state=(
            np.vstack(states) if states else np.empty((0, n_params), dtype=float)
        ),
    )
