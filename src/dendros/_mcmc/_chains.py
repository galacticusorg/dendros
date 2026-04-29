"""Read and represent Galacticus MCMC chain log files."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._config import MCMCConfig

# Matches the per-rank chain log filename: <root>_<NNNN>.log
_RANK_SUFFIX = re.compile(r"_(\d{4})\.log$")

# Matches a header line of the form: "# <int> = <description>".
_HEADER_COLUMN = re.compile(r"^\s*#\s*(\d+)\s*=\s*(.*?)\s*$")

# Matches a header parameter description, e.g. "Parameter `haloMassFunctionParameters/a`".
_HEADER_PARAM = re.compile(r"^Parameter\s+[`'\"](.+)[`'\"]\s*$")


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


@dataclass
class Chain:
    """One MPI rank's MCMC chain.

    Attributes
    ----------
    chain_index:
        MPI rank, parsed from the ``_NNNN.log`` filename suffix.
    path:
        Source log-file path.
    step:
        Integer simulation-step index, one per row.
    eval_time:
        Wall-clock evaluation time per step, in seconds.
    converged:
        Boolean flag indicating whether the simulation had declared
        convergence at this step.
    log_posterior:
        Log posterior probability per step.
    log_likelihood:
        Log likelihood per step.
    state:
        ``(n_steps, n_params)`` array of parameter values, in
        :attr:`MCMCConfig.parameters` order.  Values are in physical (model)
        space — Galacticus applies the inverse of ``operatorUnaryMapper``
        before writing.
    velocity:
        ``(n_steps, n_params)`` array of per-parameter particle velocities for
        ``particleSwarm`` simulations; ``None`` for differential-evolution and
        other state-only simulations.
    """

    chain_index: int
    path: Path
    step: np.ndarray
    eval_time: np.ndarray
    converged: np.ndarray
    log_posterior: np.ndarray
    log_likelihood: np.ndarray
    state: np.ndarray
    velocity: Optional[np.ndarray] = None

    @property
    def n_steps(self) -> int:
        return int(self.step.size)


# ---------------------------------------------------------------------------
# ChainSet
# ---------------------------------------------------------------------------


class ChainSet(Sequence[Chain]):
    """An ordered collection of :class:`Chain` objects from one MCMC run.

    Iteration yields chains in MPI-rank order.

    Parameters
    ----------
    config:
        The parsed :class:`MCMCConfig` the chains correspond to.
    chains:
        The per-rank chains.
    """

    def __init__(self, config: MCMCConfig, chains: Sequence[Chain]) -> None:
        self._config = config
        self._chains: Tuple[Chain, ...] = tuple(chains)

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chains)

    def __iter__(self) -> Iterator[Chain]:
        return iter(self._chains)

    def __getitem__(self, key):
        return self._chains[key]

    def __repr__(self) -> str:
        return (
            f"<ChainSet n_chains={len(self._chains)} "
            f"n_params={len(self._config.parameters)} "
            f"simulation_kind={self._config.simulation_kind!r}>"
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> MCMCConfig:
        return self._config

    @property
    def n_params(self) -> int:
        return len(self._config.parameters)

    # ------------------------------------------------------------------
    # Slicing helpers
    # ------------------------------------------------------------------

    def post_burn(self, burn: int) -> "ChainSet":
        """Return a new :class:`ChainSet` with the first *burn* steps dropped from each chain.

        Parameters
        ----------
        burn:
            Number of leading steps to discard.  Must be non-negative.
        """
        if burn < 0:
            raise ValueError(f"burn must be non-negative; got {burn!r}")
        new_chains = []
        for c in self._chains:
            new_chains.append(
                Chain(
                    chain_index=c.chain_index,
                    path=c.path,
                    step=c.step[burn:],
                    eval_time=c.eval_time[burn:],
                    converged=c.converged[burn:],
                    log_posterior=c.log_posterior[burn:],
                    log_likelihood=c.log_likelihood[burn:],
                    state=c.state[burn:],
                    velocity=None if c.velocity is None else c.velocity[burn:],
                )
            )
        return ChainSet(self._config, new_chains)

    def concatenated(
        self,
        *,
        burn: int = 0,
        drop_chains: Sequence[int] = (),
    ) -> np.ndarray:
        """Return a single ``(n_total_post_burn, n_params)`` state array.

        Concatenates the post-burn segments of every chain not listed in
        *drop_chains*, preserving chain order.

        Parameters
        ----------
        burn:
            Number of leading steps to discard from each chain.
        drop_chains:
            Iterable of ``chain_index`` values to exclude entirely.
        """
        drop = set(int(i) for i in drop_chains)
        parts = [
            c.state[burn:]
            for c in self._chains
            if c.chain_index not in drop
        ]
        if not parts:
            return np.empty((0, self.n_params), dtype=float)
        return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Discovery + reading
# ---------------------------------------------------------------------------


def discover_chain_files(log_file_root: Union[str, "Path"]) -> List[Path]:
    """Return all per-rank chain files matching ``<root>_NNNN.log``.

    Parameters
    ----------
    log_file_root:
        The chain log-file root (filename prefix), as recorded in the MCMC
        config.  May be a relative or absolute path.

    Returns
    -------
    list of pathlib.Path
        Sorted by MPI rank.  Empty when no files match.
    """
    root = Path(log_file_root)
    parent = root.parent if str(root.parent) else Path(".")
    stem = root.name
    pattern = f"{stem}_[0-9][0-9][0-9][0-9].log"
    candidates = sorted(parent.glob(pattern))
    # Filter to files whose suffix really is _NNNN.log (glob is permissive enough
    # that this is mostly belt-and-braces, but cheap).
    return [p for p in candidates if _RANK_SUFFIX.search(p.name)]


def read_chains(config: MCMCConfig) -> ChainSet:
    """Discover and read all per-rank chain files for *config*.

    Parameters
    ----------
    config:
        Parsed :class:`MCMCConfig`.

    Returns
    -------
    ChainSet

    Raises
    ------
    FileNotFoundError
        If no chain files are found at ``config.log_file_root``.
    """
    files = discover_chain_files(config.log_file_root)
    if not files:
        raise FileNotFoundError(
            f"No chain log files found matching "
            f"'{config.log_file_root}_[0-9][0-9][0-9][0-9].log'"
        )
    chains = [_read_chain_file(p, config) for p in files]
    return ChainSet(config, chains)


# ---------------------------------------------------------------------------
# Per-file reader
# ---------------------------------------------------------------------------


def _read_chain_file(path: Path, config: MCMCConfig) -> Chain:
    """Parse a single ``<root>_NNNN.log`` file."""
    rank = _rank_from_filename(path)
    n_params = len(config.parameters)
    n_state_cols = _state_column_count(config.simulation_kind, n_params)
    expected_total_cols = 6 + n_state_cols

    header_param_names: Optional[List[str]] = None
    rows: List[Tuple[int, int, float, bool, float, float, List[float]]] = []
    velocities: List[List[float]] = [] if config.simulation_kind == "particleSwarm" else []

    with path.open("r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Optional metadata header.  Capture column descriptions when
                # they appear, so we can validate parameter names against the
                # config when both are available.
                header_param_names = _accumulate_header_param(line, header_param_names)
                continue
            if line.startswith('"'):
                # Defensive: skip quoted lines that some tools produce.
                continue

            tokens = line.split()
            if len(tokens) < expected_total_cols:
                raise ValueError(
                    f"Chain file {path} line has {len(tokens)} columns; "
                    f"expected at least {expected_total_cols} (= 6 + "
                    f"{n_state_cols} state/velocity columns for "
                    f"simulation_kind={config.simulation_kind!r}, "
                    f"n_params={n_params})."
                )

            step = int(float(tokens[0]))
            chain_idx = int(float(tokens[1]))
            eval_t = float(tokens[2])
            conv = _parse_bool(tokens[3])
            logp = float(tokens[4])
            logl = float(tokens[5])
            state_vals = [float(t) for t in tokens[6 : 6 + n_params]]
            rows.append((step, chain_idx, eval_t, conv, logp, logl, state_vals))

            if config.simulation_kind == "particleSwarm":
                vel_vals = [
                    float(t) for t in tokens[6 + n_params : 6 + 2 * n_params]
                ]
                velocities.append(vel_vals)

    if header_param_names is not None:
        _validate_header_param_names(header_param_names, config, path)

    if not rows:
        # Honor the file's existence by returning an empty chain rather than
        # erroring; callers can detect zero-step chains via len(c.step).
        empty_state = np.empty((0, n_params), dtype=float)
        empty_vel = (
            np.empty((0, n_params), dtype=float)
            if config.simulation_kind == "particleSwarm"
            else None
        )
        return Chain(
            chain_index=rank,
            path=path,
            step=np.empty(0, dtype=np.int64),
            eval_time=np.empty(0, dtype=float),
            converged=np.empty(0, dtype=bool),
            log_posterior=np.empty(0, dtype=float),
            log_likelihood=np.empty(0, dtype=float),
            state=empty_state,
            velocity=empty_vel,
        )

    step_arr = np.array([r[0] for r in rows], dtype=np.int64)
    chain_idx_arr = np.array([r[1] for r in rows], dtype=np.int64)
    if not np.all(chain_idx_arr == chain_idx_arr[0]):
        raise ValueError(
            f"Chain file {path} contains multiple chain indices "
            f"{np.unique(chain_idx_arr).tolist()!r}; expected a single rank."
        )
    eval_arr = np.array([r[2] for r in rows], dtype=float)
    conv_arr = np.array([r[3] for r in rows], dtype=bool)
    logp_arr = np.array([r[4] for r in rows], dtype=float)
    logl_arr = np.array([r[5] for r in rows], dtype=float)
    state_arr = np.array([r[6] for r in rows], dtype=float)
    vel_arr = (
        np.array(velocities, dtype=float)
        if config.simulation_kind == "particleSwarm"
        else None
    )

    return Chain(
        chain_index=int(chain_idx_arr[0]) if rows else rank,
        path=path,
        step=step_arr,
        eval_time=eval_arr,
        converged=conv_arr,
        log_posterior=logp_arr,
        log_likelihood=logl_arr,
        state=state_arr,
        velocity=vel_arr,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_column_count(simulation_kind: str, n_params: int) -> int:
    """Return the number of trailing-column entries per row beyond the fixed 6."""
    if simulation_kind == "particleSwarm":
        return 2 * n_params
    return n_params


def _rank_from_filename(path: Path) -> int:
    m = _RANK_SUFFIX.search(path.name)
    if m is None:
        raise ValueError(
            f"Chain file {path} does not match the expected '_NNNN.log' suffix."
        )
    return int(m.group(1))


def _parse_bool(token: str) -> bool:
    """Parse a Fortran-style logical token (T/F, with optional surrounding chars)."""
    t = token.strip().lstrip(".").rstrip(".").upper()
    if t in ("T", "TRUE"):
        return True
    if t in ("F", "FALSE"):
        return False
    raise ValueError(f"Cannot parse boolean from token {token!r}")


def _accumulate_header_param(
    line: str, current: Optional[List[str]]
) -> Optional[List[str]]:
    """Update *current* with a parameter name if *line* is a parameter header line.

    Recognises lines like::

        #    7 = Parameter `haloMassFunctionParameters/a`

    and returns a list whose entry at column-index ``7`` (1-based) holds the
    parsed parameter name.  Non-parameter header lines leave *current*
    unchanged.  Returns *current* (possibly newly-allocated) so the caller
    can keep accumulating across the header block.
    """
    m = _HEADER_COLUMN.match(line)
    if m is None:
        return current
    col = int(m.group(1))
    desc = m.group(2)
    pm = _HEADER_PARAM.match(desc)
    if pm is None:
        return current
    name = pm.group(1)
    if current is None:
        current = []
    while len(current) < col:
        current.append("")
    current[col - 1] = name
    return current


def _validate_header_param_names(
    header_names: List[str], config: MCMCConfig, path: Path
) -> None:
    """Compare header-derived parameter names with the config's active parameters."""
    # Header columns are 1-based; parameters start at column 7, so index 6 onward.
    header_param_only = [n for n in header_names[6:] if n]
    if not header_param_only:
        return
    expected = list(config.parameter_names)
    if header_param_only != expected:
        raise ValueError(
            f"Chain file {path} header parameter columns "
            f"{header_param_only!r} do not match the config's active "
            f"parameters {expected!r}."
        )
