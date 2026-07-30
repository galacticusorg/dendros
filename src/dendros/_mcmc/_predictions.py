"""Read the model prediction vectors written alongside a Galacticus MCMC.

Some ``posteriorSampleLikelihood`` classes can record the model vector they
evaluated at each likelihood call.  ``haloMassFunction``, for example, writes one
file per constraint per MPI rank when its ``pathSamples`` option is set, each
line holding a simulation-step index followed by the model prediction over that
constraint's abscissa.

Two properties of these files matter and are easy to get wrong:

**They hold one record per likelihood evaluation, not per accepted state.**  Every
proposal that reaches the likelihood is recorded, whether or not it is
subsequently accepted.  At a *rejected* step the chain log stores the retained
(old) state, not the proposed one, so on its own such a record cannot be
attributed to any point in parameter space and :meth:`PredictionSet.paired` drops
it.  Runs with ``logProposals`` set also write the proposed states (see
:mod:`dendros._mcmc._proposals`); pass those to :meth:`PredictionSet.paired` and
every evaluated record becomes usable — several times as much data, covering a
wider region of parameter space than the posterior itself.

**Older runs' step indices run one behind the chain log.**  Galacticus labelled
proposals with the step counter as it stood *before* the accept/reject decision
advanced it, so a record labelled step *s* belonged to chain step *s* + 1.  This
was corrected in ``differential_evolution.F90`` (the proposal is now labelled
with the step it will be logged as), but files written before that fix retain the
old convention.  :meth:`PredictionSet.paired` detects which convention a run uses
rather than assuming; pass ``step_offset`` to override.

In either convention, records must be *joined* to chain rows on the step index.
Step indices are not contiguous — proposals rejected on the prior never reach the
likelihood — so records cannot be assumed to align with chain rows positionally.
A record labelled step ``0`` is the evaluation of the initial state, made before
stepping begins; the chain log has no row for it.

**A file's name identifies the process that evaluated the model, not the chain.**
Under Galacticus' ``[loadBalance]=true`` (its default) any process may evaluate
any chain's proposal and write the record.  Newer runs record the chain index
alongside the step, and :meth:`PredictionSet.records_by_chain` attributes records
by it, so load balancing is handled transparently.  Files written before that
column was added carry no chain index and can only be attributed wholesale to the
process that wrote them, which is valid only if load balancing was off — the
symptom of it not having been is accepted steps of a chain having no record in
that chain's own file.  The two layouts are distinguished by row width against
the abscissa in the header.
"""
from __future__ import annotations

import os
import warnings
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._chains import ChainSet
from ._proposals import ProposalSet

#: Step-index offsets :meth:`PredictionSet.paired` considers when detecting a
#: run's convention: ``0`` for runs written after the labelling fix, ``1`` for
#: those written before it.  See the module docstring.
SAMPLE_STEP_OFFSETS = (0, 1)

#: Offset used by runs predating the labelling fix in ``differential_evolution.F90``.
LEGACY_SAMPLE_STEP_OFFSET = 1

# Matches the per-rank sample filename suffix: `<label>_NNNN.txt`.
_RANK_SUFFIX = re.compile(r"_(\d{4})\.txt$")

# Matches the abscissa header line, e.g. "# Masses:  1.0e8  2.0e8 ...".
_ABSCISSA = re.compile(r"^\s*#\s*([A-Za-z][A-Za-z ]*?)\s*:\s*(.*)$")


# ---------------------------------------------------------------------------
# Per-rank series
# ---------------------------------------------------------------------------


@dataclass
class PredictionSeries:
    """One rank's model prediction records for a single constraint.

    Attributes
    ----------
    chain_index:
        MPI rank, parsed from the ``_NNNN.txt`` filename suffix.
    path:
        Source file path.
    step:
        ``(n_records,)`` integer step index exactly as recorded in the file.  It
        is *not* necessarily comparable with :attr:`dendros.Chain.step` — see the
        module docstring — so use :meth:`PredictionSet.paired` to align them.
    record_chain:
        ``(n_records,)`` chain index each record belongs to, when the file
        records one; ``None`` for files written before that column was added.
        Under load balancing this differs from :attr:`chain_index`, which is only
        the process that wrote the file.
    prediction:
        ``(n_records, n_bins)`` model prediction vectors.
    """

    chain_index: int
    path: Path
    step: np.ndarray
    prediction: np.ndarray
    record_chain: Optional[np.ndarray] = None

    @property
    def n_records(self) -> int:
        return int(self.step.size)

    @property
    def has_chain_index(self) -> bool:
        """Whether records carry their own chain index."""
        return self.record_chain is not None


# ---------------------------------------------------------------------------
# Paired result
# ---------------------------------------------------------------------------


@dataclass
class PairedPredictions:
    """Model predictions paired with the chain states that produced them.

    Contains only records at accepted steps, pooled across ranks; see the module
    docstring for why rejected-step records cannot be paired.

    Attributes
    ----------
    label:
        Constraint label (the sample filename with rank suffix removed).
    abscissa:
        ``(n_bins,)`` abscissa read from the file header, or ``None`` if absent.
    abscissa_name:
        Header name of the abscissa (e.g. ``"Masses"``), or ``None``.
    parameter_names:
        Model parameter names, in :attr:`state` column order.
    state:
        ``(n_pairs, n_params)`` parameter states.
    prediction:
        ``(n_pairs, n_bins)`` model predictions, row-matched to :attr:`state`.
    log_likelihood:
        ``(n_pairs,)`` total log likelihood logged for each state.  This is the
        sum over *all* constraints, not just this one.
    chain_index, step:
        ``(n_pairs,)`` provenance of each pair.
    multiplicity:
        ``(n_pairs,)`` number of consecutive chain steps for which each accepted
        state was retained — its posterior weight.  Use as sample weights when a
        posterior-averaged quantity is wanted; the raw rows over-represent
        frequently-rejected regions.
    """

    label: str
    abscissa: Optional[np.ndarray]
    abscissa_name: Optional[str]
    parameter_names: Tuple[str, ...]
    state: np.ndarray
    prediction: np.ndarray
    log_likelihood: np.ndarray
    chain_index: np.ndarray
    step: np.ndarray
    multiplicity: np.ndarray

    @property
    def n_pairs(self) -> int:
        return int(self.state.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.prediction.shape[1])

    def __repr__(self) -> str:
        return (
            f"<PairedPredictions {self.label!r} n_pairs={self.n_pairs} "
            f"n_bins={self.n_bins} n_params={len(self.parameter_names)}>"
        )


# ---------------------------------------------------------------------------
# Set of series for one constraint
# ---------------------------------------------------------------------------


class PredictionSet:
    """All ranks' prediction records for one constraint."""

    def __init__(
        self,
        label: str,
        series: Sequence[PredictionSeries],
        *,
        abscissa: Optional[np.ndarray] = None,
        abscissa_name: Optional[str] = None,
    ) -> None:
        self.label = label
        self._series: Tuple[PredictionSeries, ...] = tuple(
            sorted(series, key=lambda s: s.chain_index)
        )
        self.abscissa = abscissa
        self.abscissa_name = abscissa_name

    def __len__(self) -> int:
        return len(self._series)

    def __iter__(self) -> Iterator[PredictionSeries]:
        return iter(self._series)

    def __getitem__(self, key):
        return self._series[key]

    def __repr__(self) -> str:
        return (
            f"<PredictionSet {self.label!r} n_ranks={len(self._series)} "
            f"n_records={self.n_records}>"
        )

    @property
    def n_records(self) -> int:
        """Total records across ranks, including unpairable rejected-step ones."""
        return sum(s.n_records for s in self._series)

    def records_by_chain(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Return ``{chain_index: (step, prediction)}``, pooled across files.

        When records carry their own chain index they are attributed by it, so
        the result is correct even under load balancing, where a chain's model
        may have been evaluated and written by any process.  Otherwise each file
        is attributed wholesale to the process that wrote it, which is only valid
        if load balancing was off.
        """
        tagged = [s for s in self._series if s.record_chain is not None]
        if tagged and len(tagged) != len(self._series):
            raise ValueError(
                f"{self.label!r}: some prediction files carry a chain index and "
                f"others do not; they cannot have come from one run"
            )
        if not tagged:
            return {s.chain_index: (s.step, s.prediction) for s in self._series}

        parts: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for s in self._series:
            for c in np.unique(s.record_chain):
                mask = s.record_chain == c
                parts.setdefault(int(c), []).append((s.step[mask], s.prediction[mask]))
        out = {}
        for c, chunks in parts.items():
            step = np.concatenate([k for k, _ in chunks])
            prediction = np.concatenate([v for _, v in chunks], axis=0)
            order = np.argsort(step, kind="stable")
            out[c] = (step[order], prediction[order])
        return out

    def step_offset(self, chains: ChainSet) -> int:
        """Detect which step-labelling convention this run used.

        Returns whichever of :data:`SAMPLE_STEP_OFFSETS` accounts for more of the
        chains' accepted steps.  Every acceptance was necessarily evaluated and so
        must have a record, which normally makes this decisive: proposals rejected
        on the prior never reach the likelihood, so records cover only a fraction
        of steps and the wrong offset leaves a conspicuous shortfall.

        When records happen to cover *every* step the two conventions are
        indistinguishable from step indices alone — both account for all accepted
        steps, while pairing each state with a different record.  That case warns
        and returns ``0``; pass ``step_offset`` explicitly to
        :meth:`paired` if the run predates the labelling fix.
        """
        by_rank = {c.chain_index: c for c in chains}
        records = self.records_by_chain()
        hits_by_offset = {}
        for offset in SAMPLE_STEP_OFFSETS:
            hits = 0
            for index, (record_step, _) in records.items():
                chain = by_rank.get(index)
                if chain is None or chain.step.size < 2 or record_step.size == 0:
                    continue
                changed = np.zeros(chain.step.size, bool)
                changed[1:] = np.any(chain.state[1:] != chain.state[:-1], axis=1)
                accepted = chain.step[changed]
                shifted = np.sort(record_step + offset)
                idx = np.searchsorted(shifted, accepted)
                ok = (idx < shifted.size) & (
                    shifted[np.minimum(idx, shifted.size - 1)] == accepted
                )
                hits += int(ok.sum())
            hits_by_offset[offset] = hits

        best = max(hits_by_offset, key=lambda o: (hits_by_offset[o], -o))
        tied = [o for o, h in hits_by_offset.items() if h == hits_by_offset[best]]
        # With no accepted steps at all nothing will pair, so the offset is moot.
        if len(tied) > 1 and hits_by_offset[best] > 0:
            warnings.warn(
                "Cannot determine the sample step-labelling convention: offsets "
                f"{sorted(tied)} each account for {hits_by_offset[best]} accepted "
                "steps, because records cover every step. Assuming 0 (the "
                "convention since the differential_evolution.F90 labelling fix); "
                "pass step_offset explicitly for an older run.",
                stacklevel=2,
            )
            return 0
        return best

    def paired(
        self,
        chains: ChainSet,
        *,
        burn: int = 0,
        drop_chains: Sequence[int] = (),
        step_offset: Optional[int] = None,
        proposals: Optional["ProposalSet"] = None,
    ) -> PairedPredictions:
        """Join records to the parameter states that produced them.

        Without *proposals* only accepted steps can be paired, since a rejected
        proposal's parameters appear nowhere in the chain log.  Supplying a
        :class:`~dendros.ProposalSet` pairs *every* evaluated record instead,
        which for a typical acceptance rate is several times as much data and
        covers a wider region of parameter space.

        Parameters
        ----------
        chains:
            The :class:`~dendros.ChainSet` for the same run.  Matched to series
            by ``chain_index``.
        burn:
            Discard chain steps at or below the ``burn``-th recorded step of each
            chain before joining.
        drop_chains:
            ``chain_index`` values to exclude entirely.
        step_offset:
            Added to each record's step index to obtain the chain step.  Detected
            via :meth:`step_offset` when omitted, which is normally what you
            want: the convention changed between Galacticus versions.
        proposals:
            Proposed states from :func:`~dendros.read_proposals`, written when
            ``logProposals`` is set on the simulation.  When given, rejected-step
            records are paired too, and :attr:`PairedPredictions.multiplicity` is
            zero for them — they carry no posterior weight even though they are
            perfectly good samples of the model's response.  Pass
            ``weights=None`` to :func:`~dendros.jacobian_from_samples` to use
            them; weighting by multiplicity would discard exactly the extra
            coverage they provide.

        Returns
        -------
        PairedPredictions
        """
        drop = {int(i) for i in drop_chains}
        offset = self.step_offset(chains) if step_offset is None else int(step_offset)
        by_rank = {c.chain_index: c for c in chains}
        proposals_by_rank = {} if proposals is None else proposals.by_chain()

        states: List[np.ndarray] = []
        preds: List[np.ndarray] = []
        logls: List[np.ndarray] = []
        ranks: List[np.ndarray] = []
        steps: List[np.ndarray] = []
        mults: List[np.ndarray] = []

        for index, (record_step, record_prediction) in sorted(
            self.records_by_chain().items()
        ):
            if index in drop:
                continue
            chain = by_rank.get(index)
            if chain is None:
                continue

            step = chain.step
            state = chain.state
            if burn:
                if burn >= step.size:
                    continue
                step, state = step[burn:], state[burn:]
                logl = chain.log_likelihood[burn:]
            else:
                logl = chain.log_likelihood

            # Accepted steps are those at which the state changed.  The first
            # retained row cannot be classified, so it is excluded.
            changed = np.zeros(step.size, bool)
            changed[1:] = np.any(state[1:] != state[:-1], axis=1)
            acc_pos = np.flatnonzero(changed)
            if acc_pos.size == 0:
                continue
            acc_step = step[acc_pos]

            # Posterior weight: steps until the next acceptance (or the end).
            nxt = np.empty(acc_pos.size, dtype=np.int64)
            nxt[:-1] = acc_step[1:]
            nxt[-1] = step[-1] + 1
            acc_mult = nxt - acc_step

            order = np.argsort(record_step)
            sstep = record_step[order] + offset

            proposal = proposals_by_rank.get(index)
            if proposal is None:
                # Only accepted steps are attributable; join on those.
                target_step = acc_step
                target_state = state[acc_pos]
                target_mult = acc_mult
                target_logl = logl[acc_pos]
            else:
                # Every evaluated proposal has a known state.  Rejected ones get
                # zero posterior weight but are retained as model samples.
                keep = (proposal.step >= step[0]) & (proposal.step <= step[-1])
                target_step = proposal.step[keep]
                target_state = proposal.state[keep]
                target_logl = proposal.log_likelihood[keep]
                target_mult = np.zeros(target_step.size, dtype=np.int64)
                pos = np.searchsorted(acc_step, target_step)
                on_accepted = (pos < acc_step.size) & (
                    acc_step[np.minimum(pos, max(acc_step.size - 1, 0))] == target_step
                )
                target_mult[on_accepted] = acc_mult[pos[on_accepted]]

            if target_step.size == 0:
                continue
            hit = np.searchsorted(sstep, target_step)
            ok = (hit < sstep.size) & (sstep[np.minimum(hit, sstep.size - 1)] == target_step)
            if not ok.any():
                continue

            states.append(target_state[ok])
            preds.append(record_prediction[order][hit[ok]])
            logls.append(target_logl[ok])
            ranks.append(np.full(int(ok.sum()), index, dtype=np.int64))
            steps.append(target_step[ok])
            mults.append(target_mult[ok])

        n_params = chains.n_params
        n_bins = self.abscissa.size if self.abscissa is not None else 0
        if not states:
            return PairedPredictions(
                label=self.label,
                abscissa=self.abscissa,
                abscissa_name=self.abscissa_name,
                parameter_names=tuple(p.name for p in chains.config.parameters),
                state=np.empty((0, n_params)),
                prediction=np.empty((0, n_bins)),
                log_likelihood=np.empty(0),
                chain_index=np.empty(0, dtype=np.int64),
                step=np.empty(0, dtype=np.int64),
                multiplicity=np.empty(0, dtype=np.int64),
            )

        return PairedPredictions(
            label=self.label,
            abscissa=self.abscissa,
            abscissa_name=self.abscissa_name,
            parameter_names=tuple(p.name for p in chains.config.parameters),
            state=np.concatenate(states, axis=0),
            prediction=np.concatenate(preds, axis=0),
            log_likelihood=np.concatenate(logls),
            chain_index=np.concatenate(ranks),
            step=np.concatenate(steps),
            multiplicity=np.concatenate(mults),
        )


# ---------------------------------------------------------------------------
# Discovery + reading
# ---------------------------------------------------------------------------


def index_prediction_files(
    samples_dir: Union[str, Path]
) -> Dict[str, List[Tuple[int, Path]]]:
    """Scan *samples_dir* once, grouping sample files by constraint label.

    A production run writes one file per constraint per MPI rank, so this
    directory routinely holds hundreds of thousands of entries.  Scan it once
    and pass the result to :func:`read_predictions` via ``files``: globbing
    per-label instead re-walks every entry for each constraint, which dominates
    the run time and scales quadratically in the number of constraints.

    Returns
    -------
    dict
        ``{label: [(rank, path), ...]}``, each list sorted by rank.
    """
    d = Path(samples_dir)
    index: Dict[str, List[Tuple[int, Path]]] = {}
    with os.scandir(d) as it:
        for entry in it:
            if not entry.name.endswith(".txt"):
                continue
            m = _RANK_SUFFIX.search(entry.name)
            if not m:
                continue
            label = entry.name[: m.start()]
            index.setdefault(label, []).append((int(m.group(1)), Path(entry.path)))
    for v in index.values():
        v.sort()
    return index


def discover_prediction_labels(samples_dir: Union[str, Path]) -> List[str]:
    """Return the sorted constraint labels present in *samples_dir*.

    A label is a sample filename with its ``_NNNN.txt`` rank suffix removed.
    """
    return sorted(index_prediction_files(samples_dir))


def read_predictions(
    samples_dir: Union[str, Path],
    label: str,
    *,
    ranks: Optional[Sequence[int]] = None,
    files: Optional[Sequence[Tuple[int, Path]]] = None,
) -> PredictionSet:
    """Read every rank's prediction file for constraint *label*.

    Parameters
    ----------
    samples_dir:
        Directory holding the sample files.  Pass this explicitly rather than
        taking it from the config: a run analysed after being copied from the
        machine it ran on will have a stale ``pathSamples``.
    label:
        Constraint label, as returned by :func:`discover_prediction_labels`.
    ranks:
        When given, read only these MPI ranks.
    files:
        Pre-resolved ``[(rank, path), ...]`` for this label, as produced by
        :func:`index_prediction_files`.  Supplying it skips the directory scan —
        strongly preferred when reading many constraints from one directory.

    Returns
    -------
    PredictionSet

    Raises
    ------
    FileNotFoundError
        If no files match *label*.
    """
    d = Path(samples_dir)
    want = None if ranks is None else {int(r) for r in ranks}

    if files is None:
        candidates = []
        for p in sorted(d.glob(f"{label}_[0-9][0-9][0-9][0-9].txt")):
            m = _RANK_SUFFIX.search(p.name)
            if m:
                candidates.append((int(m.group(1)), p))
    else:
        candidates = sorted(files)

    series: List[PredictionSeries] = []
    abscissa: Optional[np.ndarray] = None
    abscissa_name: Optional[str] = None

    for rank, p in candidates:
        if want is not None and rank not in want:
            continue
        s, abs_, abs_name = _read_prediction_file(p, rank)
        if abscissa is None and abs_ is not None:
            abscissa, abscissa_name = abs_, abs_name
        series.append(s)

    if not series:
        raise FileNotFoundError(
            f"No prediction files found matching '{label}_[0-9][0-9][0-9][0-9].txt' "
            f"in {d}"
        )
    return PredictionSet(
        label, series, abscissa=abscissa, abscissa_name=abscissa_name
    )


def _read_prediction_file(
    path: Path, chain_index: int
) -> Tuple[PredictionSeries, Optional[np.ndarray], Optional[str]]:
    """Parse one ``<label>_NNNN.txt`` sample file.

    The whole numeric body is converted in a single call rather than row by row.
    A production run has hundreds of thousands of these files, and per-row Python
    parsing costs an order of magnitude more than reading the bytes does.
    """
    abscissa: Optional[np.ndarray] = None
    abscissa_name: Optional[str] = None

    text = path.read_text()
    data_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            # The abscissa header carries a name and values; the leading
            # provenance comment does not.
            if abscissa is None:
                m = _ABSCISSA.match(stripped)
                if m:
                    try:
                        values = np.array(m.group(2).split(), dtype=float)
                    except ValueError:
                        continue
                    if values.size:
                        abscissa_name = m.group(1).strip()
                        abscissa = values
            continue
        data_lines.append(stripped)

    if data_lines:
        n_col = len(data_lines[0].split())
        # `fromstring` with a separator parses straight from the text, avoiding the
        # intermediate list of token strings that dominates the cost otherwise.
        # (Only `fromstring`'s binary mode is deprecated, not this one.)
        flat = np.fromstring(" ".join(data_lines), sep=" ")
        if n_col > 1 and flat.size == n_col * len(data_lines):
            table = flat.reshape(len(data_lines), n_col)
        else:
            # Ragged rows (a run interrupted mid-write, say): fall back to
            # per-row parsing and keep only rows of the modal width.
            rows = [np.array(l.split(), dtype=float) for l in data_lines]
            n_col = max(sorted({r.size for r in rows}), key=lambda w: (
                sum(1 for r in rows if r.size == w), w))
            table = np.vstack([r for r in rows if r.size == n_col])
        step = table[:, 0].astype(np.int64)
        # Files written before the chain-index column was added hold
        # `step` + one value per abscissa point; newer ones interpose the chain
        # index.  The abscissa in the header settles which this is.
        n_bins = abscissa.size if abscissa is not None else None
        if n_bins is not None and table.shape[1] == n_bins + 2:
            record_chain = table[:, 1].astype(np.int64)
            prediction = table[:, 2:]
        elif n_bins is not None and table.shape[1] != n_bins + 1:
            raise ValueError(
                f"{path}: rows have {table.shape[1]} columns; expected "
                f"{n_bins + 1} (step + {n_bins} values) or {n_bins + 2} "
                f"(step + chain index + {n_bins} values)"
            )
        else:
            record_chain = None
            prediction = table[:, 1:]
    else:
        n_bins = abscissa.size if abscissa is not None else 0
        prediction = np.empty((0, n_bins))
        step = np.empty(0, dtype=np.int64)
        record_chain = None

    return (
        PredictionSeries(
            chain_index=chain_index,
            path=path,
            step=step,
            prediction=prediction,
            record_chain=record_chain,
        ),
        abscissa,
        abscissa_name,
    )
