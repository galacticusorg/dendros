"""Read and plot Galacticus ``/analyses`` group results.

Galacticus optionally writes a top-level ``/analyses`` group to its HDF5
output containing reduced analysis results — one subgroup per analysis.
This module discovers ``function1D`` analyses, reads their data, and
produces matplotlib plots showing the model curve plus an optional
target/observational overlay.

For MPI multi-file collections the ``/analyses`` data has been reduced
across all ranks and is identical in every file, so only the primary
file is read.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from ._collection import Collection, _decode, _default_model_label, _make_table

if TYPE_CHECKING:
    from matplotlib.figure import Figure


_ANALYSIS_TYPE = "function1D"
_ANALYSES_GROUP = "analyses"

# Style ---------------------------------------------------------------------

_MODEL_COLOR = "#1f4e79"   # deep blue, used for single-model plots
_TARGET_COLOR = "#d1495b"  # brick red, used for the target/observational overlay
_MULTI_MODEL_CMAP = "tab10"

_RC: Dict[str, object] = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "font.size": 11,
    "legend.fontsize": 10,
    "mathtext.fontset": "cm",
    "figure.autolayout": False,
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover(group: "h5py.Group", prefix: str = "") -> Iterator[Tuple[str, "h5py.Group"]]:
    """Yield ``(name_path, group)`` pairs for every ``function1D`` analysis.

    Walks recursively so an optional ``stepN:chainM`` (MCMC) intermediate
    layer is handled transparently.  ``name_path`` is the path under the
    ``/analyses`` group, joined with ``"/"``.
    """
    for child_name in group.keys():
        try:
            child = group[child_name]
        except KeyError:
            continue
        if not hasattr(child, "keys"):  # not a group
            continue
        full = f"{prefix}/{child_name}" if prefix else child_name
        atype = child.attrs.get("type")
        if atype is not None and _decode(atype) == _ANALYSIS_TYPE:
            yield full, child
        else:
            yield from _discover(child, full)


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------


def _attr_str(group: "h5py.Group", key: str, default: str = "") -> str:
    if key not in group.attrs:
        return default
    return _decode(group.attrs[key])


def _attr_bool(group: "h5py.Group", key: str) -> bool:
    if key not in group.attrs:
        return False
    val = group.attrs[key]
    try:
        return int(val) == 1
    except (TypeError, ValueError):
        return False


def _ds_by_attr(group: "h5py.Group", attr_key: str) -> Optional[np.ndarray]:
    """Return the dataset whose name is stored in ``group.attrs[attr_key]``.

    Returns ``None`` if the attribute is missing or the named dataset is
    absent.  Raises :class:`TypeError` if the attribute resolves to
    something other than an ``h5py.Dataset`` (e.g. a subgroup) — that
    indicates a malformed analysis group rather than missing data, so we
    surface it loudly instead of silently returning ``None``.
    """
    if attr_key not in group.attrs:
        return None
    ds_name = _decode(group.attrs[attr_key])
    if not ds_name or ds_name not in group:
        return None
    obj = group[ds_name]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(
            f"Analysis '{group.name}' attribute {attr_key!r} points at "
            f"'{ds_name}', which is a {type(obj).__name__}, not an "
            f"h5py.Dataset."
        )
    return np.asarray(obj[()])


def _resolve_errors(
    group: "h5py.Group",
    y: np.ndarray,
    target: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (y_lower, y_upper) absolute bounds, or (None, None).

    Priority:
    1. Asymmetric ``yErrorLower`` / ``yErrorUpper`` datasets.
    2. Symmetric ``sqrt(diag(yCovariance))``.
    """
    suffix = "Target" if target else ""
    lower = _ds_by_attr(group, f"yErrorLower{suffix}")
    upper = _ds_by_attr(group, f"yErrorUpper{suffix}")
    if lower is not None and upper is not None:
        if lower.shape != y.shape or upper.shape != y.shape:
            raise ValueError(
                f"yErrorLower{suffix}/yErrorUpper{suffix} shape "
                f"{lower.shape}/{upper.shape} does not match y shape {y.shape}"
            )
        return lower, upper
    cov = _ds_by_attr(group, f"yCovariance{suffix}")
    if cov is not None:
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != y.size:
            raise ValueError(
                f"yCovariance{suffix} shape {cov.shape} not compatible with "
                f"y of size {y.size}"
            )
        sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return y - sigma, y + sigma
    return None, None


# Characters that are invalid in filenames on Windows (a strict superset of
# POSIX's just-``/``).  ASCII control codes (0–31) are also invalid on
# Windows; we strip them too.
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename(name: str) -> str:
    """Make ``name`` safe to use as a single filename component on any OS.

    Replaces filesystem-invalid characters (``< > : " / \\ | ? *`` and ASCII
    control codes) with ``_``, collapses repeated ``_``, and strips trailing
    whitespace and dots (Windows quietly removes those).
    """
    safe = _UNSAFE_FILENAME_CHARS.sub("_", name)
    safe = re.sub(r"_+", "_", safe).rstrip(" .")
    return safe or "_"


_LATEX_FIXES = (
    (re.compile(r"\\hbox"), r"\\mathrm"),
    (re.compile(r"\\le(?![a-zA-Z])"), r"\\leq"),
    (re.compile(r"\\ge(?![a-zA-Z])"), r"\\geq"),
)


def _latex_fix(s: str) -> str:
    """Massage Galacticus LaTeX strings for matplotlib mathtext."""
    if not s:
        return s
    out = s
    for pat, repl in _LATEX_FIXES:
        out = pat.sub(repl, out)
    return out


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _read_analysis(group: "h5py.Group") -> Dict[str, object]:
    """Read all data + metadata for a single ``function1D`` analysis."""
    x = _ds_by_attr(group, "xDataset")
    y = _ds_by_attr(group, "yDataset")
    if x is None or y is None:
        raise KeyError(
            f"Analysis '{group.name}' missing required xDataset or yDataset"
        )
    if x.ndim != 1:
        raise ValueError(
            f"Analysis '{group.name}': xDataset must be 1D, got shape {x.shape}"
        )
    if y.shape != x.shape:
        raise ValueError(
            f"Analysis '{group.name}': yDataset shape {y.shape} does not "
            f"match xDataset shape {x.shape}"
        )
    y_err_lo, y_err_hi = _resolve_errors(group, y, target=False)

    y_target = _ds_by_attr(group, "yDatasetTarget")
    if y_target is not None:
        if y_target.shape != x.shape:
            raise ValueError(
                f"Analysis '{group.name}': yDatasetTarget shape "
                f"{y_target.shape} does not match xDataset shape {x.shape}"
            )
        yt_err_lo, yt_err_hi = _resolve_errors(group, y_target, target=True)
    else:
        yt_err_lo = yt_err_hi = None

    return {
        "x": x,
        "y": y,
        "y_err_lo": y_err_lo,
        "y_err_hi": y_err_hi,
        "y_target": y_target,
        "has_target": y_target is not None,
        "yt_err_lo": yt_err_lo,
        "yt_err_hi": yt_err_hi,
        "x_log": _attr_bool(group, "xAxisIsLog"),
        "y_log": _attr_bool(group, "yAxisIsLog"),
        "x_label": _attr_str(group, "xAxisLabel"),
        "y_label": _attr_str(group, "yAxisLabel"),
        "description": _attr_str(group, "description"),
        "target_label": _attr_str(group, "targetLabel", "Target"),
    }


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def _analyses_root(collection: "Collection") -> "h5py.Group":
    primary = collection._primary
    if _ANALYSES_GROUP not in primary:
        raise KeyError(
            f"No '/{_ANALYSES_GROUP}' group in '{primary.filename}'. "
            "The Galacticus run may not have been configured to write analyses."
        )
    return primary[_ANALYSES_GROUP]


def list_analyses(collection: "Collection", format: str = "astropy"):
    """Return a table of ``function1D`` analyses available in the collection.

    Parameters
    ----------
    collection:
        A :class:`~dendros.Collection`.  Only the primary file is consulted —
        for MPI runs, the ``/analyses`` data has been reduced over all ranks
        and is identical in every file.
    format:
        ``"astropy"`` (default), ``"pandas"``, or ``"tabulate"``.

    Returns
    -------
    astropy.table.Table, pandas.DataFrame, or tabulate-formatted string

    Raises
    ------
    KeyError
        If the file has no top-level ``/analyses`` group.
    """
    root = _analyses_root(collection)
    rows: List[dict] = []
    for name, grp in _discover(root):
        rows.append(
            {
                "name": name,
                "description": _attr_str(grp, "description"),
                "xAxisLabel": _attr_str(grp, "xAxisLabel"),
                "yAxisLabel": _attr_str(grp, "yAxisLabel"),
                "xAxisIsLog": _attr_bool(grp, "xAxisIsLog"),
                "yAxisIsLog": _attr_bool(grp, "yAxisIsLog"),
                "hasTarget": "yDatasetTarget" in grp.attrs,
            }
        )
    rows.sort(key=lambda r: r["name"])
    return _make_table(rows, format=format)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _yerr_2xN(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Convert (lower, upper) absolute bounds into matplotlib's (2, N) yerr."""
    return np.vstack([np.clip(y - lo, 0.0, None), np.clip(hi - y, 0.0, None)])


def _plot_model_curve(ax, info, label, color) -> None:
    """Draw one model curve (with optional errorbars) onto *ax*."""
    x, y = info["x"], info["y"]
    ylo, yhi = info["y_err_lo"], info["y_err_hi"]
    yerr = _yerr_2xN(y, ylo, yhi) if ylo is not None else None
    ax.errorbar(
        x, y,
        yerr=yerr,
        fmt="-",
        lw=2.0,
        color=color,
        ecolor=color,
        elinewidth=1.0,
        capsize=0,
        label=label,
        zorder=3,
    )


def _plot_target(ax, info) -> None:
    """Draw the target/observational overlay onto *ax*."""
    yt = info["y_target"]
    tlo, thi = info["yt_err_lo"], info["yt_err_hi"]
    terr = _yerr_2xN(yt, tlo, thi) if tlo is not None else None
    ax.errorbar(
        info["x"], yt,
        yerr=terr,
        fmt="o",
        ms=5,
        mfc=_TARGET_COLOR,
        mec=_TARGET_COLOR,
        ecolor=_TARGET_COLOR,
        elinewidth=1.0,
        capsize=2,
        linestyle="none",
        label=info["target_label"] or "Target",
        zorder=4,
    )


def _apply_axis_metadata(ax, name: str, info: Dict[str, object]) -> None:
    """Set axis scales, labels, title, grid, legend from one analysis info."""
    if info["x_log"]:
        ax.set_xscale("log")
    if info["y_log"]:
        ax.set_yscale("log")
    ax.set_xlabel(_latex_fix(info["x_label"]))
    ax.set_ylabel(_latex_fix(info["y_label"]))
    ax.set_title(_latex_fix(info["description"]) or name)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False, loc="best")


def _plot_one(
    name: str,
    info: Dict[str, object],
    *,
    show_target: bool,
    figsize: Tuple[float, float],
    dpi: int,
) -> "Figure":
    import matplotlib.pyplot as plt

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _plot_model_curve(ax, info, label="Model", color=_MODEL_COLOR)
        if show_target and info["has_target"]:
            _plot_target(ax, info)
        _apply_axis_metadata(ax, name, info)
        fig.tight_layout()

    # Detach from pyplot's state machine so that returning many Figures from
    # a notebook cell doesn't trigger duplicate inline-backend rendering and
    # so callers don't accumulate memory.  The Figure itself remains valid:
    # its axes, savefig, and IPython display all continue to work.
    plt.close(fig)
    return fig


def _plot_multi(
    name: str,
    infos: List[Tuple[str, Dict[str, object]]],
    *,
    show_target: bool,
    figsize: Tuple[float, float],
    dpi: int,
) -> "Figure":
    """Plot one analysis with overlaid curves from several models.

    *infos* is a list of ``(label, info_dict)`` pairs in the order the
    models should be drawn / appear in the legend.  Only the first model
    that has a target supplies the target overlay — it should be identical
    across models, so plotting it once keeps the figure uncluttered.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        cmap = plt.get_cmap(_MULTI_MODEL_CMAP)
        n_cmap = getattr(cmap, "N", 10)
        for i, (label, info) in enumerate(infos):
            _plot_model_curve(ax, info, label=label, color=cmap(i % n_cmap))

        if show_target:
            for _, info in infos:
                if info["has_target"]:
                    _plot_target(ax, info)
                    break

        # Axis metadata comes from the first contributing model — all
        # models claiming to be the "same analysis" share the same axes.
        _apply_axis_metadata(ax, name, infos[0][1])
        fig.tight_layout()

    plt.close(fig)
    return fig


def _select_names(
    available: List[str], name: Union[None, str, List[str]]
) -> List[str]:
    if name is None:
        return list(available)
    requested = [name] if isinstance(name, str) else list(name)
    available_set = set(available)
    missing = [n for n in requested if n not in available_set]
    if missing:
        raise KeyError(
            f"Analyses not found: {missing!r}. Available: {available!r}"
        )
    return requested


_MultiInput = Union[
    "Collection",
    Sequence["Collection"],
    Mapping[str, "Collection"],
]


def _normalize_collections(
    collection: _MultiInput,
    labels: Optional[Sequence[str]],
) -> Tuple[bool, List[Tuple[str, "Collection"]]]:
    """Normalize the ``collection`` argument into ``(is_multi, [(label, c), ...])``.

    A single :class:`Collection` produces ``is_multi=False`` and preserves
    legacy single-curve, ``label="Model"`` behaviour.  Lists and dicts —
    even of length 1 — produce ``is_multi=True`` so the legend always
    identifies the model.
    """
    if isinstance(collection, Collection):
        if labels is not None:
            raise ValueError(
                "labels= is only meaningful when passing several Collections; "
                "pass a list or dict of Collections."
            )
        return False, [("Model", collection)]

    if isinstance(collection, Mapping):
        if labels is not None:
            raise ValueError(
                "labels= cannot be combined with a dict input; the dict keys "
                "already specify labels."
            )
        items: List[Tuple[str, Collection]] = []
        for label, c in collection.items():
            if not isinstance(c, Collection):
                raise TypeError(
                    f"Expected Collection values in dict; got "
                    f"{type(c).__name__} for key {label!r}."
                )
            items.append((str(label), c))
        if not items:
            raise ValueError("collection mapping is empty.")
        return True, items

    try:
        seq = list(collection)
    except TypeError as exc:
        raise TypeError(
            "collection must be a Collection, a list of Collections, or a "
            f"dict of {{label: Collection}}; got {type(collection).__name__!r}."
        ) from exc
    if not seq:
        raise ValueError("collection sequence is empty.")
    for c in seq:
        if not isinstance(c, Collection):
            raise TypeError(
                f"Expected Collection elements; got {type(c).__name__}."
            )

    if labels is not None:
        labels_list = list(labels)
        if len(labels_list) != len(seq):
            raise ValueError(
                f"labels has length {len(labels_list)} but {len(seq)} "
                "collections were provided."
            )
        return True, list(zip((str(label) for label in labels_list), seq))

    auto = [_default_model_label(c._files[0]) for c in seq]
    duplicates = [lbl for lbl in auto if auto.count(lbl) > 1]
    if duplicates:
        raise ValueError(
            f"Default labels collide ({sorted(set(duplicates))!r}). Pass an "
            "explicit labels= sequence or a dict {label: Collection}."
        )
    return True, list(zip(auto, seq))


def plot_analyses(
    collection: _MultiInput,
    name: Union[None, str, List[str]] = None,
    output_directory: Union[None, str, "Path"] = None,
    *,
    labels: Optional[Sequence[str]] = None,
    show_target: bool = True,
    figsize: Tuple[float, float] = (7.0, 5.0),
    dpi: int = 120,
    file_format: str = "pdf",
) -> Dict[str, "Figure"]:
    """Plot one, several, or all ``function1D`` analyses.

    A single :class:`~dendros.Collection` produces one model curve per
    figure (legacy behaviour).  A list, dict, or
    :class:`~dendros.ModelCollection` of Collections overlays one curve
    per model on each figure, plotting the target/observational overlay
    once (since it is shared across models).  The union of analyses
    discovered across models is plotted — figures whose analysis is
    absent from a given model simply do not include its curve.

    Parameters
    ----------
    collection:
        A :class:`~dendros.Collection`; a sequence of Collections; or a
        mapping ``{label: Collection}`` (e.g. one returned by
        :func:`~dendros.open_models`).
    name:
        ``None`` (default) plots every ``function1D`` analysis discovered
        across all models.  A single name (str) or list of names plots
        only those.
    output_directory:
        If given, each figure is also saved as
        ``<output_directory>/<safe_name>.<file_format>``.  The directory
        is created if it does not exist.
    labels:
        Optional sequence of legend labels, one per Collection, used only
        when *collection* is a list/tuple of Collections.  When omitted,
        each model is labelled by its primary file's stem (with any
        ``:MPIxxxx`` suffix stripped).  Cannot be combined with a dict
        input.
    show_target:
        If ``True`` (default), overlay target/observational data when
        present.  For multi-model plots the target is plotted only once,
        from the first model that has it.
    figsize, dpi, file_format:
        Forwarded to matplotlib.

    Returns
    -------
    dict
        Mapping from analysis name to :class:`matplotlib.figure.Figure`.

    Raises
    ------
    KeyError
        If a model has no ``/analyses`` group, or if a requested name is
        missing from every model.
    ImportError
        If matplotlib is not installed; install with ``pip install
        'dendros[plot]'``.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib is not installed. "
            "Install it with: pip install 'dendros[plot]'"
        ) from exc

    is_multi, label_coll = _normalize_collections(collection, labels)

    # Discover analyses per collection and build the union, preserving the
    # first-appearance order within each model (then sorted at the end).
    per_collection: List[Tuple[str, Dict[str, "h5py.Group"]]] = []
    union: List[str] = []
    seen = set()
    for label, c in label_coll:
        root = _analyses_root(c)
        discovered = dict(_discover(root))
        per_collection.append((label, discovered))
        for n in discovered:
            if n not in seen:
                seen.add(n)
                union.append(n)

    if not union:
        warnings.warn(
            f"No '{_ANALYSIS_TYPE}' analyses found under "
            f"'/{_ANALYSES_GROUP}'.",
            UserWarning,
            stacklevel=2,
        )
        return {}

    union.sort()
    selected = _select_names(union, name)

    out_dir: Optional[Path] = None
    if output_directory is not None:
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

    figs: Dict[str, "Figure"] = {}
    for n in selected:
        contributing: List[Tuple[str, Dict[str, object]]] = []
        for label, discovered in per_collection:
            grp = discovered.get(n)
            if grp is not None:
                contributing.append((label, _read_analysis(grp)))
        if not contributing:
            continue  # _select_names guarantees this can't happen

        if is_multi:
            fig = _plot_multi(
                n, contributing,
                show_target=show_target,
                figsize=figsize,
                dpi=dpi,
            )
        else:
            fig = _plot_one(
                n, contributing[0][1],
                show_target=show_target,
                figsize=figsize,
                dpi=dpi,
            )
        figs[n] = fig
        if out_dir is not None:
            fig.savefig(
                out_dir / f"{_safe_filename(n)}.{file_format}",
                format=file_format,
            )

    return figs
