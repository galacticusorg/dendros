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
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union

import h5py
import numpy as np

from ._collection import _decode, _make_table

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ._collection import Collection


_ANALYSIS_TYPE = "function1D"
_ANALYSES_GROUP = "analyses"

# Style ---------------------------------------------------------------------

_MODEL_COLOR = "#1f4e79"   # deep blue
_TARGET_COLOR = "#d1495b"  # brick red

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
    y_err_lo, y_err_hi = _resolve_errors(group, y, target=False)

    y_target = _ds_by_attr(group, "yDatasetTarget")
    if y_target is not None:
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

        x = info["x"]
        y = info["y"]
        ylo, yhi = info["y_err_lo"], info["y_err_hi"]
        yerr = _yerr_2xN(y, ylo, yhi) if ylo is not None else None
        ax.errorbar(
            x, y,
            yerr=yerr,
            fmt="-",
            lw=2.0,
            color=_MODEL_COLOR,
            ecolor=_MODEL_COLOR,
            elinewidth=1.0,
            capsize=0,
            label="Model",
            zorder=3,
        )

        if show_target and info["has_target"]:
            yt = info["y_target"]
            tlo, thi = info["yt_err_lo"], info["yt_err_hi"]
            terr = _yerr_2xN(yt, tlo, thi) if tlo is not None else None
            ax.errorbar(
                x, yt,
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

        if info["x_log"]:
            ax.set_xscale("log")
        if info["y_log"]:
            ax.set_yscale("log")

        ax.set_xlabel(_latex_fix(info["x_label"]))
        ax.set_ylabel(_latex_fix(info["y_label"]))
        title = _latex_fix(info["description"]) or name
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()

    # Detach from pyplot's state machine so that returning many Figures from
    # a notebook cell doesn't trigger duplicate inline-backend rendering and
    # so callers don't accumulate memory.  The Figure itself remains valid:
    # its axes, savefig, and IPython display all continue to work.
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


def plot_analyses(
    collection: "Collection",
    name: Union[None, str, List[str]] = None,
    output_directory: Union[None, str, "Path"] = None,
    *,
    show_target: bool = True,
    figsize: Tuple[float, float] = (7.0, 5.0),
    dpi: int = 120,
    file_format: str = "pdf",
) -> Dict[str, "Figure"]:
    """Plot one, several, or all ``function1D`` analyses from a collection.

    Parameters
    ----------
    collection:
        A :class:`~dendros.Collection`.  Only the primary file is read; for
        MPI runs the ``/analyses`` data is identical in every rank's file.
    name:
        ``None`` (default) plots every ``function1D`` analysis discovered.
        A single name (str) or list of names plots only those.
    output_directory:
        If given, each figure is also saved as
        ``<output_directory>/<safe_name>.<file_format>``.  The directory is
        created if it does not exist.
    show_target:
        If ``True`` (default), overlay target/observational data when present.
    figsize, dpi, file_format:
        Forwarded to matplotlib.

    Returns
    -------
    dict
        Mapping from analysis name to :class:`matplotlib.figure.Figure`.

    Raises
    ------
    KeyError
        If no ``/analyses`` group is present, or if a requested name is
        missing.
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

    root = _analyses_root(collection)
    discovered = list(_discover(root))
    if not discovered:
        warnings.warn(
            f"No '{_ANALYSIS_TYPE}' analyses found under "
            f"'/{_ANALYSES_GROUP}'.",
            UserWarning,
            stacklevel=2,
        )
        return {}

    by_name = {n: g for n, g in discovered}
    selected = _select_names(sorted(by_name.keys()), name)

    out_dir: Optional[Path] = None
    if output_directory is not None:
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

    figs: Dict[str, "Figure"] = {}
    for n in selected:
        info = _read_analysis(by_name[n])
        fig = _plot_one(
            n, info,
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
