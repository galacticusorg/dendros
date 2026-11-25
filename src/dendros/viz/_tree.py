"""River diagrams of galaxy merger trees.

Galacticus' ``galaxyMergerTree`` node operator samples the properties of every galaxy
along every branch of its merger tree, and records where branches merge.  Here each
branch is drawn as a ribbon running up the page with time, its width set by its mass,
its color by a property such as its specific star formation rate.  Progenitors flow
into the branch they merge with.

The layout is computed independently at every time: around each branch, the branches
merging into it are stacked outwards on alternating sides, in order of merger time,
so that the earliest to merge lies innermost.  A branch's ribbon then never has to
cross another to reach the branch it merges into.  Each branch reserves space for
itself and for everything merging into it, so ribbons never overlap.  Space opens
smoothly as a branch appears and closes as it merges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_PREFIX = "galaxyMergerTree"


@dataclass
class Branch:
    """One branch of a galaxy merger tree: the history of one galaxy until it merges."""

    id: int
    #: Sample times [Gyr], increasing.
    time: np.ndarray
    #: Sampled properties, keyed by the extractor's name (e.g. ``"MassStellarTotal"``).
    properties: Dict[str, np.ndarray]
    #: The branch it merges into (``None`` for the main branch and for surviving satellites).
    parent: Optional[int] = None
    #: The time at which it merges [Gyr] (``None`` if it survives to the end).
    time_merge: Optional[float] = None
    children: List[int] = field(default_factory=list)
    #: True for the main branch of a satellite that survives to the end of the tree.
    satellite: bool = False
    #: Times [Gyr] at which other branches merged into this one.
    merger_times: List[float] = field(default_factory=list)

    @property
    def time_start(self) -> float:
        return float(self.time[0])

    @property
    def time_end(self) -> float:
        return float(self.time_merge if self.time_merge is not None else self.time[-1])


@dataclass
class GalaxyMergerTree:
    """The galaxy merger tree of one central galaxy, optionally with the trees of its surviving satellites."""

    branches: Dict[int, Branch]
    #: The main branch of the central galaxy.
    root: int
    tree_index: int
    time_final: float

    def properties(self) -> List[str]:
        return list(self.branches[self.root].properties)


def _branches_of_node(group, row: int) -> Tuple[Dict[int, Branch], int]:
    """Read the branches recorded for the galaxy at ``row``; return them and the id of its main branch."""
    counts = np.asarray(group[f"{_PREFIX}Count"][row], dtype=np.int64)
    ids = np.asarray(group[f"{_PREFIX}NodeIndex"][row], dtype=np.int64)
    time = np.asarray(group[f"{_PREFIX}Time"][row], dtype=float)
    names = [
        k[len(_PREFIX) :]
        for k in group.keys()
        if k.startswith(_PREFIX)
        and k[len(_PREFIX) :]
        not in (
            "Count",
            "NodeIndex",
            "Time",
            "MergeCentralIndex",
            "MergeSatelliteIndex",
            "MergeTime",
        )
    ]
    values = {
        name: np.asarray(group[_PREFIX + name][row], dtype=float) for name in names
    }
    end = np.cumsum(counts)
    start = end - counts
    branches = {}
    for k, b in enumerate(ids):
        s = slice(start[k], end[k])
        branches[int(b)] = Branch(
            int(b), time[s], {name: v[s] for name, v in values.items()}
        )
    centrals = np.asarray(group[f"{_PREFIX}MergeCentralIndex"][row], dtype=np.int64)
    satellites = np.asarray(group[f"{_PREFIX}MergeSatelliteIndex"][row], dtype=np.int64)
    times = np.asarray(group[f"{_PREFIX}MergeTime"][row], dtype=float)
    for c, s, t in zip(centrals, satellites, times):
        if int(s) in branches and int(c) in branches:
            branches[int(s)].parent = int(c)
            branches[int(s)].time_merge = float(t)
            branches[int(c)].children.append(int(s))
    for branch in branches.values():
        branch.merger_times = sorted(branches[k].time_merge for k in branch.children)
    roots = [b for b in branches.values() if b.parent is None]
    if len(roots) != 1:
        raise ValueError(
            f"expected one main branch for the galaxy at row {row}, found {len(roots)}"
        )
    return branches, roots[0].id


def galaxy_merger_tree(
    c, output, tree_index: int, include_satellites: bool = True
) -> GalaxyMergerTree:
    """Read the galaxy merger tree of the central galaxy of merger tree ``tree_index``.

    The model must have run the ``galaxyMergerTree`` node operator, and output its data
    with the ``galaxyMergerTree`` node property extractor (together with ``nodeIndices``).
    With ``include_satellites``, the trees of the central's satellites that survive to
    the output are attached to its main branch, as branches that never merge.
    """
    name = output if isinstance(output, str) else f"Output{int(output)}"
    out = c[f"Outputs/{name}"]
    group = out["nodeData"]
    if f"{_PREFIX}Count" not in group:
        raise KeyError(
            "output lacks galaxy merger tree data: run the galaxyMergerTree node operator and extractor"
        )
    trees = np.asarray(out["mergerTreeIndex"][()])
    select = np.nonzero(trees == tree_index)[0]
    if select.size != 1:
        raise KeyError(f"merger tree {tree_index} not found in {name}")
    first = int(np.asarray(out["mergerTreeStartIndex"][()])[select[0]])
    count = int(np.asarray(out["mergerTreeCount"][()])[select[0]])
    rows = np.arange(first, first + count)
    isolated = np.asarray(group["nodeIsIsolated"][first : first + count])
    centrals = rows[isolated == 1]
    if centrals.size != 1:
        raise ValueError(
            f"expected one isolated galaxy in merger tree {tree_index}, found {centrals.size}"
        )
    branches, root = _branches_of_node(group, int(centrals[0]))
    if include_satellites:
        for row in rows[isolated == 0]:
            sub, sub_root = _branches_of_node(group, int(row))
            if set(sub) & set(branches):
                raise ValueError(
                    "branch identifiers of a satellite collide with those of the central"
                )
            sub[sub_root].satellite = True
            sub[sub_root].parent = root
            branches[root].children.append(sub_root)
            branches.update(sub)
    time_final = max(float(b.time[-1]) for b in branches.values())
    return GalaxyMergerTree(branches, root, int(tree_index), time_final)


def main_branch_rows(
    c, tree: GalaxyMergerTree, id_dataset: str = "nodeUniqueIDBranchTip"
) -> List[Tuple[object, int]]:
    """Find the galaxy on the main branch of ``tree`` in each output.

    Branches of a galaxy merger tree are identified by the unique ID of their branch tip,
    which the main progenitor of a galaxy carries at all times.  The model must output it
    (the ``uniqueIDBranchTip`` node property extractor).  Returns a list of
    ``(output, row)`` pairs, in order of increasing time, where ``output`` is the
    :class:`~dendros.OutputMeta` of an output and ``row`` the index of the galaxy in it
    (for :func:`~dendros.viz.galaxy_scene`); outputs in which the galaxy is absent (e.g.
    not yet above an output filter's mass threshold) are skipped.
    """
    rows = []
    for output in c.outputs:
        group = c[f"{output.path}/nodeData"]
        if id_dataset not in group:
            raise KeyError(
                f"{output.name} lacks '{id_dataset}': run the uniqueIDBranchTip node property extractor"
            )
        ids = np.asarray(group[id_dataset][()])
        match = np.nonzero(ids == tree.root)[0]
        if match.size > 1:
            raise ValueError(
                f"branch {tree.root} matches {match.size} galaxies in {output.name}"
            )
        if match.size == 1:
            rows.append((output, int(match[0])))
    return sorted(rows, key=lambda pair: pair[0].time)


def halo_merger_tree(c, tree_index: int, mass: str = "massBasic") -> GalaxyMergerTree:
    """Read the dark matter halo merger tree ``tree_index``, as written before evolution.

    The model must run the ``outputStructure`` merger tree operator with (at least) the
    ``nodeIndices``, ``massBasic`` and ``time`` node property extractors, which writes
    each tree to ``mergerTreeStructures/tree<index>``.  Each halo's main progenitor
    is its most massive one; a branch follows main progenitors back in time, and merges
    into the branch of its descendant at the descendant's time.  The result has the same
    form as :func:`galaxy_merger_tree`, with the halo mass as the property
    ``"MassBasic"`` (and any other extracted properties under their dataset names, with
    the first letter capitalized), so it can be laid out and drawn in the same ways.
    """
    structures = c["mergerTreeStructures"]
    # The groups are named "tree<index>" (the operator's documentation says "mergerTree<index>").
    name = next(
        (
            n
            for n in (f"tree{int(tree_index)}", f"mergerTree{int(tree_index)}")
            if n in structures
        ),
        None,
    )
    if name is None:
        raise KeyError(f"merger tree {tree_index} not found in mergerTreeStructures")
    group = structures[name]
    for required in ("nodeIndex", "parentIndex", mass, "time"):
        if required not in group:
            raise KeyError(
                f"merger tree structure lacks '{required}': run the outputStructure operator with the "
                "nodeIndices, massBasic and time extractors"
            )
    index = np.asarray(group["nodeIndex"][()], dtype=np.int64)
    parent = np.asarray(group["parentIndex"][()], dtype=np.int64)
    time = np.asarray(group["time"][()], dtype=float)
    names = [
        k
        for k in group.keys()
        if k
        not in (
            "nodeIndex",
            "parentIndex",
            "siblingIndex",
            "satelliteIndex",
            "nodeIsIsolated",
            "time",
        )
        and np.ndim(group[k]) == 1
        and np.size(group[k]) == index.size
    ]
    values = {
        n[0].upper() + n[1:]: np.asarray(group[n][()], dtype=float) for n in names
    }
    halo_mass = values[mass[0].upper() + mass[1:]]
    position = {int(k): i for i, k in enumerate(index)}
    # The main progenitor of each halo: its most massive progenitor.
    main = {}
    for i, p in enumerate(parent):
        if p >= 0 and int(p) in position:
            p = int(p)
            if p not in main or halo_mass[i] > halo_mass[position[main[p]]]:
                main[p] = int(index[i])
    # A branch ends at each halo which is not the main progenitor of its descendant (or has no descendant), and is traced
    # back through main progenitors.
    branches = {}
    branch_of = {}
    for i, k in enumerate(index):
        k = int(k)
        p = int(parent[i])
        if p >= 0 and p in position and main.get(p) == k:
            continue
        chain = [k]
        while chain[-1] in main:
            chain.append(main[chain[-1]])
        rows = np.array([position[n] for n in reversed(chain)])
        branch = Branch(k, time[rows], {n: v[rows] for n, v in values.items()})
        branches[k] = branch
        for n in chain:
            branch_of[n] = k
    root = None
    for b, branch in branches.items():
        p = int(parent[position[b]])
        if p >= 0 and p in position:
            branch.parent = branch_of[p]
            branch.time_merge = float(time[position[p]])
            branches[branch.parent].children.append(b)
        else:
            if root is not None:
                raise ValueError(f"merger tree {tree_index} has more than one root")
            root = b
    for branch in branches.values():
        branch.merger_times = sorted(branches[k].time_merge for k in branch.children)
    time_final = float(np.max(time))
    return GalaxyMergerTree(branches, root, int(tree_index), time_final)


def halo_growth_rate(
    branch: Branch, mass: str = "MassBasic", window: float = 1.0
) -> np.ndarray:
    """log10 of the specific growth rate [Gyr⁻¹] along a branch, averaged over the preceding ``window`` Gyr.

    The rate at time t is ``ln[M(t)/M(t-window)]/window``, with M interpolated in time (and
    held at its first value before the branch begins, where the window is shortened to the
    branch's age).  Averaging smooths over the jumps in mass at mergers.  NaN where the
    mass does not grow.
    """
    m = np.maximum(branch.properties[mass], 1.0e-30)
    t = branch.time
    if m.size < 2:
        return np.full(m.shape, np.nan)
    earlier = np.maximum(t - window, t[0])
    span = t - earlier
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.log(m / np.interp(earlier, t, m)) / span
        # At the first sample the window is empty: use the rate over the first interval.
        rate[0] = np.log(m[1] / m[0]) / (t[1] - t[0])
        return np.where(rate > 0.0, np.log10(rate), np.nan)


def _smooth_envelope(values, size):
    """A smooth function never (much) below ``values``: a running maximum over ``size`` grid points, then a Gaussian
    smoothing of the same width.  Used for the space reserved by each branch, so that the layout changes smoothly while
    ribbons still never overlap."""
    if size < 2:
        return values
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d

    return gaussian_filter1d(
        maximum_filter1d(values, size=2 * size + 1, mode="nearest"),
        size / 2.0,
        mode="nearest",
    )


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass
class RiverLayout:
    """Ribbon geometry on a common grid of times.

    For each branch, ``center`` and ``width`` are arrays over ``time`` (NaN where the
    branch does not exist), in units in which the widest ribbon has width one.
    """

    time: np.ndarray
    #: The vertical coordinate of each time (``log10(time)`` or ``time``).
    y: np.ndarray
    center: Dict[int, np.ndarray]
    width: Dict[int, np.ndarray]
    #: Draw order: parents after their children.
    order: List[int]
    #: ``"log"`` or ``"linear"``: how ``y`` follows from ``time``.
    axis: str = "log"

    def coordinate(self, time) -> np.ndarray:
        """The vertical coordinate of ``time`` [Gyr]."""
        return np.log10(time) if self.axis == "log" else np.asarray(time, dtype=float)


def river_layout(
    tree: GalaxyMergerTree,
    mass: str = "MassStellarTotal",
    exponent: float = 0.5,
    gap: float = 0.08,
    n_times: int = 800,
    axis: str = "log",
    transition: float = 0.04,
    mass_minimum: float = 0.0,
    width_scale=None,
    smoothing: Optional[float] = None,
) -> RiverLayout:
    """Lay out a galaxy merger tree as a river diagram.

    Parameters
    ----------
    mass:
        The property setting ribbon widths.
    exponent:
        Widths scale as ``mass**exponent``. With 1, widths add at each merger but
        minor progenitors are too thin to see; smaller values show them.
    gap:
        Space between neighboring ribbons, in units of the widest ribbon's width.
    n_times:
        Number of times in the layout grid.
    axis:
        ``"log"`` to space times logarithmically (which spreads out the early, merger-rich
        epochs) or ``"linear"``.
    transition:
        Duration, in units of the vertical coordinate, over which a branch's space opens
        as it appears and closes as it merges.
    mass_minimum:
        Branches whose mass never reaches this are omitted (with everything merging into them).
    width_scale:
        Optional function scaling ribbon widths and gaps with time: called with the
        fraction of the way from the earliest time (0) to the final time (1) of the
        layout, it returns the factor by which to multiply them.  Used by
        :func:`radial_layout` so that widths are true once the layout is bent into a circle.
    smoothing:
        Time scale (in units of the vertical coordinate) over which the space reserved for
        each branch is smoothed.  Space is reserved through a smooth envelope that never
        falls below what is needed, so ribbons never overlap; a longer time scale reduces
        the sideways shifts of outer ribbons as space opens and closes inside them, at the
        cost of some empty space.  Defaults to ``transition``.
    """
    if axis not in ("log", "linear"):
        raise ValueError("axis must be 'log' or 'linear'")
    branches = tree.branches

    def keep(b):
        return np.nanmax(branches[b].properties[mass]) >= mass_minimum

    def coordinate(t):
        return np.log10(t) if axis == "log" else np.asarray(t, dtype=float)

    kept = {b for b in branches if keep(b)}
    # Drop branches whose parent chain has been dropped.
    changed = True
    while changed:
        changed = False
        for b in list(kept):
            p = branches[b].parent
            if p is not None and p not in kept:
                kept.discard(b)
                changed = True
    time_start = min(branches[b].time_start for b in kept)
    y = np.linspace(coordinate(time_start), coordinate(tree.time_final), n_times)
    time = 10.0**y if axis == "log" else y
    mass_reference = np.nanmax(branches[tree.root].properties[mass])
    scale = (
        np.ones_like(y)
        if width_scale is None
        else np.asarray(width_scale((y - y[0]) / (y[-1] - y[0])), dtype=float)
    )
    gap = gap * scale

    width, alive, factor = {}, {}, {}
    for b in kept:
        branch = branches[b]
        y_start, y_end = coordinate(branch.time_start), coordinate(branch.time_end)
        alive[b] = (y >= y_start) & (y <= y_end)
        m = np.interp(
            y, coordinate(branch.time), np.maximum(branch.properties[mass], 1.0e-30)
        )
        width[b] = np.where(
            alive[b], np.maximum(m / mass_reference, 0.0) ** exponent * scale, 0.0
        )
        # The space a branch reserves opens as it appears and closes as it merges.
        opening = _smoothstep((y - y_start) / transition)
        closing = (
            _smoothstep((y_end - y) / transition)
            if branch.time_merge is not None
            else 1.0
        )
        factor[b] = np.where(alive[b], opening * closing, 0.0)

    # Order each branch's children: those merging earliest innermost; surviving satellites outermost, larger ones first.
    def child_order(b):
        children = [c for c in branches[b].children if c in kept]
        return sorted(
            children,
            key=lambda c: (
                branches[c].time_merge is None,
                (
                    branches[c].time_merge
                    if branches[c].time_merge is not None
                    else -np.nanmax(branches[c].properties[mass])
                ),
            ),
        )

    children = {b: child_order(b) for b in kept}
    side = {}
    for b in kept:
        for k, c in enumerate(children[b]):
            side[c] = 1.0 if k % 2 == 0 else -1.0

    # Extents of each branch's subtree to its left and right, at each time (bottom up).
    extent_left, extent_right = {}, {}

    # Space is reserved with a smooth envelope of each quantity, over the transition time.
    smoothing = int(
        round((transition if smoothing is None else smoothing) / (y[1] - y[0]))
    )
    half = {b: _smooth_envelope(0.5 * width[b], smoothing) for b in kept}
    footprint = {}

    def extents(b):
        left = half[b].copy()
        right = half[b].copy()
        for c in children[b]:
            extents(c)
            footprint[c] = _smooth_envelope(
                (gap + extent_left[c] + extent_right[c]) * factor[c], smoothing
            )
            if side[c] > 0.0:
                right += footprint[c]
            else:
                left += footprint[c]
        extent_left[b], extent_right[b] = left, right

    extents(tree.root)

    # Positions (top down).
    center = {tree.root: np.zeros_like(y)}
    order = []

    def place(b):
        filled = {1.0: half[b], -1.0: half[b]}
        for c in children[b]:
            s = side[c]
            inner = extent_left[c] if s > 0.0 else extent_right[c]
            # A branch sits at the inner edge of the space reserved for it. As it merges, it swoops into its parent's
            # ribbon, ending just inside the parent's edge.
            offset = filled[s] + gap + inner
            if branches[c].time_merge is not None:
                living = _smoothstep(
                    (coordinate(branches[c].time_end) - y) / (2.0 * transition)
                )
                # The target uses the true (not the smoothed) widths, so the branch ends within its parent's ribbon.
                offset = living * offset + (1.0 - living) * np.maximum(
                    0.5 * (width[b] - width[c]), 0.0
                )
            center[c] = center[b] + s * offset
            filled[s] = filled[s] + footprint[c]
            place(c)
        order.append(b)

    place(tree.root)
    for b in kept:
        center[b] = np.where(alive[b], center[b], np.nan)
        width[b] = np.where(alive[b], width[b], np.nan)
    return RiverLayout(
        time, y, {b: center[b] for b in kept}, {b: width[b] for b in kept}, order, axis
    )


def specific_star_formation_rate(branch: Branch) -> np.ndarray:
    """Specific star formation rate [yr⁻¹] along a branch (Galacticus rates are per Gyr)."""
    mass = branch.properties["MassStellarTotal"]
    rate = branch.properties["TotalStarFormationRate"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mass > 0.0, rate / mass / 1.0e9, np.nan)


def river_colormap():
    """Quenched (red) through transitional (pale gold) to star-forming (blue), for dark backgrounds."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "river", ["#d7301f", "#fc8d59", "#fee8a8", "#d9f0ff", "#74b9ff", "#3a6cf4"]
    )


def _default_color(branch):
    """log10 of the specific star formation rate [yr⁻¹]."""
    with np.errstate(divide="ignore"):
        return np.log10(specific_star_formation_rate(branch))


def render_river(
    tree: GalaxyMergerTree,
    layout: RiverLayout,
    npix: int = 1200,
    color=None,
    norm=None,
    cmap=None,
    background=(0.02, 0.024, 0.04),
    shading: float = 0.45,
    satellite_brightness: float = 0.7,
    glow: float = 0.35,
    glow_sigma: float = 0.006,
    margin: float = 0.03,
):
    """Render a river diagram as an image, one image row per time in ``layout``.

    Each ribbon is shaded across its width like a lit tube (``shading`` sets how much
    darker its edges are than its center) and drawn over the ribbons that merge into it.
    ``color`` gives the value to color by along each branch: a function of a
    :class:`Branch` returning an array over its sample times (default: log10 of the
    specific star formation rate [yr⁻¹], mapped by ``norm``, default -12 to -9.3).
    Branches of surviving satellites are dimmed by ``satellite_brightness``.  ``glow``
    adds a soft halo around bright ribbons, of width ``glow_sigma`` (in units of the
    image width).

    Returns ``(image, extent, mappable)``: an RGB float image with time increasing
    upwards (row 0 is the earliest time; draw it with ``origin="lower"``), its extent
    ``(x0, x1, y0, y1)`` in layout coordinates, and a ``ScalarMappable`` for a colorbar.
    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    if color is None:
        color = _default_color
        norm = norm or Normalize(-12.0, -9.3)
    cmap = cmap or river_colormap()
    if norm is None:
        values = np.concatenate([color(tree.branches[b]) for b in layout.order])
        norm = Normalize(*np.nanpercentile(values[np.isfinite(values)], [2.0, 98.0]))
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    left = min(
        np.nanmin(layout.center[b] - 0.5 * layout.width[b]) for b in layout.order
    )
    right = max(
        np.nanmax(layout.center[b] + 0.5 * layout.width[b]) for b in layout.order
    )
    pad = margin * (right - left)
    x0, x1 = left - pad, right + pad
    scale = npix / (x1 - x0)  # pixels per layout unit
    n_rows = layout.y.size
    image = np.empty((n_rows, npix, 3))
    image[:] = background
    edges = np.arange(npix + 1)
    for b in layout.order:
        branch = tree.branches[b]
        rows = np.nonzero(np.isfinite(layout.center[b]))[0]
        if rows.size == 0:
            continue
        center = (layout.center[b][rows] - x0) * scale
        half = 0.5 * np.maximum(
            layout.width[b][rows] * scale, 0.75
        )  # never thinner than a pixel and a half
        c = np.asarray(color(branch), dtype=float)
        usable = np.isfinite(c) | np.isneginf(
            c
        )  # -inf (no star formation) is a value; NaN is a sample to skip
        if not np.any(usable):
            c, usable = np.full(c.shape, norm.vmin), np.ones(c.shape, dtype=bool)
        c = np.where(np.isneginf(c), norm.vmin, c)
        values = np.interp(
            layout.y[rows], layout.coordinate(branch.time[usable]), c[usable]
        )
        rgb = mappable.to_rgba(values)[:, :3]
        if _is_satellite_subtree(tree, b):
            rgb = rgb * satellite_brightness
        lo = int(max(np.floor(np.min(center - half)), 0))
        hi = int(min(np.ceil(np.max(center + half)) + 1, npix))
        if hi <= lo:
            continue
        pix_lo = edges[lo:hi][None, :]
        pix_hi = pix_lo + 1.0
        # Fraction of each pixel covered by the ribbon (antialiasing), and the tube shading at the pixel center.
        coverage = np.clip(
            np.minimum(pix_hi, (center + half)[:, None])
            - np.maximum(pix_lo, (center - half)[:, None]),
            0.0,
            1.0,
        )
        u = np.clip((pix_lo + 0.5 - center[:, None]) / half[:, None], -1.0, 1.0)
        light = 1.0 - shading * u * u
        target = image[rows, lo:hi]
        image[rows, lo:hi] = (
            target * (1.0 - coverage[..., None])
            + coverage[..., None] * light[..., None] * rgb[:, None, :]
        )
    image = _add_glow(image, background, glow, glow_sigma * npix)
    return np.clip(image, 0.0, 1.0), (x0, x1, layout.y[0], layout.y[-1]), mappable


def _add_glow(image, background, glow, sigma):
    """Screen-blend a blurred copy of the light above the background onto ``image``."""
    if glow <= 0.0:
        return image
    from scipy.ndimage import gaussian_filter

    luminous = np.clip(image - np.asarray(background), 0.0, None)
    halo = np.stack(
        [gaussian_filter(luminous[..., k], sigma) for k in range(3)], axis=-1
    )
    return 1.0 - (1.0 - image) * (1.0 - glow * halo)


def _radius(fraction, inner):
    """Radius (in units of the image half-width) of a time ``fraction`` of the way from the earliest time (0) to the
    final time (1): the final time lies on the inner circle, the earliest on the rim."""
    return inner + (1.0 - inner) * (1.0 - np.asarray(fraction))


def radial_layout(
    tree: GalaxyMergerTree,
    inner: float = 0.08,
    compensation: float = 0.0,
    smoothing: float = 0.25,
    transition: float = 0.1,
    **kwargs,
) -> RiverLayout:
    """Lay out a galaxy merger tree for a radial diagram (see :func:`render_radial`).

    This is :func:`river_layout` (whose options it accepts), optionally with ribbon
    widths and gaps scaled by ``(inner/r)**compensation``, where ``r`` is the radius at
    which each time will be drawn.  Bent into a circle, a layout spans the same angle at
    every radius, so ribbons are drawn narrower towards the center, where rings are
    shorter.  With ``compensation`` 0 (the default) the tree fills the circle, as in a
    classic radial tree, and widths are true only in angle; with 1, widths are true
    lengths, but the early tree - narrow compared with the long rings near the rim - fills
    only part of the circle.  ``inner`` is the radius of the final time, in units of the
    image half-width; it must match the ``inner`` passed to :func:`render_radial`.
    ``smoothing`` and ``transition`` (see :func:`river_layout`) default to longer time
    scales than for a river diagram, as sideways shifts of the ribbons are more prominent
    once bent into a circle.
    """
    width_scale = (
        (lambda fraction: (inner / _radius(fraction, inner)) ** compensation)
        if compensation != 0.0
        else None
    )
    return river_layout(
        tree,
        width_scale=width_scale,
        smoothing=smoothing,
        transition=transition,
        **kwargs,
    )


def render_radial(
    tree: GalaxyMergerTree,
    layout: RiverLayout,
    npix: int = 1600,
    inner: float = 0.08,
    background=(0.02, 0.024, 0.04),
    glow: float = 0.35,
    glow_sigma: float = 0.004,
    core: bool = True,
    fill: bool = True,
    angle: float = 360.0,
    **kwargs,
):
    """Render a galaxy merger tree as a radial diagram: the river diagram bent into a circle.

    Time runs inwards: the earliest progenitors lie at the rim and the final galaxy at
    the center, on a circle of radius ``inner`` (in units of the image half-width), which
    is filled with the final galaxy's color when ``core`` is set.  Position across the
    river becomes the angle, spanning ``angle`` degrees (360 for a full circle; less for a
    fan opening upwards).  With ``fill``, the tree is stretched at each radius to span the
    whole angle, as in a classic radial tree: early times, when the tree is narrow, then
    spread around the rim (the stretch is smoothed over time, so ribbons do not wobble).
    Without it, every radius uses the same scale, and the tree keeps the shape of the
    river diagram.  Use a layout from :func:`radial_layout` with the same
    ``inner``, so that ribbon widths are true.  Other keywords are passed to
    :func:`render_river` (``color``, ``norm``, ``cmap``, ``shading``, ...).

    Returns ``(image, mappable)``: an ``(npix, npix, 3)`` RGB float image and a
    ``ScalarMappable`` for a colorbar.
    """
    from scipy.ndimage import map_coordinates

    # The river image, with enough columns to resolve the rim, and no glow (added after bending, so that it stays round).
    columns = int(np.ceil(np.pi * npix))
    river, extent, mappable = render_river(
        tree,
        layout,
        npix=columns,
        background=background,
        glow=0.0,
        margin=0.01,
        **kwargs,
    )
    rows = river.shape[0]
    # The columns spanned by the tree at each time.
    scale = columns / (extent[1] - extent[0])
    lo = np.full(rows, np.inf)
    hi = np.full(rows, -np.inf)
    for b in layout.order:
        ok = np.isfinite(layout.center[b])
        lo[ok] = np.minimum(lo[ok], (layout.center[b] - 0.5 * layout.width[b])[ok])
        hi[ok] = np.maximum(hi[ok], (layout.center[b] + 0.5 * layout.width[b])[ok])
    if fill:
        # Extend to times with no branches, and smooth, so that the stretch changes slowly with radius. The smooth
        # envelope never falls inside the tree's true edges, so no ribbon is cut off at the seam.
        valid = np.isfinite(lo)
        index = np.arange(rows)
        lo = np.interp(index, index[valid], lo[valid])
        hi = np.interp(index, index[valid], hi[valid])
        margin = 0.04 * (hi - lo)
        size = max(int(0.04 * rows), 1)
        lo = -_smooth_envelope(-(lo - margin), size)
        hi = _smooth_envelope(hi + margin, size)
        column_lo = (lo - extent[0]) * scale
        column_span = (hi - lo) * scale
    else:
        column_lo = np.zeros(rows)
        column_span = np.full(rows, float(columns))
    xy = (np.arange(npix) + 0.5) / npix * 2.0 - 1.0
    X, Y = np.meshgrid(xy, -xy)  # image row 0 at the top
    R = np.hypot(X, Y)
    # Angle measured clockwise from the bottom, so that the seam where the two sides of the river meet lies at the bottom
    # (for a fan, the angle is centered on the top).
    span = np.deg2rad(angle)
    phi = np.mod(np.arctan2(-X, -Y), 2.0 * np.pi) - (np.pi - 0.5 * span)
    fraction = 1.0 - (R - inner) / (1.0 - inner)
    row = fraction * (rows - 1)
    row_index = np.clip(np.round(row), 0, rows - 1).astype(int)
    column = column_lo[row_index] + phi / span * column_span[row_index] - 0.5
    image = np.empty((npix, npix, 3))
    for k in range(3):
        image[..., k] = map_coordinates(
            river[..., k], [row, column], order=1, mode="constant", cval=background[k]
        )
    outside = (row < 0.0) | (row > rows - 1) | (phi < 0.0) | (phi > span)
    image[outside] = background
    if core:
        # The final galaxy, filling the inner circle: its color at the final time, brightest at the center.
        branch = tree.branches[tree.root]
        render_color = kwargs.get("color") or _default_color
        value = np.asarray(render_color(branch), dtype=float)
        value = value[np.isfinite(value)]
        rgb = np.asarray(
            mappable.to_rgba(value[-1] if value.size else mappable.norm.vmin)[:3]
        )
        center = R < inner
        light = 1.0 - kwargs.get("shading", 0.45) * (R[center] / inner) ** 2
        image[center] = light[:, None] * rgb[None, :]
    image = _add_glow(image, background, glow, glow_sigma * npix)
    return np.clip(image, 0.0, 1.0), mappable


def draw_river_tree(
    ax, tree: GalaxyMergerTree, layout: RiverLayout, npix: int = 1200, **kwargs
):
    """Render a river diagram (see :func:`render_river`) into the matplotlib Axes ``ax``.

    The vertical axis is in the layout's coordinate (``log10`` of time in Gyr for a
    logarithmic layout).  Returns the ``ScalarMappable`` for a colorbar.
    """
    image, extent, mappable = render_river(tree, layout, npix=npix, **kwargs)
    ax.imshow(
        image, origin="lower", extent=extent, aspect="auto", interpolation="antialiased"
    )
    return mappable


def _is_satellite_subtree(tree: GalaxyMergerTree, b: int) -> bool:
    while b is not None:
        if tree.branches[b].satellite:
            return True
        b = tree.branches[b].parent
    return False
