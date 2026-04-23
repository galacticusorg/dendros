"""Trace the history of individual galaxies across Galacticus outputs."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np

from ._collection import _resolve_output_name

if TYPE_CHECKING:
    from ._collection import Collection


_RESERVED_LABELS = frozenset(
    {"time", "expansion_factor", "redshift", "present", "output_names", "ids"}
)

_DUPLICATE_MODES = ("error", "warn", "first")


def _fill_value_for_dtype(dtype: np.dtype, int_sentinel: int):
    """Pick a missing-slot sentinel appropriate to *dtype*."""
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.integer):
        return int_sentinel
    if dtype == np.bool_:
        return False
    raise NotImplementedError(
        f"Dtype {dtype!r} is not supported by trace_galaxy_history. "
        "Only floating, integer, and boolean per-galaxy properties are handled "
        "(strings, complex, and structured dtypes are not yet supported)."
    )


def _format_id_list(ids: np.ndarray, maximum: int = 20) -> str:
    """Render a compact comma-separated list of IDs, truncating if long."""
    if ids.size <= maximum:
        return ", ".join(str(int(x)) for x in ids)
    shown = ", ".join(str(int(x)) for x in ids[:maximum])
    return f"{shown}, ... ({ids.size - maximum} more)"


def trace_galaxy_history(
    collection: "Collection",
    ids,
    properties: Union[List[str], Dict[str, str]],
    outputs: Optional[Sequence[Union[int, str]]] = None,
    *,
    id_dataset: str = "nodeData/nodeUniqueIDBranchTip",
    on_duplicate_file_match: str = "error",
    int_sentinel: int = -1,
) -> Dict[str, np.ndarray]:
    """Extract per-galaxy property histories across Galacticus outputs.

    Galaxies are traced across ``Output*`` groups via an integer branch-tip
    identifier (usually ``nodeUniqueIDBranchTip``) that is constant over time
    for a given object and unique within a single HDF5 file.  For each
    requested property and each chosen output, this function locates every
    requested ID in every file of the collection, assembles the per-galaxy
    slice, and stacks the results along a trailing "output" axis.

    Slots where a galaxy is absent at a given output are filled with
    :data:`numpy.nan` (floating-point properties, and the ``time``, ``redshift``
    and ``expansion_factor`` metadata arrays), with ``int_sentinel`` (integer
    properties), or with ``False`` (boolean properties).  The returned
    ``present`` mask is the canonical indicator of presence/absence and should
    be preferred to sentinel checks.

    Parameters
    ----------
    collection:
        An open :class:`~dendros.Collection`.
    ids:
        Array-like of integer ``nodeUniqueIDBranchTip`` values to trace.
        Coerced to :class:`numpy.ndarray` of ``int64``.  Input order is
        preserved along the first axis of every returned array.
    properties:
        Either a list of relative dataset paths under each ``Output*`` group
        (e.g. ``["nodeData/basicMass"]``), matching
        :meth:`Collection.read`, or a :class:`dict` mapping user-chosen
        labels to relative paths.
    outputs:
        Optional iterable selecting a subset of outputs to include.  Each
        element may be a 1-based integer (e.g. ``3``) or a group name (e.g.
        ``"Output3"``).  A :class:`range` is accepted.  Defaults to all
        outputs in the collection, in temporal order.
    id_dataset:
        Relative path of the tracing ID dataset under each ``Output*``
        group.  Defaults to ``"nodeData/nodeUniqueIDBranchTip"``.
    on_duplicate_file_match:
        What to do if the same ID is found in more than one file at the
        same output (IDs are only unique within a file in multi-file
        collections):

        * ``"error"`` (default) – raise :exc:`ValueError`.
        * ``"warn"`` – emit a :class:`UserWarning` and keep the first
          file's match.
        * ``"first"`` – silently keep the first file's match.
    int_sentinel:
        Missing-slot value used for integer-typed properties.  Defaults to
        ``-1``.

    Returns
    -------
    dict
        Contains:

        * one entry per property – ``numpy.ndarray`` of shape
          ``(n_galaxies,) + per_galaxy_tail + (n_outputs,)``.  A 1-D source
          dataset yields a 2-D ``(n_galaxies, n_outputs)`` array; a 2-D
          dataset of shape ``(N, W)`` yields a 3-D
          ``(n_galaxies, W, n_outputs)`` array; and so on.
        * ``"time"`` – float array ``(n_galaxies, n_outputs)`` of output
          times, NaN where the galaxy is absent.
        * ``"redshift"`` – float array ``(n_galaxies, n_outputs)`` of redshifts,
          NaN where the galaxy is absent.
        * ``"expansion_factor"`` – float array ``(n_galaxies, n_outputs)``
          of expansion factors, NaN where the galaxy is absent.
        * ``"present"`` – bool array ``(n_galaxies, n_outputs)`` that is
          ``True`` exactly where the galaxy was located.
        * ``"output_names"`` – 1-D object array of output group names in
          temporal order.
        * ``"ids"`` – 1-D ``int64`` array of normalized input IDs.

    Raises
    ------
    KeyError
        If ``id_dataset`` is not present in any chosen output of any file
        (e.g. the Galacticus run did not emit ``nodeUniqueIDBranchTip``),
        or if a requested property is missing from a chosen output.
    ValueError
        If ``properties`` contains a reserved label, if ``outputs`` is
        empty, if the tail shape of a property differs between outputs,
        or (by default) if an ID appears in more than one file at the
        same output.
    NotImplementedError
        If a property has a dtype other than integer, floating, or
        boolean.

    Notes
    -----
    A galaxy need not be present at every output (it may have formed
    later or merged earlier); ragged histories are expected.  Requesting
    IDs that are never found anywhere produces a :class:`UserWarning`
    rather than an error, since exploratory workflows often probe IDs of
    uncertain provenance.

    """
    if on_duplicate_file_match not in _DUPLICATE_MODES:
        raise ValueError(
            "on_duplicate_file_match must be one of "
            f"{_DUPLICATE_MODES!r}; got {on_duplicate_file_match!r}"
        )

    ids_arr = np.asarray(ids, dtype=np.int64)
    if ids_arr.ndim != 1:
        ids_arr = ids_arr.reshape(-1)
    n_galaxies = int(ids_arr.size)

    if isinstance(properties, dict):
        properties_map: Dict[str, str] = dict(properties)
    else:
        properties_map = {p: p for p in properties}

    reserved_clash = _RESERVED_LABELS.intersection(properties_map.keys())
    if reserved_clash:
        bad = sorted(reserved_clash)
        raise ValueError(
            f"Reserved label(s) {bad!r} cannot be used in properties; "
            f"reserved labels are {sorted(_RESERVED_LABELS)!r}."
        )

    all_outputs = list(collection.outputs)
    if outputs is None:
        chosen = all_outputs
    else:
        chosen_list = list(outputs)
        if not chosen_list:
            raise ValueError("outputs= is empty; nothing to trace.")
        by_name = {o.name: o for o in all_outputs}
        chosen = []
        for item in chosen_list:
            name = _resolve_output_name(item)
            if name not in by_name:
                raise KeyError(
                    f"Output {name!r} not found in collection "
                    f"(available: {[o.name for o in all_outputs]})."
                )
            chosen.append(by_name[name])

    n_outputs = len(chosen)
    root = collection.output_root
    handles = collection._handles
    files = collection.files

    present = np.zeros((n_galaxies, n_outputs), dtype=bool)
    time = np.full((n_galaxies, n_outputs), np.nan, dtype=float)
    expansion_factor = np.full((n_galaxies, n_outputs), np.nan, dtype=float)

    results: Dict[str, np.ndarray] = {}
    result_tail_shapes: Dict[str, tuple] = {}
    result_first_output: Dict[str, str] = {}
    duplicate_warned_files: set = set()

    for o, meta in enumerate(chosen):
        this_output_matched = np.zeros(n_galaxies, dtype=bool)
        # Per-file info for later property reads: (h_idx, row_in_file, galaxy_index).
        file_hits: List[tuple] = []
        first_file_path: Optional[str] = None

        for h_idx, h in enumerate(handles):
            id_path = f"{root}/{meta.name}/{id_dataset.lstrip('/')}"
            try:
                ids_file = h[id_path][()]
            except KeyError:
                raise KeyError(
                    f"Tracing ID dataset '{id_path}' not found in "
                    f"'{files[h_idx]}'. The Galacticus run may not have been "
                    f"configured to emit '{id_dataset}' — add "
                    "'nodeUniqueIDBranchTip' to the outputs list in your "
                    "Galacticus parameter file."
                ) from None

            ids_file = np.asarray(ids_file, dtype=np.int64)
            if ids_file.size == 0 or n_galaxies == 0:
                continue

            if (
                files[h_idx] not in duplicate_warned_files
                and np.unique(ids_file).size != ids_file.size
            ):
                warnings.warn(
                    f"Duplicate IDs detected in '{id_path}' within "
                    f"'{files[h_idx]}'. Using the leftmost match.",
                    UserWarning,
                    stacklevel=2,
                )
                duplicate_warned_files.add(files[h_idx])

            sort_idx = np.argsort(ids_file, kind="stable")
            sorted_ids = ids_file[sort_idx]
            pos = np.clip(
                np.searchsorted(sorted_ids, ids_arr),
                0,
                sorted_ids.size - 1,
            )
            matched = sorted_ids[pos] == ids_arr
            row_in_file = sort_idx[pos]

            collisions = matched & this_output_matched
            if collisions.any():
                colliding_ids = ids_arr[collisions]
                msg = (
                    f"IDs {_format_id_list(colliding_ids)} appear in both "
                    f"'{first_file_path}' and '{files[h_idx]}' at output "
                    f"'{meta.name}'. nodeUniqueIDBranchTip is only unique "
                    "within a single file; resolve the collision or pass "
                    "on_duplicate_file_match='warn' or 'first'."
                )
                if on_duplicate_file_match == "error":
                    raise ValueError(msg)
                if on_duplicate_file_match == "warn":
                    warnings.warn(msg, UserWarning, stacklevel=2)
                # For "warn" and "first": drop colliding entries from this
                # file's match so the earlier file's values are retained.
                matched = matched & ~this_output_matched

            new_galaxy_index = np.flatnonzero(matched)
            if new_galaxy_index.size == 0:
                continue

            if first_file_path is None:
                first_file_path = files[h_idx]

            file_hits.append((h_idx, row_in_file[matched], new_galaxy_index))
            this_output_matched |= matched

        if this_output_matched.any():
            present[this_output_matched, o] = True
            if meta.time is not None:
                time[this_output_matched, o] = meta.time
            if meta.scale_factor is not None:
                expansion_factor[this_output_matched, o] = meta.scale_factor

        # Read each property once per file, then slice out the hit rows.
        for label, rel in properties_map.items():
            prop_path_template = f"{root}/{meta.name}/{rel.lstrip('/')}"
            # Cache per-file reads for this output.
            file_arrays: Dict[int, np.ndarray] = {}
            for h_idx, rows, gal_idx in file_hits:
                if h_idx not in file_arrays:
                    try:
                        file_arrays[h_idx] = handles[h_idx][prop_path_template][()]
                    except KeyError:
                        raise KeyError(
                            f"Dataset '{prop_path_template}' not found in "
                            f"'{files[h_idx]}'"
                        ) from None
                arr_file = file_arrays[h_idx]

                if label not in results:
                    tail = arr_file.shape[1:]
                    fill = _fill_value_for_dtype(arr_file.dtype, int_sentinel)
                    full_shape = (n_galaxies,) + tail + (n_outputs,)
                    results[label] = np.full(full_shape, fill, dtype=arr_file.dtype)
                    result_tail_shapes[label] = tail
                    result_first_output[label] = meta.name
                else:
                    expected_tail = result_tail_shapes[label]
                    if arr_file.shape[1:] != expected_tail:
                        raise ValueError(
                            f"Property '{rel}' has tail shape "
                            f"{arr_file.shape[1:]!r} at output "
                            f"'{meta.name}' but "
                            f"{expected_tail!r} at output "
                            f"'{result_first_output[label]}'. "
                            "Per-galaxy shape must be consistent across "
                            "the chosen outputs."
                        )

                results[label][gal_idx, ..., o] = arr_file[rows]

    # Emit warnings for IDs never found.
    if n_galaxies > 0:
        never_found = ~present.any(axis=1)
        if never_found.any():
            missing_ids = ids_arr[never_found]
            warnings.warn(
                "Requested IDs never found in any chosen output: "
                f"{_format_id_list(missing_ids)}.",
                UserWarning,
                stacklevel=2,
            )

    # Allocate result arrays for any property that never materialized
    # (because no requested galaxy was ever present).  We fall back to a
    # NaN float array of shape (n_galaxies, n_outputs).
    for label in properties_map:
        if label not in results:
            results[label] = np.full((n_galaxies, n_outputs), np.nan, dtype=float)

    output = dict(results)
    output["time"] = time
    output["expansion_factor"] = expansion_factor
    output["redshift"] = 1.0 / expansion_factor - 1.0
    output["present"] = present
    output["output_names"] = np.array([m.name for m in chosen], dtype=object)
    output["ids"] = ids_arr
    return output
