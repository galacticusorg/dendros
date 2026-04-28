"""Read, modify, and write Galacticus base-parameter XML files.

Implements:

* A Galacticus-style parameter-path resolver supporting ``::`` / ``/``
  separators and ``[N]`` / ``[@value='...']`` selectors.
* :func:`apply_state` — set parameter elements' ``value=`` attributes from a
  state vector.
* :func:`read_parameter_file` / :func:`write_parameter_file_to` — XML round-trip
  using the standard library's ``xml.etree``.

Notes
-----
``xml.etree.ElementTree`` strips comments and reformats whitespace.  If you
need byte-perfect round-trips, use ``lxml`` directly via :mod:`lxml.etree`
(this module's API is intentionally narrow so swapping is easy).  Since the
typical use of these files is "generate, hand to Galacticus", that loss is
acceptable.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union
from xml.etree import ElementTree as ET

import numpy as np

from ._config import ModelParameter


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


_SEGMENT = re.compile(
    r"""
    ^
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)            # element name
    (?:\[                                       # optional predicate
        (?P<predicate>
            \d+                                 # 1-based index
          |
            @value\s*=\s*['"][^'"]*['"]         # attribute filter
        )
    \])?
    $
    """,
    re.VERBOSE,
)

_VALUE_PRED = re.compile(r"""@value\s*=\s*(?P<quote>['"])(?P<val>[^'"]*)(?P=quote)""")


def resolve_parameter_path(root: ET.Element, path: str) -> ET.Element:
    """Locate the XML element identified by a Galacticus parameter *path*.

    Parameters
    ----------
    root:
        Root :class:`xml.etree.ElementTree.Element` to search.
    path:
        Slash- or ``::``-separated parameter path.  Each segment is an
        element name, optionally followed by a ``[N]`` integer (1-based)
        instance selector or a ``[@value='x']`` attribute filter — matching
        the Galacticus parameter-file convention.

    Returns
    -------
    xml.etree.ElementTree.Element

    Raises
    ------
    KeyError
        If any segment does not match an element under the current node.
    ValueError
        If a path segment is malformed.
    """
    segments = [s for s in path.replace("::", "/").strip("/").split("/") if s]
    if not segments:
        raise ValueError(f"Empty parameter path: {path!r}")

    current = root
    walked: List[str] = []
    for seg in segments:
        m = _SEGMENT.match(seg)
        if m is None:
            raise ValueError(f"Malformed parameter-path segment {seg!r} in {path!r}")
        name = m.group("name")
        predicate = m.group("predicate")
        children = list(current.findall(name))
        if not children:
            raise KeyError(
                f"No element {name!r} found under "
                f"{('/' + '/'.join(walked)) if walked else 'root'} "
                f"(path: {path!r})"
            )
        if predicate is None:
            current = children[0]
        elif predicate.isdigit():
            idx = int(predicate)
            if idx < 1 or idx > len(children):
                raise KeyError(
                    f"Index [{idx}] out of range for {name!r} "
                    f"(found {len(children)} matches; path {path!r})"
                )
            current = children[idx - 1]
        else:
            vm = _VALUE_PRED.match(predicate)
            if vm is None:
                raise ValueError(
                    f"Malformed predicate {predicate!r} in path {path!r}"
                )
            wanted = vm.group("val")
            matches = [c for c in children if c.get("value") == wanted]
            if not matches:
                raise KeyError(
                    f"No {name!r} element with value={wanted!r} found "
                    f"(path: {path!r})"
                )
            current = matches[0]
        walked.append(seg)
    return current


# ---------------------------------------------------------------------------
# State application
# ---------------------------------------------------------------------------


def _format_value(x: float) -> str:
    """Format *x* as a string Galacticus will round-trip cleanly."""
    return repr(float(x))


def apply_state(
    tree: ET.ElementTree,
    parameters: Sequence[ModelParameter],
    state: np.ndarray,
    *,
    parameter_map: Optional[Iterable[str]] = None,
) -> None:
    """Set ``value=`` attributes in *tree* from a state vector.

    Parameters
    ----------
    tree:
        Parsed XML tree to modify in place.
    parameters:
        Active model parameters (the same ordering used by chain columns).
    state:
        ``(n_params,)`` state vector aligned with *parameters*.
    parameter_map:
        Optional iterable of parameter names to apply.  ``None`` applies
        every entry of *parameters*; non-``None`` applies only those named
        and is the typical case for an ``independentLikelihoods`` leaf,
        whose base parameter file mentions only that leaf's parameters.

    Raises
    ------
    KeyError
        If a parameter's path does not resolve in *tree*, or if a name in
        *parameter_map* is not among *parameters*.
    ValueError
        If *state* doesn't match the length of *parameters*.
    """
    if state.shape != (len(parameters),):
        raise ValueError(
            f"state has shape {state.shape!r}; expected ({len(parameters)},)"
        )

    if parameter_map is None:
        targets = list(range(len(parameters)))
    else:
        index_by_name = {p.name: i for i, p in enumerate(parameters)}
        targets = []
        for name in parameter_map:
            if name not in index_by_name:
                raise KeyError(
                    f"parameterMap entry {name!r} is not among the active "
                    f"model parameters {list(index_by_name)!r}."
                )
            targets.append(index_by_name[name])

    root = tree.getroot()
    for i in targets:
        param = parameters[i]
        el = resolve_parameter_path(root, param.name)
        el.set("value", _format_value(state[i]))


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def read_parameter_file(path: Union[str, "Path"]) -> ET.ElementTree:
    """Parse a Galacticus parameter XML file.

    Parameters
    ----------
    path:
        Path to the file.

    Returns
    -------
    xml.etree.ElementTree.ElementTree
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Base parameter file not found: {p}")
    return ET.parse(str(p))


def write_parameter_file_to(tree: ET.ElementTree, path: Union[str, "Path"]) -> Path:
    """Write *tree* to *path* with an XML declaration.

    Returns the resolved output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out), encoding="utf-8", xml_declaration=True)
    return out.resolve()


# ---------------------------------------------------------------------------
# Convenience: emit one parameter file per likelihood leaf
# ---------------------------------------------------------------------------


def emit_parameter_files(
    state: np.ndarray,
    config,
    out_dir: Union[str, "Path"],
    *,
    name_format: Optional[str] = None,
) -> List[Tuple[int, Path]]:
    """Write one Galacticus parameter file per leaf of ``config.likelihood``.

    Reads each leaf's ``base_parameters_file``, applies the subset of *state*
    selected by that leaf's ``parameter_map`` (or the full state when no map
    is set), and writes the modified XML into *out_dir*.

    Parameters
    ----------
    state:
        ``(n_params,)`` state vector (in physical / model space, as stored
        in the chain log file — no mapper inversion is applied).
    config:
        :class:`MCMCConfig`.
    out_dir:
        Output directory (created if missing).
    name_format:
        Format string for output filenames; receives ``leaf_index`` and
        ``stem`` (the base file's stem).  Defaults to ``"{stem}.xml"`` for a
        single leaf and ``"{leaf_index:02d}_{stem}.xml"`` for multiple, so
        per-leaf files don't collide when several leaves share a base stem.

    Returns
    -------
    list of (leaf_index, written_path)
        One tuple per leaf, in document order.

    Raises
    ------
    ValueError
        If ``config.likelihood`` is ``None``, or if any leaf lacks a
        ``base_parameters_file``.
    KeyError
        If a parameter's path does not resolve in the corresponding base
        file, or a ``parameter_map`` references an unknown parameter.
    """
    state = np.asarray(state, dtype=float)
    if config.likelihood is None:
        raise ValueError(
            "Config has no <posteriorSampleLikelihood>; cannot emit parameter "
            "files."
        )
    leaves = config.likelihood.leaves()
    if any(leaf.base_parameters_file is None for leaf in leaves):
        missing = [i for i, leaf in enumerate(leaves) if leaf.base_parameters_file is None]
        raise ValueError(
            f"Likelihood leaves {missing!r} have no <baseParametersFileName>; "
            "those leaves do not derive from posteriorSampleLikelihoodBaseParameters."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_resolved = out_dir.resolve()

    if name_format is None:
        name_format = "{stem}.xml" if len(leaves) == 1 else "{leaf_index:02d}_{stem}.xml"

    written: List[Tuple[int, Path]] = []
    for i, leaf in enumerate(leaves):
        tree = read_parameter_file(leaf.base_parameters_file)
        apply_state(
            tree,
            config.parameters,
            state,
            parameter_map=leaf.parameter_map,
        )
        stem = Path(leaf.base_parameters_file).stem
        formatted = name_format.format(leaf_index=i, stem=stem)
        out_path = (out_dir / formatted)
        # Containment check: refuse to write outside out_dir, regardless of
        # whether name_format contained an absolute path or `..` segments.
        try:
            out_path.resolve().relative_to(out_dir_resolved)
        except ValueError:
            raise ValueError(
                f"name_format={name_format!r} produced path "
                f"{out_path} which resolves outside out_dir "
                f"{out_dir_resolved}. Refusing to write."
            ) from None
        write_parameter_file_to(tree, out_path)
        written.append((i, out_path.resolve()))
    return written
