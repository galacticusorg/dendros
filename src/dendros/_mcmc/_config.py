"""Parse a Galacticus MCMC posterior-sample configuration file."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorSpec:
    """Prior on a single model parameter.

    Attributes
    ----------
    kind:
        Value of the ``distributionFunction1DPrior`` element, e.g. ``"uniform"``
        or ``"normal"``.
    params:
        Mapping of sub-element name to its parsed ``value`` (as a float when
        possible, else the raw string).  For ``uniform`` priors the keys are
        ``"limitLower"`` and ``"limitUpper"``; for (truncated) ``normal`` they
        are ``"mean"``, ``"variance"``, and optionally ``"limitLower"`` /
        ``"limitUpper"``.
    """

    kind: str
    params: dict


@dataclass(frozen=True)
class PerturberSpec:
    """Perturber on a single model parameter.

    Attributes
    ----------
    kind:
        Value of the ``distributionFunction1DPerturber`` element.
    params:
        Mapping of sub-element name to its parsed ``value``.
    """

    kind: str
    params: dict


@dataclass(frozen=True)
class ModelParameter:
    """A single ``<modelParameter value="active">`` entry from the config.

    Attributes
    ----------
    name:
        Galacticus parameter path, e.g. ``"haloMassFunctionParameters/a"``.
    label:
        Optional LaTeX label for plotting.  ``None`` when the config omits the
        ``<label>`` sub-element.  Use :attr:`display_label` to obtain a
        plottable string regardless.
    prior:
        Parsed ``distributionFunction1DPrior`` block, if present.
    mapper:
        Value of ``operatorUnaryMapper``; defaults to ``"identity"``.
    perturber:
        Parsed ``distributionFunction1DPerturber`` block, if present.
    """

    name: str
    label: Optional[str] = None
    prior: Optional[PriorSpec] = None
    mapper: str = "identity"
    perturber: Optional[PerturberSpec] = None

    @property
    def display_label(self) -> str:
        """A plottable label: :attr:`label` if set, else the trailing component of :attr:`name`."""
        if self.label:
            return self.label
        return self.name.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Likelihood:
    """A node in the ``posteriorSampleLikelihood`` tree.

    Attributes
    ----------
    kind:
        Value attribute of the ``posteriorSampleLikelihood`` element.
    base_parameters_file:
        Resolved path to the ``baseParametersFileName`` element's value when
        present.  ``None`` for non-leaf nodes (e.g. ``independentLikelihoods``
        without a base file of its own).
    parameter_map:
        For children of ``posteriorSampleLikelihoodIndependentLikelihoods``,
        the parsed ``<parameterMap value="space separated names"/>`` for *this*
        child.  Each entry is a parameter name from the active model
        parameters.  ``None`` outside of an ``independentLikelihoods`` context,
        in which case identity mapping (all active parameters) is implied.
    children:
        Tuple of child :class:`Likelihood` instances.  Empty for leaves.
    """

    kind: str
    base_parameters_file: Optional[Path] = None
    parameter_map: Optional[Tuple[str, ...]] = None
    children: Tuple["Likelihood", ...] = field(default_factory=tuple)

    def leaves(self) -> Tuple["Likelihood", ...]:
        """Flatten the tree to its leaf likelihoods (in document order)."""
        if not self.children:
            return (self,)
        out: list = []
        for c in self.children:
            out.extend(c.leaves())
        return tuple(out)


@dataclass(frozen=True)
class MCMCConfig:
    """Parsed Galacticus MCMC configuration.

    Attributes
    ----------
    config_path:
        Absolute path to the parsed XML file.
    log_file_root:
        Resolved chain log-file root (relative paths resolved against
        :attr:`config_path`'s directory).  Per-rank chain files are at
        ``f"{log_file_root}_{rank:04d}.log"``.
    simulation_kind:
        Value attribute of ``posteriorSampleSimulation``, e.g.
        ``"differentialEvolution"`` or ``"particleSwarm"``.  Determines whether
        chain rows carry trailing per-particle velocity columns.
    parameters:
        Tuple of active :class:`ModelParameter` entries in document order.
        This is the canonical ordering used by chain-file columns.
    likelihood:
        Root of the ``posteriorSampleLikelihood`` tree, or ``None`` if the
        config lacks a likelihood block.
    """

    config_path: Path
    log_file_root: Path
    simulation_kind: str
    parameters: Tuple[ModelParameter, ...]
    likelihood: Optional[Likelihood]

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """Tuple of parameter ``name`` strings, in column order."""
        return tuple(p.name for p in self.parameters)

    def state_indices_for(self, leaf: Likelihood) -> Tuple[int, ...]:
        """Indices of the global state vector applicable to *leaf*.

        For a leaf inside an ``independentLikelihoods`` subtree, returns the
        positions in :attr:`parameters` named in the leaf's ``parameter_map``.
        For other leaves, returns ``(0, 1, ..., n_params - 1)`` (identity).

        Parameters
        ----------
        leaf:
            A :class:`Likelihood` from :meth:`Likelihood.leaves`.

        Raises
        ------
        KeyError
            If a name in ``parameter_map`` isn't among the active parameters.
        """
        if leaf.parameter_map is None:
            return tuple(range(len(self.parameters)))
        index_by_name = {p.name: i for i, p in enumerate(self.parameters)}
        out = []
        for name in leaf.parameter_map:
            if name not in index_by_name:
                raise KeyError(
                    f"parameterMap entry {name!r} is not among the active "
                    f"model parameters {list(index_by_name)!r}."
                )
            out.append(index_by_name[name])
        return tuple(out)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_mcmc_config(path: Union[str, "Path"]) -> MCMCConfig:
    """Parse a Galacticus MCMC ``<parameters>`` XML file.

    Parameters
    ----------
    path:
        Path to the MCMC configuration XML.

    Returns
    -------
    MCMCConfig

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file's root element is not ``<parameters>``, or if the required
        ``posteriorSampleSimulation`` / ``logFileRoot`` elements are missing.
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"MCMC config file not found: {config_path}")

    tree = ET.parse(str(config_path))
    root = tree.getroot()
    if root.tag != "parameters":
        raise ValueError(
            f"Expected root element <parameters> in {config_path}, got <{root.tag}>"
        )

    sim = root.find("posteriorSampleSimulation")
    if sim is None:
        raise ValueError(
            f"No <posteriorSampleSimulation> element found in {config_path}"
        )
    simulation_kind = sim.get("value", "")

    log_root_el = sim.find("logFileRoot")
    if log_root_el is None or log_root_el.get("value") is None:
        raise ValueError(
            f"<posteriorSampleSimulation> in {config_path} has no "
            "<logFileRoot value=\"...\"/> child"
        )
    log_file_root = _resolve_path(log_root_el.get("value"), config_path.parent)

    parameters = tuple(
        _parse_model_parameter(el) for el in sim.findall("modelParameter")
        if el.get("value") == "active"
    )

    lik_el = root.find("posteriorSampleLikelihood")
    likelihood = (
        _parse_likelihood(lik_el, config_path.parent)
        if lik_el is not None
        else None
    )

    return MCMCConfig(
        config_path=config_path,
        log_file_root=log_file_root,
        simulation_kind=simulation_kind,
        parameters=parameters,
        likelihood=likelihood,
    )


# ---------------------------------------------------------------------------
# Element parsers
# ---------------------------------------------------------------------------


def _parse_model_parameter(el: ET.Element) -> ModelParameter:
    name = _value_of(el, "name")
    if name is None:
        raise ValueError(
            "<modelParameter> entry has no <name value=\"...\"/> child"
        )

    label = _value_of(el, "label")

    prior_el = el.find("distributionFunction1DPrior")
    prior = _parse_distribution(prior_el) if prior_el is not None else None

    perturber_el = el.find("distributionFunction1DPerturber")
    perturber_kind_params = (
        _parse_distribution(perturber_el) if perturber_el is not None else None
    )
    perturber = (
        PerturberSpec(kind=perturber_kind_params.kind, params=perturber_kind_params.params)
        if perturber_kind_params is not None
        else None
    )

    mapper = _value_of(el, "operatorUnaryMapper") or "identity"

    return ModelParameter(
        name=name,
        label=label,
        prior=prior,
        mapper=mapper,
        perturber=perturber,
    )


def _parse_distribution(el: ET.Element) -> PriorSpec:
    """Parse a ``distributionFunction1D{Prior,Perturber}`` block into a PriorSpec."""
    kind = el.get("value", "")
    params: dict = {}
    for child in el:
        v = child.get("value")
        if v is None:
            continue
        params[child.tag] = _maybe_float(v)
    return PriorSpec(kind=kind, params=params)


def _parse_likelihood(el: ET.Element, base_dir: Path) -> Likelihood:
    """Recursively parse a ``posteriorSampleLikelihood`` element into a Likelihood tree."""
    kind = el.get("value", "")

    base_file_raw = _value_of(el, "baseParametersFileName")
    base_file = _resolve_path(base_file_raw, base_dir) if base_file_raw else None

    # Children: posteriorSampleLikelihood entries paired with their preceding
    # parameterMap sibling, if any (Galacticus convention is one parameterMap
    # per child likelihood, in document order).
    children: list = []
    pending_map: Optional[Tuple[str, ...]] = None
    for child in el:
        if child.tag == "parameterMap":
            v = child.get("value", "")
            pending_map = tuple(v.split())
        elif child.tag == "posteriorSampleLikelihood":
            sub = _parse_likelihood(child, base_dir)
            if pending_map is not None:
                sub = Likelihood(
                    kind=sub.kind,
                    base_parameters_file=sub.base_parameters_file,
                    parameter_map=pending_map,
                    children=sub.children,
                )
                pending_map = None
            children.append(sub)

    return Likelihood(
        kind=kind,
        base_parameters_file=base_file,
        parameter_map=None,
        children=tuple(children),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _value_of(parent: ET.Element, tag: str) -> Optional[str]:
    """Return ``parent/<tag value="...">`` or ``None`` if absent."""
    child = parent.find(tag)
    if child is None:
        return None
    return child.get("value")


def _maybe_float(s: str):
    """Convert *s* to ``float`` if possible; otherwise return the stripped string."""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return s


def _resolve_path(raw: str, base_dir: Path) -> Path:
    """Resolve *raw* against *base_dir* if relative, leaving absolute paths alone."""
    p = Path(raw)
    if not p.is_absolute():
        p = base_dir / p
    return p
