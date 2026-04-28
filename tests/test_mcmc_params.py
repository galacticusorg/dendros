"""Tests for parameter-path resolution and parameter-file emission."""
from __future__ import annotations

from xml.etree import ElementTree as ET

import numpy as np
import pytest

from dendros import (
    apply_state,
    open_mcmc,
    parse_mcmc_config,
    read_parameter_file,
    resolve_parameter_path,
)
from dendros._mcmc._config import ModelParameter, PriorSpec


# ---------------------------------------------------------------------------
# resolve_parameter_path
# ---------------------------------------------------------------------------


def test_resolve_simple_slash_path(base_parameters_simple):
    tree = read_parameter_file(base_parameters_simple)
    el = resolve_parameter_path(tree.getroot(), "p/a")
    assert el.tag == "a"
    assert el.get("value") == "0.0"


def test_resolve_with_double_colon_separator(base_parameters_simple):
    tree = read_parameter_file(base_parameters_simple)
    el = resolve_parameter_path(tree.getroot(), "p::a")
    assert el.tag == "a"


def test_resolve_value_predicate(base_parameters_selectors):
    tree = read_parameter_file(base_parameters_selectors)
    path = "nodeOperator/nodeOperator[@value='stellarFeedbackDisks']/stellarFeedback/velocityCharacteristic"
    el = resolve_parameter_path(tree.getroot(), path)
    assert el.tag == "velocityCharacteristic"
    el.set("value", "250.0")
    assert el.get("value") == "250.0"


def test_resolve_value_predicate_double_quotes(base_parameters_selectors):
    tree = read_parameter_file(base_parameters_selectors)
    path = 'nodeOperator/nodeOperator[@value="darkMatterProfileShape"]/shape'
    el = resolve_parameter_path(tree.getroot(), path)
    assert el.tag == "shape"


def test_resolve_index_predicate(base_parameters_selectors):
    tree = read_parameter_file(base_parameters_selectors)
    el2 = resolve_parameter_path(tree.getroot(), "multi/slot[2]")
    assert el2.tag == "slot"
    el2.set("value", "marker")
    # Confirm we modified the second slot, not the first or third.
    slots = tree.getroot().findall("multi/slot")
    assert [s.get("value") for s in slots] == ["0.0", "marker", "0.0"]


def test_resolve_unknown_segment_raises(base_parameters_simple):
    tree = read_parameter_file(base_parameters_simple)
    with pytest.raises(KeyError, match="No element"):
        resolve_parameter_path(tree.getroot(), "p/z")


def test_resolve_unknown_value_predicate_raises(base_parameters_selectors):
    tree = read_parameter_file(base_parameters_selectors)
    with pytest.raises(KeyError, match="value="):
        resolve_parameter_path(
            tree.getroot(),
            "nodeOperator/nodeOperator[@value='absent']/shape",
        )


def test_resolve_index_out_of_range_raises(base_parameters_selectors):
    tree = read_parameter_file(base_parameters_selectors)
    with pytest.raises(KeyError, match="out of range"):
        resolve_parameter_path(tree.getroot(), "multi/slot[10]")


def test_resolve_malformed_segment_raises(base_parameters_simple):
    tree = read_parameter_file(base_parameters_simple)
    with pytest.raises(ValueError, match="Malformed"):
        resolve_parameter_path(tree.getroot(), "p/a[bogus]")


def test_resolve_empty_path_raises(base_parameters_simple):
    tree = read_parameter_file(base_parameters_simple)
    with pytest.raises(ValueError, match="Empty"):
        resolve_parameter_path(tree.getroot(), "")


# ---------------------------------------------------------------------------
# apply_state
# ---------------------------------------------------------------------------


def _uniform(name, lo, hi):
    return ModelParameter(
        name=name,
        prior=PriorSpec(kind="uniform", params={"limitLower": lo, "limitUpper": hi}),
        mapper="identity",
    )


def test_apply_state_writes_all_values(base_parameters_simple):
    tree = read_parameter_file(base_parameters_simple)
    parameters = [_uniform("p/a", 0, 1), _uniform("p/b", 0, 1)]
    apply_state(tree, parameters, np.array([0.42, -0.17]))
    a = tree.getroot().find("p/a").get("value")
    b = tree.getroot().find("p/b").get("value")
    assert float(a) == pytest.approx(0.42)
    assert float(b) == pytest.approx(-0.17)


def test_apply_state_with_parameter_map_filters(base_parameters_simple):
    tree = read_parameter_file(base_parameters_simple)
    parameters = [_uniform("p/a", 0, 1), _uniform("p/b", 0, 1)]
    apply_state(
        tree, parameters, np.array([0.42, -0.17]), parameter_map=["p/b"]
    )
    # Only p/b was overwritten.
    assert tree.getroot().find("p/a").get("value") == "0.0"
    assert float(tree.getroot().find("p/b").get("value")) == pytest.approx(-0.17)


def test_apply_state_unknown_parameter_map_raises(base_parameters_simple):
    tree = read_parameter_file(base_parameters_simple)
    parameters = [_uniform("p/a", 0, 1)]
    with pytest.raises(KeyError, match="not among"):
        apply_state(tree, parameters, np.array([0.0]), parameter_map=["nope"])


def test_apply_state_shape_mismatch_raises(base_parameters_simple):
    tree = read_parameter_file(base_parameters_simple)
    parameters = [_uniform("p/a", 0, 1), _uniform("p/b", 0, 1)]
    with pytest.raises(ValueError, match="shape"):
        apply_state(tree, parameters, np.array([0.0]))  # only 1 element


def test_apply_state_round_trip_repr_precision(base_parameters_simple):
    """Values should round-trip through repr() with full double precision."""
    tree = read_parameter_file(base_parameters_simple)
    parameters = [_uniform("p/a", 0, 1), _uniform("p/b", 0, 1)]
    state = np.array([3.141592653589793, -2.718281828459045])
    apply_state(tree, parameters, state)
    a = float(tree.getroot().find("p/a").get("value"))
    b = float(tree.getroot().find("p/b").get("value"))
    assert a == state[0]
    assert b == state[1]


def test_apply_state_with_selectors(base_parameters_selectors):
    tree = read_parameter_file(base_parameters_selectors)
    parameters = [
        _uniform(
            "nodeOperator/nodeOperator[@value='stellarFeedbackDisks']"
            "/stellarFeedback/velocityCharacteristic",
            0,
            1000,
        ),
        _uniform("multi/slot[2]", 0, 1),
    ]
    apply_state(tree, parameters, np.array([250.0, 0.5]))
    el = resolve_parameter_path(
        tree.getroot(),
        "nodeOperator/nodeOperator[@value='stellarFeedbackDisks']/stellarFeedback/velocityCharacteristic",
    )
    assert float(el.get("value")) == pytest.approx(250.0)
    slots = tree.getroot().findall("multi/slot")
    assert [s.get("value") for s in slots] == ["0.0", repr(0.5), "0.0"]


# ---------------------------------------------------------------------------
# MCMCRun.write_parameter_file (single leaf)
# ---------------------------------------------------------------------------


def test_write_parameter_file_round_trip(mcmc_de_run_with_base, tmp_path):
    out_path = tmp_path / "out.xml"
    with open_mcmc(mcmc_de_run_with_base) as run:
        state = np.array([0.123, -0.456])
        written = run.write_parameter_file(state, out_path)
    assert written.is_file()

    tree = ET.parse(str(written))
    a = float(tree.getroot().find("p/a").get("value"))
    b = float(tree.getroot().find("p/b").get("value"))
    assert a == pytest.approx(0.123)
    assert b == pytest.approx(-0.456)


def test_write_parameter_file_creates_parent_dir(mcmc_de_run_with_base, tmp_path):
    out_path = tmp_path / "nested" / "deeper" / "out.xml"
    with open_mcmc(mcmc_de_run_with_base) as run:
        run.write_parameter_file(np.array([0.0, 0.0]), out_path)
    assert out_path.is_file()


def test_write_parameter_file_no_likelihood_raises(tmp_path):
    """A config with no likelihood block can't emit a parameter file."""
    cfg = tmp_path / "nolik.xml"
    cfg.write_text(
        """<?xml version="1.0"?>
<parameters>
  <posteriorSampleSimulation value="differentialEvolution">
    <logFileRoot value="chains"/>
    <modelParameter value="active">
      <name value="x"/>
      <distributionFunction1DPrior value="uniform">
        <limitLower value="0"/><limitUpper value="1"/>
      </distributionFunction1DPrior>
      <operatorUnaryMapper value="identity"/>
      <distributionFunction1DPerturber value="cauchy">
        <median value="0"/><scale value="1e-4"/>
      </distributionFunction1DPerturber>
    </modelParameter>
  </posteriorSampleSimulation>
</parameters>"""
    )
    # Skip chain reading for this test by accessing the run object before it
    # tries to read chains.
    from dendros._mcmc._run import MCMCRun

    run = MCMCRun(parse_mcmc_config(cfg))
    with pytest.raises(ValueError, match="no <posteriorSampleLikelihood>"):
        run.write_parameter_file(np.array([0.5]), tmp_path / "out.xml")


def test_write_parameter_file_index_out_of_range(mcmc_de_run_with_base, tmp_path):
    with open_mcmc(mcmc_de_run_with_base) as run:
        with pytest.raises(IndexError, match="out of range"):
            run.write_parameter_file(
                np.array([0.0, 0.0]),
                tmp_path / "out.xml",
                likelihood_index=5,
            )


def test_write_parameter_file_with_max_posterior_state(
    mcmc_de_run_with_base, tmp_path
):
    """End-to-end: take the max-posterior state and write a parameter file."""
    with open_mcmc(mcmc_de_run_with_base) as run:
        res = run.maximum_posterior()
        out = run.write_parameter_file(res.state, tmp_path / "max_post.xml")
    tree = ET.parse(str(out))
    a = float(tree.getroot().find("p/a").get("value"))
    assert a == pytest.approx(res.state[0])


# ---------------------------------------------------------------------------
# MCMCRun.write_parameter_files (independent likelihoods)
# ---------------------------------------------------------------------------


def test_write_parameter_files_multi_leaf(
    mcmc_independent_run_with_bases, tmp_path
):
    out_dir = tmp_path / "out"
    with open_mcmc(mcmc_independent_run_with_bases) as run:
        state = np.array([0.4, 0.7])
        result = run.write_parameter_files(state, out_dir)

    assert len(result) == 2
    leaf_indices = [r[0] for r in result]
    paths = [r[1] for r in result]
    assert leaf_indices == [0, 1]
    assert all(p.is_file() for p in paths)
    # Default name format for >1 leaves prefixes with the leaf index.
    assert "00_" in paths[0].name
    assert "01_" in paths[1].name

    # Leaf 0 has parameter_map=("alpha",) → only alpha is written; baseA.xml
    # has only an <alpha> element so this test confirms only alpha was set.
    treeA = ET.parse(str(paths[0]))
    assert float(treeA.getroot().find("alpha").get("value")) == pytest.approx(0.4)

    # Leaf 1 has parameter_map=("alpha", "beta") → both are written.
    treeB = ET.parse(str(paths[1]))
    assert float(treeB.getroot().find("alpha").get("value")) == pytest.approx(0.4)
    assert float(treeB.getroot().find("beta").get("value")) == pytest.approx(0.7)


def test_write_parameter_files_default_naming_single_leaf(
    mcmc_de_run_with_base, tmp_path
):
    """For single-leaf configs the default name format omits the leaf index."""
    out_dir = tmp_path / "out"
    with open_mcmc(mcmc_de_run_with_base) as run:
        result = run.write_parameter_files(np.array([0.0, 0.0]), out_dir)
    assert len(result) == 1
    path = result[0][1]
    # base.xml → base.xml in out dir.
    assert path.name == "base.xml"


def test_write_parameter_files_custom_name_format(
    mcmc_independent_run_with_bases, tmp_path
):
    out_dir = tmp_path / "out"
    with open_mcmc(mcmc_independent_run_with_bases) as run:
        result = run.write_parameter_files(
            np.array([0.0, 0.0]),
            out_dir,
            name_format="leaf-{leaf_index}-{stem}.cfg",
        )
    names = sorted(p.name for _, p in result)
    assert names == ["leaf-0-baseA.cfg", "leaf-1-baseB.cfg"]


def test_write_parameter_files_empty_state_raises(
    mcmc_independent_run_with_bases, tmp_path
):
    out_dir = tmp_path / "out"
    with open_mcmc(mcmc_independent_run_with_bases) as run:
        with pytest.raises(ValueError, match="shape"):
            run.write_parameter_files(np.array([]), out_dir)


def test_write_parameter_files_chain_value_passes_through(
    mcmc_de_run_with_base, tmp_path
):
    """A row pulled directly from the chain should write successfully — no
    mapper inversion is applied (chain values are in physical space)."""
    with open_mcmc(mcmc_de_run_with_base) as run:
        chain0 = run.chains[0]
        # Use the third row's state as input.
        result = run.write_parameter_files(chain0.state[2], tmp_path / "out")
    tree = ET.parse(str(result[0][1]))
    a = float(tree.getroot().find("p/a").get("value"))
    b = float(tree.getroot().find("p/b").get("value"))
    assert a == chain0.state[2, 0]
    assert b == chain0.state[2, 1]


def test_write_parameter_files_rejects_parent_traversal(
    mcmc_independent_run_with_bases, tmp_path
):
    """name_format containing `..` segments must not escape out_dir."""
    # The fixture's baseA.xml already lives in tmp_path; capture its contents
    # so we can assert it wasn't overwritten by the escape attempt.
    baseA = tmp_path / "baseA.xml"
    pre = baseA.read_bytes()

    out_dir = tmp_path / "out"
    with open_mcmc(mcmc_independent_run_with_bases) as run:
        with pytest.raises(ValueError, match="outside out_dir"):
            run.write_parameter_files(
                np.array([0.5, 0.7]),
                out_dir,
                name_format="../{stem}.xml",
            )
    # The fixture file was not overwritten.
    assert baseA.read_bytes() == pre


def test_write_parameter_files_rejects_absolute_path(
    mcmc_independent_run_with_bases, tmp_path
):
    """An absolute name_format must not write to that absolute location."""
    out_dir = tmp_path / "out"
    target = tmp_path / "elsewhere.xml"
    with open_mcmc(mcmc_independent_run_with_bases) as run:
        with pytest.raises(ValueError, match="outside out_dir"):
            run.write_parameter_files(
                np.array([0.0, 0.0]),
                out_dir,
                name_format=str(target),
            )
    assert not target.exists()
