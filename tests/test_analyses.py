"""Tests for the ``/analyses`` group reader and plotter."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from dendros import ModelCollection, open_models, open_outputs, plot_analyses
from dendros._analyses import _latex_fix, _safe_filename
from dendros._collection import _default_model_label


# ---------------------------------------------------------------------------
# list_analyses
# ---------------------------------------------------------------------------


def test_list_analyses_returns_table_with_function1d_only(analyses_file):
    with open_outputs(analyses_file) as c:
        table = c.list_analyses(format="astropy")

    names = list(table["name"])
    assert "simpleSMF" in names
    assert "withTarget" in names
    assert "withCov" in names
    assert "step1:chain1/inner" in names
    assert "notFunction1D" not in names

    row = table[list(table["name"]).index("withTarget")]
    assert row["hasTarget"]
    assert row["xAxisIsLog"] in (False, np.False_, 0)


def test_list_analyses_missing_group_raises(single_file):
    with open_outputs(single_file) as c:
        with pytest.raises(KeyError, match="No '/analyses' group"):
            c.list_analyses()


# ---------------------------------------------------------------------------
# plot_analyses
# ---------------------------------------------------------------------------


def test_plot_analyses_returns_figures(analyses_file):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    with open_outputs(analyses_file) as c:
        figs = c.plot_analyses()

    assert set(figs.keys()) == {
        "simpleSMF", "withCov", "withTarget", "step1:chain1/inner",
    }
    for fig in figs.values():
        assert isinstance(fig, Figure)


def test_plot_analyses_axis_scales_and_target(analyses_file):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    with open_outputs(analyses_file) as c:
        figs = c.plot_analyses()

    smf = figs["simpleSMF"].axes[0]
    assert smf.get_xscale() == "log"
    assert smf.get_yscale() == "log"

    wt = figs["withTarget"].axes[0]
    assert wt.get_xscale() == "linear"
    # Two errorbar containers: model + target.
    assert len(wt.containers) == 2
    labels = [t.get_text() for t in wt.get_legend().get_texts()]
    assert "Model" in labels
    assert "Foo+24" in labels


def test_plot_analyses_specific_name_string(analyses_file):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    with open_outputs(analyses_file) as c:
        figs = c.plot_analyses(name="simpleSMF")
    assert list(figs.keys()) == ["simpleSMF"]


def test_plot_analyses_specific_name_list(analyses_file):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    with open_outputs(analyses_file) as c:
        figs = c.plot_analyses(name=["simpleSMF", "withTarget"])
    assert set(figs.keys()) == {"simpleSMF", "withTarget"}


def test_plot_analyses_unknown_name_raises(analyses_file):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    with open_outputs(analyses_file) as c:
        with pytest.raises(KeyError, match="not found"):
            c.plot_analyses(name="bogus")


def test_plot_analyses_writes_files(analyses_file, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    out = tmp_path / "figs"
    with open_outputs(analyses_file) as c:
        figs = c.plot_analyses(output_directory=out, file_format="pdf")

    assert out.is_dir()
    written = sorted(p.name for p in out.glob("*.pdf"))
    assert "simpleSMF.pdf" in written
    assert "withTarget.pdf" in written
    # 'step1:chain1/inner' must be sanitized: both ':' and '/' become '_',
    # then runs of '_' are collapsed.
    assert "step1_chain1_inner.pdf" in written
    assert len(written) == len(figs)


def test_plot_uses_covariance_when_no_asymmetric(analyses_file):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    with open_outputs(analyses_file) as c:
        figs = c.plot_analyses(name="withCov")
    ax = figs["withCov"].axes[0]
    # One model errorbar container, no target.
    assert len(ax.containers) == 1
    container = ax.containers[0]
    # errorbar containers have a 'has_yerr' attribute set when yerr was given.
    assert getattr(container, "has_yerr", False)


def test_plot_recurses_nested(analyses_file):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    with open_outputs(analyses_file) as c:
        figs = c.plot_analyses(name="step1:chain1/inner")
    assert "step1:chain1/inner" in figs


def test_plot_uses_primary_only_for_mpi(analyses_mpi_files):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    with open_outputs(analyses_mpi_files[0]) as c:
        # Auto-discovers both MPI files; only file 0 has /analyses, but the
        # primary file is file 0 so this should still work.
        assert len(c.files) == 2
        figs = c.plot_analyses()
    assert "simple" in figs


def test_plot_show_target_false(analyses_file):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    with open_outputs(analyses_file) as c:
        figs = c.plot_analyses(name="withTarget", show_target=False)
    ax = figs["withTarget"].axes[0]
    assert len(ax.containers) == 1  # only model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_latex_fix_substitutions():
    assert _latex_fix(r"\hbox{Mpc}") == r"\mathrm{Mpc}"
    assert _latex_fix(r"x \le y") == r"x \leq y"
    assert _latex_fix(r"x \ge y") == r"x \geq y"
    # \le inside a longer command name is preserved.
    assert _latex_fix(r"\left(") == r"\left("
    assert _latex_fix("") == ""


def test_safe_filename_replaces_unsafe_chars():
    # POSIX path separator and Windows-invalid characters all collapse.
    assert _safe_filename("a/b") == "a_b"
    assert _safe_filename("step1:chain1/inner") == "step1_chain1_inner"
    assert _safe_filename(r'na<m>e:"/\|?*') == "na_m_e_"
    # Backslash specifically (Windows path separator).
    assert _safe_filename("foo\\bar") == "foo_bar"
    # ASCII control codes are stripped.
    assert _safe_filename("a\x00b\tc") == "a_b_c"
    # Trailing whitespace and dots (Windows silently strips these).
    assert _safe_filename("name . ") == "name"
    # Plain names pass through untouched.
    assert _safe_filename("simpleSMF") == "simpleSMF"
    # All-unsafe input doesn't yield an empty filename.
    assert _safe_filename("///") == "_"


def _write_minimal_analysis(path, *, x, y, y_target=None):
    """Helper for shape-mismatch tests: build a one-analysis HDF5 file."""
    with h5py.File(path, "w") as f:
        f.attrs["statusCompletion"] = 0
        f.create_group("Outputs/Output1").attrs["outputTime"] = 13.8
        a = f.create_group("analyses").create_group("a")
        a.attrs["type"] = np.bytes_("function1D")
        a.create_dataset("x", data=np.asarray(x, dtype=float))
        a.attrs["xDataset"] = np.bytes_("x")
        a.create_dataset("y", data=np.asarray(y, dtype=float))
        a.attrs["yDataset"] = np.bytes_("y")
        if y_target is not None:
            a.create_dataset("yt", data=np.asarray(y_target, dtype=float))
            a.attrs["yDatasetTarget"] = np.bytes_("yt")


def test_x_y_shape_mismatch_raises(tmp_path):
    p = tmp_path / "mismatched.hdf5"
    _write_minimal_analysis(p, x=[1.0, 2.0, 3.0], y=[10.0, 20.0])
    with open_outputs(str(p)) as c:
        with pytest.raises(ValueError, match="yDataset shape .* does not match"):
            c.plot_analyses()


def test_x_must_be_1d(tmp_path):
    p = tmp_path / "x2d.hdf5"
    _write_minimal_analysis(
        p, x=[[1.0, 2.0], [3.0, 4.0]], y=[[10.0, 20.0], [30.0, 40.0]]
    )
    with open_outputs(str(p)) as c:
        with pytest.raises(ValueError, match="xDataset must be 1D"):
            c.plot_analyses()


def test_target_shape_mismatch_raises(tmp_path):
    p = tmp_path / "target_mismatch.hdf5"
    _write_minimal_analysis(
        p, x=[1.0, 2.0, 3.0], y=[10.0, 20.0, 30.0], y_target=[1.0, 2.0]
    )
    with open_outputs(str(p)) as c:
        with pytest.raises(ValueError, match="yDatasetTarget shape"):
            c.plot_analyses()


def test_attr_pointing_at_subgroup_raises(tmp_path):
    """If e.g. yDataset points at a subgroup instead of a dataset, surface a
    clear TypeError rather than h5py's confusing 'Group' object indexing
    error."""
    p = tmp_path / "malformed.hdf5"
    with h5py.File(p, "w") as f:
        f.attrs["statusCompletion"] = 0
        f.create_group("Outputs/Output1").attrs["outputTime"] = 13.8
        a = f.create_group("analyses").create_group("bad")
        a.attrs["type"] = np.bytes_("function1D")
        a.create_dataset("x", data=np.array([1.0, 2.0]))
        a.attrs["xDataset"] = np.bytes_("x")
        # yDataset points at a subgroup, not a dataset.
        a.create_group("y_is_a_group")
        a.attrs["yDataset"] = np.bytes_("y_is_a_group")

    with open_outputs(str(p)) as c:
        with pytest.raises(TypeError, match="not an h5py.Dataset"):
            c.plot_analyses()


# ---------------------------------------------------------------------------
# Multi-model plotting
# ---------------------------------------------------------------------------


def test_default_model_label_strips_mpi_suffix():
    assert _default_model_label("/runs/fiducial.hdf5") == "fiducial"
    assert _default_model_label("/runs/fid:MPI0000.hdf5") == "fid"
    assert _default_model_label("/runs/fid:MPI0007.hdf5") == "fid"
    # Not an MPI suffix — leave alone.
    assert _default_model_label("/runs/fid:MPI007.hdf5") == "fid:MPI007"


def test_plot_analyses_dict_overlays_models(analyses_two_models):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    a, b = analyses_two_models
    with open_outputs(a) as ca, open_outputs(b) as cb:
        figs = plot_analyses({"Fid": ca, "Var": cb}, name="shared")

    ax = figs["shared"].axes[0]
    # Two model curves + one target overlay = 3 errorbar containers.
    assert len(ax.containers) == 3
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Fid" in legend_labels
    assert "Var" in legend_labels
    assert "Obs+24" in legend_labels
    # Model curves should use different colours.
    line_a = ax.containers[0].lines[0]
    line_b = ax.containers[1].lines[0]
    assert tuple(line_a.get_color()) != tuple(line_b.get_color())


def test_plot_analyses_target_plotted_once(analyses_two_models):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    a, b = analyses_two_models
    with open_outputs(a) as ca, open_outputs(b) as cb:
        figs = plot_analyses([ca, cb], name="shared")

    ax = figs["shared"].axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    # Exactly one target entry, even though both models have it.
    assert legend_labels.count("Obs+24") == 1


def test_plot_analyses_list_default_labels_from_filenames(analyses_two_models):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    a, b = analyses_two_models
    with open_outputs(a) as ca, open_outputs(b) as cb:
        figs = plot_analyses([ca, cb], name="shared")

    legend_labels = [t.get_text() for t in figs["shared"].axes[0].get_legend().get_texts()]
    assert "fiducial" in legend_labels
    assert "variant" in legend_labels


def test_plot_analyses_explicit_labels_override(analyses_two_models):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    a, b = analyses_two_models
    with open_outputs(a) as ca, open_outputs(b) as cb:
        figs = plot_analyses([ca, cb], name="shared", labels=["A", "B"])

    legend_labels = [t.get_text() for t in figs["shared"].axes[0].get_legend().get_texts()]
    assert "A" in legend_labels
    assert "B" in legend_labels


def test_plot_analyses_union_skips_missing_per_model(analyses_two_models):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    a, b = analyses_two_models
    with open_outputs(a) as ca, open_outputs(b) as cb:
        figs = plot_analyses({"Fid": ca, "Var": cb})

    # Union: shared, alsoShared, onlyB.
    assert set(figs.keys()) == {"shared", "alsoShared", "onlyB"}
    # onlyB exists only in Var; legend should not list Fid.
    ax = figs["onlyB"].axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Var" in legend_labels
    assert "Fid" not in legend_labels


def test_plot_analyses_labels_with_dict_raises(analyses_two_models):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    a, b = analyses_two_models
    with open_outputs(a) as ca, open_outputs(b) as cb:
        with pytest.raises(ValueError, match="labels="):
            plot_analyses({"A": ca, "B": cb}, labels=["X", "Y"])


def test_plot_analyses_labels_with_single_collection_raises(analyses_file):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    with open_outputs(analyses_file) as c:
        with pytest.raises(ValueError, match="labels="):
            plot_analyses(c, labels=["X"])


def test_plot_analyses_label_count_mismatch_raises(analyses_two_models):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    a, b = analyses_two_models
    with open_outputs(a) as ca, open_outputs(b) as cb:
        with pytest.raises(ValueError, match="length"):
            plot_analyses([ca, cb], labels=["only-one"])


def test_plot_analyses_duplicate_default_labels_raises(tmp_path):
    """List form with two files that produce the same stem-derived label."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    d1 = tmp_path / "run1"
    d2 = tmp_path / "run2"
    d1.mkdir()
    d2.mkdir()
    for d in (d1, d2):
        p = d / "out.hdf5"
        with h5py.File(p, "w") as f:
            f.attrs["statusCompletion"] = 0
            o = f.create_group("Outputs/Output1")
            o.attrs["outputTime"] = 13.8
            o.create_group("nodeData")
            a = f.create_group("analyses").create_group("x")
            a.attrs["type"] = np.bytes_("function1D")
            a.create_dataset("x", data=np.array([1.0, 2.0]))
            a.attrs["xDataset"] = np.bytes_("x")
            a.create_dataset("y", data=np.array([3.0, 4.0]))
            a.attrs["yDataset"] = np.bytes_("y")

    with open_outputs(str(d1 / "out.hdf5")) as c1, open_outputs(str(d2 / "out.hdf5")) as c2:
        with pytest.raises(ValueError, match="collide"):
            plot_analyses([c1, c2])


def test_plot_analyses_mpi_collection_acts_as_one_model(analyses_mpi_files, analyses_file):
    """An MPI-split Collection should plot a single curve, not one per rank."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    # Pair the MPI collection with a regular one so we're in multi-model mode.
    with open_outputs(analyses_mpi_files[0]) as cmpi, open_outputs(analyses_file) as c:
        assert len(cmpi.files) == 2  # both MPI peers were detected
        figs = plot_analyses({"mpi-run": cmpi, "ref": c})

    # cmpi only has "simple"; c has the others.  In the "simple" figure
    # only mpi-run contributes (one curve, no target on this analysis).
    ax = figs["simple"].axes[0]
    assert len(ax.containers) == 1
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["mpi-run"]


def test_open_models_dict_returns_model_collection(analyses_two_models):
    a, b = analyses_two_models
    with open_models({"Fid": a, "Var": b}) as m:
        assert isinstance(m, ModelCollection)
        assert set(m.keys()) == {"Fid", "Var"}
        assert "shared" in m["Fid"].list_analyses()["name"]


def test_open_models_list_default_labels(analyses_two_models):
    a, b = analyses_two_models
    with open_models([a, b]) as m:
        assert set(m.keys()) == {"fiducial", "variant"}


def test_open_models_duplicate_default_labels_raises(tmp_path):
    d1 = tmp_path / "r1"
    d2 = tmp_path / "r2"
    d1.mkdir()
    d2.mkdir()
    for d in (d1, d2):
        p = d / "g.hdf5"
        with h5py.File(p, "w") as f:
            f.attrs["statusCompletion"] = 0
            o = f.create_group("Outputs/Output1")
            o.attrs["outputTime"] = 13.8
            o.create_group("nodeData")
    with pytest.raises(ValueError, match="Duplicate default label"):
        open_models([str(d1 / "g.hdf5"), str(d2 / "g.hdf5")])


def test_open_models_context_manager_closes_collections(analyses_two_models):
    a, b = analyses_two_models
    with open_models({"A": a, "B": b}) as m:
        ca = m["A"]
        assert ca._handles
    # After context exit, both should be closed.
    assert not ca._handles


def test_plot_analyses_accepts_open_models_result(analyses_two_models):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    a, b = analyses_two_models
    with open_models({"Fid": a, "Var": b}) as m:
        figs = plot_analyses(m, name="shared")
    legend_labels = [t.get_text() for t in figs["shared"].axes[0].get_legend().get_texts()]
    assert "Fid" in legend_labels
    assert "Var" in legend_labels


def test_plot_missing_matplotlib_raises(analyses_file, monkeypatch):
    real_mpl = sys.modules.get("matplotlib")
    real_pyplot = sys.modules.get("matplotlib.pyplot")
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    try:
        with open_outputs(analyses_file) as c:
            with pytest.raises(ImportError, match="dendros\\[plot\\]"):
                c.plot_analyses()
    finally:
        if real_mpl is not None:
            sys.modules["matplotlib"] = real_mpl
        if real_pyplot is not None:
            sys.modules["matplotlib.pyplot"] = real_pyplot
