"""Tests for the ``/analyses`` group reader and plotter."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from dendros import open_outputs
from dendros._analyses import _latex_fix, _safe_filename


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
