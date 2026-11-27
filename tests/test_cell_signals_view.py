"""Tests for the Cell Signals tab (stage 5, one cell at a time). They need
Tk and a display (e.g. xvfb-run python -m pytest tests) and are skipped
otherwise. Run with:  python -m pytest tests"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tk = pytest.importorskip("tkinter")
import matplotlib  # noqa: E402

matplotlib.use("TkAgg")

from neclab_core import analysis, events, signals as core_signals  # noqa: E402
from neclab_core.project import Project  # noqa: E402


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    r.withdraw()
    yield r
    r.destroy()


def _signals():
    t = np.arange(300)
    spikes = np.zeros(300)
    spikes[[50, 150, 250]] = 1.0
    transient = np.convolve(spikes, np.exp(-np.arange(15) / 3.0))[:300]
    return pd.DataFrame({
        "cell_2": 100 + 60 * transient + 0.01 * t,
        "cell_5": 200 + 40 * np.roll(transient, 20),
        "cell_9": 150 + 30 * np.roll(transient, 40),
    }, index=pd.RangeIndex(300, name="frame"))


def _record(project, settings):
    raw = _signals()
    result = analysis.Analyzer().run(raw, {**analysis.DEFAULT_SETTINGS, **settings})
    project.set_stage("signals", {"signals": raw, **result.items},
                      params=analysis.stage_params({"cells": [2, 5, 9]}, result))
    return result


def _view(root, project, applied):
    from tkinter import ttk
    from cell_signals_view import CellSignalsView
    notebook = ttk.Notebook(root)
    notebook.pack()

    def apply_settings(settings):
        applied.append(settings)
        _record(project, settings)

    return CellSignalsView(notebook, root, project, apply_settings)


def test_lists_every_cell_with_its_peaks_and_follows_the_project(root):
    project = Project()
    applied = []
    view = _view(root, project, applied)
    assert view.columns == [] and view.listbox.size() == 0     # nothing in stage 5 yet

    result = _record(project, {"peak_method": "Peak Caller"})
    assert view.columns == ["cell_2", "cell_5", "cell_9"]
    peaks = events.events_by_column(result.events_df, view.columns)
    for i, col in enumerate(view.columns):
        assert view.listbox.get(i).startswith(core_signals.signal_label(col))
        assert f"{len(peaks[col])} peak" in view.listbox.get(i)
    assert view.current == "cell_2"
    assert view.peak_var.get() == "Peak Caller"

    # Selecting another cell draws it: raw on top, processed below, with its peaks.
    view.listbox.selection_clear(0, tk.END)
    view.listbox.selection_set(1)
    view._on_list_select()
    assert view.current == "cell_5"
    ax_raw, ax_proc = view.fig.axes
    assert "Cell 5" in ax_raw.get_title() and "Cell 5" in ax_proc.get_title()
    table = view.cell_peaks_table()
    np.testing.assert_array_equal(table["frame"], peaks["cell_5"])

    # A new normalization in the sidebar goes to apply_settings; the cell stays selected.
    view.norm_var.set(view._norm_labels[view._norm_keys.index("lowest_points")])
    view.apply()
    assert applied[-1]["method"] == "lowest_points" and applied[-1]["peak_method"] == "Peak Caller"
    assert view.current == "cell_5" and not view.identity
    assert view._lowest_points("cell_5") is not None
    assert len(view.fig.axes[0].get_lines()) >= 2       # raw signal + its lowest-points baseline

    view._step(1)
    assert view.current == "cell_9"

    project.clear_stage("signals")
    assert view.columns == [] and view.current is None
    view.close()
    assert not view.exists()
