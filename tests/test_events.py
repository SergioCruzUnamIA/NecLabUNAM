"""Tests for event (peak) detection, stage 6 events. Run with:  python -m pytest tests"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neclab_core import events, stages  # noqa: E402
from neclab_core.project import Project  # noqa: E402


def _signals():
    t = np.arange(300)
    spikes = np.zeros(300)
    spikes[[50, 150, 250]] = 1.0
    kernel = np.exp(-np.arange(15) / 3.0)
    transient = np.convolve(spikes, kernel)[:300]
    return pd.DataFrame({
        "cell_2": 100 + 60 * transient + 0.01 * t,
        "cell_5": 200 + 40 * np.roll(transient, 20),
        "cell_9": np.full(300, 50.0),
    }, index=pd.RangeIndex(300, name="frame"))


def test_peak_caller_finds_the_transients():
    ev = events.detect_events(_signals(), "Peak Caller")
    assert list(ev.columns) == ["cell_id", "frame", "value"]
    assert ev[ev.cell_id == 2].frame.tolist() == [50, 150, 250]
    assert ev[ev.cell_id == 5].frame.tolist() == [70, 170, 270]
    assert (ev.cell_id == 9).sum() == 0           # a flat signal has no peaks
    np.testing.assert_allclose(ev.value, [_signals()[f"cell_{c}"][f] for c, f in zip(ev.cell_id, ev.frame)])


def test_events_sorted_by_cell_then_frame_and_params_used():
    ev = events.detect_events(_signals(), "Peak Caller")
    assert ev.equals(ev.sort_values(["cell_id", "frame"]).reset_index(drop=True))
    # A 60% rise is more than the cell_5 transients (40 on a baseline of 200)
    strict = events.detect_events(_signals(), "Peak Caller", {"rise_percent": 60})
    assert (strict.cell_id == 5).sum() == 0


@pytest.mark.parametrize("method", list(events.PEAK_METHODS))
def test_every_method_runs_and_matches_compute_peaks(method):
    df = _signals()
    ev = events.detect_events(df, method)
    params = events.default_params(method)
    assert (ev.cell_id == 9).sum() == 0           # flat: skipped (Elliptic Envelope can't fit it)
    for col in ["cell_2", "cell_5"]:
        expected = sorted(events.compute_peaks(df[col].to_numpy().reshape(-1, 1), 0, method, params))
        assert ev[ev.cell_id == int(col.split("_")[1])].frame.tolist() == expected


def test_peak_functions_reuses_the_shared_detection():
    peak_functions = pytest.importorskip("peak_functions")
    assert peak_functions.compute_peaks is events.compute_peaks


def test_unknown_method_and_empty_table():
    with pytest.raises(ValueError):
        events.detect_events(_signals(), "Nope")
    ev = events.detect_events(_signals()[["cell_9"]], "Peak Caller")
    assert ev.empty and list(ev.columns) == ["cell_id", "frame", "value"]


def test_events_by_column():
    ev = events.detect_events(_signals(), "Peak Caller")
    by_col = events.events_by_column(ev, ["cell_2", "cell_9"])
    assert by_col["cell_2"].tolist() == [50, 150, 250] and by_col["cell_9"].tolist() == []


def test_events_save_and_load_as_stage_6(tmp_path):
    ev = events.detect_events(_signals(), "Peak Caller")
    stages.save_item("events", ev, tmp_path / "events.csv")
    loaded = stages.load_item("events", tmp_path / "events.csv",
                              stages.Context(labels=np.array([[0, 2], [5, 9]])))
    assert loaded.data[["cell_id", "frame"]].equals(ev[["cell_id", "frame"]])

    p = Project()
    p.set_stage("analysis", {"events": ev}, params={"method": "none", "peak_method": "Peak Caller",
                                                     "peak_params": events.default_params("Peak Caller")})
    p.save(tmp_path / "x.neclab")
    q, warnings = Project.load(tmp_path / "x.neclab")
    assert not warnings
    assert q.get("analysis").items["events"][["cell_id", "frame"]].equals(ev[["cell_id", "frame"]])
    assert q.get("analysis").params["peak_params"]["rise_percent"] == 5
