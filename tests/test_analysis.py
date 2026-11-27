"""Tests for the normalization + events of a set of signals (stage 5). Run with:  python -m pytest tests"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neclab_core import analysis, events, normalization  # noqa: E402
from neclab_core.project import Project  # noqa: E402


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


def _settings(**kw):
    return {**analysis.DEFAULT_SETTINGS, **kw}


def test_defaults_pass_the_raw_signals_through_and_are_recorded():
    df = _signals()
    result = analysis.Analyzer().run(df, analysis.DEFAULT_SETTINGS)
    assert result.identity and result.description == "raw"
    pd.testing.assert_frame_equal(result.norm_df, df)
    assert result.events_df is None
    # The stage is recorded even with no normalization, so it is usable.
    assert set(result.items) == {"normalized_signals"}
    assert result.params["method"] == "none" and "peak_method" not in result.params
    p = Project()
    p.set_stage("signals", {"signals": df, **result.items}, params=result.params)
    assert p.status("signals") == "done"


def test_normalization_matches_normalize_signals():
    df = _signals()
    result = analysis.Analyzer().run(df, _settings(method="delta_f_over_f0", smooth=True, lowest_points=6))
    expected = normalization.normalize_signals(df, "delta_f_over_f0", 6, 6)
    pd.testing.assert_frame_equal(result.norm_df, expected)
    assert not result.identity
    assert result.params["smoothing_points"] == 6
    assert set(result.lowest_points_used) == set(df.columns)


def test_smoothing_is_ignored_when_subtracting():
    s = _settings(method="none", smooth=True, subtract=True)
    assert analysis.smoothing_points(s) is None
    assert analysis.smoothing_points(_settings(method="lowest_points", smooth=True)) is None


def test_events_of_a_subset_match_detect_events():
    df = _signals()
    s = _settings(peak_method="Peak Caller")
    analyzer = analysis.Analyzer()
    full = analyzer.run(df, s).events_df
    pd.testing.assert_frame_equal(full, events.detect_events(df, "Peak Caller"))
    sub = analyzer.run(df[["cell_2", "cell_9"]], s).events_df
    pd.testing.assert_frame_equal(sub, events.detect_events(df[["cell_2", "cell_9"]], "Peak Caller"))
    result = analyzer.run(df, s)
    assert result.params["peak_method"] == "Peak Caller"
    assert result.params["peaks_on"] == "raw signals"
    assert "events" in result.items


def test_events_are_cached_per_cell_until_settings_change(monkeypatch):
    df = _signals()
    calls = []
    real = events.detect_events

    def counting(signals_df, method, params=None):
        calls.append(list(signals_df.columns))
        return real(signals_df, method, params)

    monkeypatch.setattr(events, "detect_events", counting)
    analyzer = analysis.Analyzer()
    s = _settings(peak_method="Peak Caller")
    analyzer.run(df[["cell_2"]], s)
    analyzer.run(df[["cell_2", "cell_5"]], s)          # only the new cell is detected
    assert calls == [["cell_2"], ["cell_5"]]
    analyzer.run(df[["cell_2", "cell_5"]], _settings(peak_method="Peak Caller",
                                                     peak_params={"rise_percent": 20}))
    assert calls[2:] == [["cell_2"], ["cell_5"]]       # new parameters: detected again
    changed = df.copy()
    changed["cell_2"] = changed["cell_2"] * 2          # a cell's signal changed (e.g. new recording)
    calls.clear()
    analyzer.run(changed[["cell_2", "cell_5"]], _settings(peak_method="Peak Caller",
                                                          peak_params={"rise_percent": 20}))
    assert calls == [["cell_2"]]


def test_normalization_failure_raises():
    df = pd.DataFrame({"cell_1": np.zeros(50)})
    with pytest.raises(ValueError, match="cell_1"):
        analysis.Analyzer().run(df, _settings(method="lowest_points"))


def test_failing_peak_method_is_reported_not_raised(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no fit")

    monkeypatch.setattr(events, "detect_events", boom)
    result = analysis.Analyzer().run(_signals(), _settings(peak_method="Isolation Forest"))
    assert result.events_df is None and "no fit" in result.events_error
    assert "events" not in result.items


def test_settings_from_saved_params():
    s = analysis.settings_from_params({"method": "delta_f_over_f0", "smoothing_points": 8, "lowest_points": 8,
                                       "subtract_baseline": False, "peak_method": "Peak Caller",
                                       "peak_params": {"rise_percent": 10}})
    assert s["method"] == "delta_f_over_f0" and s["smooth"] and s["lowest_points"] == 8
    assert s["peak_method"] == "Peak Caller" and s["peak_params"] == {"rise_percent": 10}
    legacy = analysis.settings_from_params({"method": normalization.LEGACY_SUBTRACT_METHOD})
    assert legacy["method"] == "none" and legacy["subtract"]
    assert analysis.settings_from_params(None) == analysis.DEFAULT_SETTINGS
    # The saved params round-trip.
    result = analysis.Analyzer().run(_signals(), s)
    assert analysis.settings_from_params(result.params) == {**s, "peak_params": analysis.peak_params(s)}
