"""Tests for the shared normalization (stage 6) and clustering (stage 7)
modules. Run with:  python -m pytest tests"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neclab_core import correlation, normalization as norm  # noqa: E402


def _signals():
    t = np.arange(200)
    return pd.DataFrame({
        "cell_1": 100 + 50 * (np.sin(t / 8) > 0.9) + 0.05 * t,
        "cell_2": 200 + 30 * np.cos(t / 11),
        "cell_3": np.full(200, 7.0),
    })


# ------------------------------------------------------------ single signals

def test_divide_by_min_and_zero_guard():
    np.testing.assert_allclose(norm.divide_by_min([2.0, 4.0, np.nan]), [1.0, 2.0, np.nan])
    np.testing.assert_allclose(norm.divide_by_min([0.0, 4.0]), [0.0, 4.0])    # zero minimum: unchanged


def test_delta_f_over_f0():
    v = np.array([10.0] * 9 + [20.0])
    np.testing.assert_allclose(norm.delta_f_over_f0(v), [0.0] * 9 + [1.0])
    np.testing.assert_allclose(norm.delta_f_over_f0(v, percentile=0), [0.0] * 9 + [1.0])
    np.testing.assert_allclose(norm.delta_f_over_f0([0.0, 0.0, 3.0]), [0.0, 0.0, 3.0])  # F0 = 0: F - F0


def test_detrend_minmax_range_and_flat_signal():
    out = norm.detrend_minmax(_signals()["cell_1"])
    assert out.min() == pytest.approx(0) and out.max() == pytest.approx(1)
    np.testing.assert_array_equal(norm.detrend_minmax(np.full(10, 3.0)), np.zeros(10))


def test_smoothing_keeps_gaps():
    v = _signals()["cell_1"].to_numpy().copy()
    v[[5, 50]] = np.nan
    out = norm.smooth_keeping_gaps(v, 6)
    assert np.isnan(out[[5, 50]]).all() and np.isfinite(np.delete(out, [5, 50])).all()


def test_fill_missing():
    np.testing.assert_allclose(norm.fill_missing([np.nan, 1.0, np.nan, 3.0, np.nan]), [1, 1, 2, 3, 3])


def test_divide_by_lowest_points_puts_them_on_a_horizontal_line():
    t = np.arange(300)
    y = 100 + 0.2 * t + 40 * (np.sin(t / 9) > 0.85) + np.sin(t / 3)     # drifting baseline + events
    for n in (2, 6, 10):
        out = norm.divide_by_lowest_points(y, n)
        baseline, px = norm.lowest_points_baseline(y, n)
        np.testing.assert_allclose(out[px], 1.0)                          # lowest points -> 1
        np.testing.assert_allclose(out, y / baseline)
        assert out.min() > 0.9                                             # nothing far below the line


def test_divide_by_lowest_points_matches_the_smoothing_baseline():
    t = np.arange(200)
    y = 50 + 0.1 * t + 20 * (np.cos(t / 7) > 0.9)
    baseline, px = norm.lowest_points_baseline(y, 6)
    smoothed = norm.smooth_convex_envelope(y, 6)                           # subtraction
    np.testing.assert_allclose(smoothed, y - baseline + y[px].min())
    np.testing.assert_allclose(norm.divide_by_lowest_points(y, 6), y / baseline)


def test_divide_by_lowest_points_keeps_gaps_and_rejects_near_zero_baselines():
    t = np.arange(120)
    y = 80 + 10 * (np.sin(t / 5) > 0.9)
    y = y.astype(float)
    y[[4, 60]] = np.nan
    out = norm.divide_by_lowest_points(y, 6)
    assert np.isnan(out[[4, 60]]).all() and np.isfinite(np.delete(out, [4, 60])).all()
    with pytest.raises(ValueError, match="blow up"):
        norm.divide_by_lowest_points(np.sin(t / 5), 6)                    # baseline below zero
    df = pd.DataFrame({"cell_1": 80 + np.sin(t / 5), "cell_9": np.sin(t / 5)})
    with pytest.raises(ValueError, match="cell_9"):
        norm.normalize_signals(df, "lowest_points")
    ok = norm.normalize_signals(df[["cell_1"]], "lowest_points", lowest_points=4)
    _, px = norm.lowest_points_baseline(df["cell_1"], 4)
    np.testing.assert_allclose(ok["cell_1"].to_numpy()[px], 1.0)


# ------------------------------------------------------------ tables

def test_normalize_signals_methods():
    df = _signals()
    for method in norm.NORMALIZATION_METHODS:
        out = norm.normalize_signals(df, method)
        assert list(out.columns) == list(df.columns) and out.shape == df.shape
    pd.testing.assert_frame_equal(norm.normalize_signals(df, "none"), df.astype(float))
    np.testing.assert_allclose(norm.normalize_signals(df, "divide_by_min")["cell_2"].min(), 1.0)
    smoothed = norm.normalize_signals(df, "none", smoothing_points=6)
    assert not np.allclose(smoothed["cell_1"], df["cell_1"])
    with pytest.raises(ValueError):
        norm.normalize_signals(df, "zscore")


def test_is_identity():
    assert norm.is_identity("none") and norm.is_identity("none", 0)
    assert not norm.is_identity("none", 4) and not norm.is_identity("delta_f_over_f0")


def test_divide_by_min_tables_modes():
    a = pd.DataFrame({"0": [2.0, 4.0], "1": [10.0, 20.0]})
    b = pd.DataFrame({"0": [1.0, 8.0], "1": [5.0, np.nan]})
    expected = {
        "local": ([[1, 1], [2, 2]], [[1, 1], [8, np.nan]]),
        "sheet": ([[1, 5], [2, 10]], [[1, 5], [8, np.nan]]),
        "column_global": ([[2, 2], [4, 4]], [[1, 1], [8, np.nan]]),
        "all_global": ([[2, 10], [4, 20]], [[1, 5], [8, np.nan]]),
        "none": ([[2, 10], [4, 20]], [[1, 5], [8, np.nan]]),
    }
    for mode, (ea, eb) in expected.items():
        oa, ob = norm.divide_by_min_tables([a, b], mode)
        np.testing.assert_allclose(oa.to_numpy(), ea, err_msg=mode)
        np.testing.assert_allclose(ob.to_numpy(), eb, err_msg=mode)
    with pytest.raises(ValueError):
        norm.divide_by_min_tables([a], "median")


def test_divide_by_min_tables_skips_zero_and_empty_columns():
    t = pd.DataFrame({"0": [0.0, 3.0], "1": [np.nan, np.nan]})
    for mode in ("local", "sheet", "column_global", "all_global"):
        (out,) = norm.divide_by_min_tables([t], mode)
        np.testing.assert_allclose(out["0"], [0.0, 3.0], err_msg=mode)
        assert out["1"].isna().all()


# ------------------------------------------------------------ clustering

def test_clustering_groups_correlated_and_anticorrelated_signals():
    rng = np.random.RandomState(0)
    a = rng.normal(size=300)
    b = rng.normal(size=300)
    data = pd.DataFrame({"a1": a, "a2": a + 0.05 * rng.normal(size=300), "a_neg": -a,
                         "b1": b, "b2": b + 0.05 * rng.normal(size=300)})
    corr = correlation.correlation_matrix(data)
    link = correlation.cluster_linkage(corr)
    from scipy.cluster.hierarchy import fcluster
    groups = fcluster(link, 2, criterion="maxclust")
    assert groups[0] == groups[1] == groups[2] != groups[3] == groups[4]
    assert sorted(correlation.dendrogram_order(link)) == list(range(5))
    table = correlation.linkage_table(link)
    assert list(table.columns) == ["Merge_Step", "Child_1", "Child_2", "Distance", "Size"]
    assert table["Size"].iloc[-1] == 5


def test_correlation_distance():
    corr = pd.DataFrame([[1.0, 0.8, -1.02], [0.8, 1.0, 0.0], [-1.02, 0.0, 1.0]])
    d = correlation.correlation_distance(corr)
    np.testing.assert_allclose(np.diag(d), 0)
    assert d[0, 1] == pytest.approx(0.2) and d[0, 2] == 0 and d[1, 2] == 1   # clipped at 0
    with pytest.raises(ValueError, match="missing"):
        correlation.correlation_distance(pd.DataFrame([[1, np.nan], [np.nan, 1]]))
    with pytest.raises(ValueError):
        correlation.cluster_linkage(pd.DataFrame([[1.0]]))


def test_constant_signal_gives_a_readable_error():
    corr = correlation.correlation_matrix(_signals())   # cell_3 is constant -> NaN correlations
    with pytest.raises(ValueError, match="constant"):
        correlation.cluster_linkage(corr)


def test_linkage_method_is_shared_setting(monkeypatch):
    rng = np.random.RandomState(1)
    corr = correlation.correlation_matrix(rng.normal(size=(50, 4)))
    ward = correlation.cluster_linkage(corr)
    monkeypatch.setattr(correlation, "LINKAGE_METHOD", "average")
    assert not np.allclose(correlation.cluster_linkage(corr), ward)
