"""Tests for neclab_core (no GUI needed). Run with:  python -m pytest tests"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neclab_core import correlation, display, formats, segmentation, signals, variability  # noqa: E402


# ----------------------------------------------------------------- fixtures

def _blob(shape, center, radius):
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    return (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius ** 2


@pytest.fixture
def stack():
    """60 frames of 64x64 uint8: three blinking round cells on a noisy background,
    plus one large blob (two touching cells) for the splitting methods."""
    rng = np.random.RandomState(0)
    t, h, w = 60, 64, 64
    data = rng.normal(40, 2, size=(t, h, w))
    frames = np.arange(t)
    cells = [((12, 12), 4, 0.0), ((12, 45), 4, 1.0), ((45, 12), 4, 2.0)]
    for center, radius, phase in cells:
        m = _blob((h, w), center, radius)
        data[:, m] += (60 * (1 + np.sin(frames / 3.0 + phase)))[:, None]
    big = _blob((h, w), (46, 42), 7) | _blob((h, w), (46, 54), 7)
    data[:, big] += (60 * (1 + np.sin(frames / 5.0)))[:, None]
    return np.clip(data, 0, 255).astype(np.uint8)


def _old_neighboring_groups(mask):
    """The original pure-Python 8-connected grouping, kept here as a reference."""
    nodes = [(i, j) for i in range(mask.shape[0]) for j in range(mask.shape[1]) if mask[i, j]]
    remain = set(nodes)
    while remain:
        visit = [remain.pop()]
        group = []
        while visit:
            node = visit.pop()
            group.append(node)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (node[0] + di, node[1] + dj)
                    if nb in remain:
                        remain.remove(nb)
                        visit.append(nb)
        yield group


# --------------------------------------------------------------- variability

def test_all_variability_methods(stack):
    assert len(variability.method_names()) == 7
    for m in range(7):
        var_im, th, name = variability.calculate_variability(stack, m)
        assert var_im.shape == stack.shape[1:]
        assert th == variability.VARIABILITY_METHODS[m][1]
        assert name == variability.VARIABILITY_METHODS[m][0]
    with pytest.raises(ValueError):
        variability.calculate_variability(stack, 7)


def test_population_variance_matches_numpy(stack):
    var_im, _, _ = variability.calculate_variability(stack, 1)
    np.testing.assert_allclose(var_im, np.var(stack, axis=0))


# -------------------------------------------------------------- segmentation

def test_connected_components_match_original_algorithm():
    rng = np.random.RandomState(1)
    mask = rng.rand(40, 40) > 0.6
    new = segmentation.label_image_to_pixel_lists(segmentation.connected_components(mask))
    old = list(_old_neighboring_groups(mask))
    assert len(new) == len(old)
    assert {frozenset(g) for g in new} == {frozenset(g) for g in old}


def test_detect_cells_basic_ids_are_stable(stack):
    var_im, _, _ = variability.calculate_variability(stack, 1)
    processed = variability.sharpen(var_im)
    labels, mask = segmentation.detect_cells(var_im, processed, 1500, 20, 200, split="none")
    assert mask.dtype == bool
    assert labels.dtype == np.int32
    assert segmentation.cell_ids(labels) == [1, 2, 3]
    # IDs follow raster order of each cell's first pixel: top-left, top-right, bottom-left.
    table = segmentation.cell_table(labels)
    assert list(table["cell_id"]) == [1, 2, 3]
    assert table.loc[0, "centroid_col"] < table.loc[1, "centroid_col"]
    assert table.loc[0, "centroid_row"] < table.loc[2, "centroid_row"]
    # Same input -> same label image.
    again, _ = segmentation.detect_cells(var_im, processed, 1500, 20, 200, split="none")
    np.testing.assert_array_equal(labels, again)


def test_kmeans_split_keeps_the_split_cells(stack):
    var_im, _, _ = variability.calculate_variability(stack, 1)
    processed = variability.sharpen(var_im)
    basic, _ = segmentation.detect_cells(var_im, processed, 1500, 20, 200, split="none")
    split, _ = segmentation.detect_cells(var_im, processed, 1500, 20, 200, split="kmeans")
    # The large two-cell blob is dropped by the basic size filter but split by KMeans.
    assert len(segmentation.cell_ids(split)) >= len(segmentation.cell_ids(basic)) + 2
    sizes = segmentation.cell_table(split)["n_pixels"]
    assert (sizes >= 20).all()


def test_peak_split_runs(stack):
    var_im, _, _ = variability.calculate_variability(stack, 1)
    processed = variability.sharpen(var_im)
    labels, _ = segmentation.detect_cells(var_im, processed, 1500, 20, 200, split="peaks")
    assert len(segmentation.cell_ids(labels)) >= 3


def test_detect_cells_rejects_unknown_split(stack):
    var_im, _, _ = variability.calculate_variability(stack, 1)
    with pytest.raises(ValueError):
        segmentation.detect_cells(var_im, var_im, 10, split="watershed")


def test_relabel_raster_order():
    labels = np.array([[0, 7, 7], [3, 0, 0], [3, 0, 9]])
    out = segmentation.relabel_raster_order(labels)
    np.testing.assert_array_equal(out, [[0, 1, 1], [2, 0, 0], [2, 0, 3]])


def test_subset_labels_keeps_ids():
    labels = np.array([[1, 2], [3, 0]], dtype=np.int32)
    np.testing.assert_array_equal(segmentation.subset_labels(labels, [3]), [[0, 0], [3, 0]])


# ------------------------------------------------------------------- signals

def test_extract_signals_is_mean_of_cell_pixels(stack):
    labels = np.zeros(stack.shape[1:], dtype=np.int32)
    labels[10:13, 10:13] = 4
    labels[40:42, 40:44] = 9
    df = signals.extract_signals(stack, labels)
    assert list(df.columns) == ["cell_4", "cell_9"]
    assert df.index.name == "frame" and len(df) == stack.shape[0]
    np.testing.assert_allclose(df["cell_4"], stack[:, 10:13, 10:13].reshape(stack.shape[0], -1).mean(axis=1))
    only9 = signals.extract_signals(stack, labels, cell_ids=[9])
    assert list(only9.columns) == ["cell_9"]
    with pytest.raises(ValueError):
        signals.extract_signals(stack, labels, cell_ids=[5])
    with pytest.raises(ValueError):
        signals.extract_signals(stack, labels[:10])


def test_cell_column_names():
    assert signals.cell_column(12) == "cell_12"
    assert signals.parse_cell_column("cell_12") == 12
    assert signals.parse_cell_column("Column 1") is None


# ------------------------------------------------------------------- formats

@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".npy"])
def test_signals_round_trip(tmp_path, ext):
    df = pd.DataFrame({"cell_2": [1.0, 2.0, 3.5], "cell_7": [0.5, 0.25, 0.0]})
    path = tmp_path / f"signals{ext}"
    formats.write_signals(df, path)
    back, warnings = formats.read_signals(path)
    assert warnings == []
    np.testing.assert_allclose(back.to_numpy(), df.to_numpy())
    if ext != ".npy":
        assert formats.signal_cell_ids(back) == [2, 7]


def test_signals_npy_layout_has_frame_column(tmp_path):
    df = pd.DataFrame({"cell_1": [5.0, 6.0, 7.0]})
    path = tmp_path / "s.npy"
    formats.write_signals(df, path)
    arr = np.load(path)
    np.testing.assert_array_equal(arr[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(arr[:, 1], [5, 6, 7])


def test_is_frame_index():
    assert formats.is_frame_index([0, 1, 2, 3])
    assert not formats.is_frame_index([1, 2, 3, 4])
    assert not formats.is_frame_index([0.0, 1.0, 2.5])
    assert not formats.is_frame_index([0])


@pytest.mark.parametrize("ext", [".tif", ".npy"])
def test_cells_round_trip(tmp_path, ext):
    labels = np.zeros((20, 30), dtype=np.int32)
    labels[2:5, 2:5] = 3
    labels[10:12, 20:25] = 8
    path = tmp_path / f"cells{ext}"
    img_path, table_path = formats.save_cells(labels, path)
    assert os.path.exists(table_path)
    table = pd.read_csv(table_path)
    assert list(table["cell_id"]) == [3, 8]
    back = formats.load_cells(img_path, shape=(20, 30))
    np.testing.assert_array_equal(back, labels)
    with pytest.raises(ValueError):
        formats.load_cells(img_path, shape=(21, 30))


def test_load_legacy_coordinates_txt(tmp_path):
    path = tmp_path / "cluster_coordinates.txt"
    path.write_text(
        "# Coordinates of Selected Clusters\n"
        "# Variability method: Population Variance\n\n"
        "Cluster 5 (2 points):\n1,1\n1,2\n\n"
        "Cluster 0 (1 points):\n0,3\n\n"
    )
    labels = formats.load_cells(path, shape=(4, 4))
    assert segmentation.cell_ids(labels) == [1, 2]
    assert labels[0, 3] == 1 and labels[1, 1] == 2 and labels[1, 2] == 2
    with pytest.raises(ValueError):
        formats.load_cells(path)


def test_load_cells_rejects_bad_images(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.array([[0.5, 1.0]]))
    with pytest.raises(ValueError):
        formats.load_cells(path)
    np.save(path, np.zeros((2, 3, 4), dtype=np.int32))
    with pytest.raises(ValueError):
        formats.load_cells(path)


# --------------------------------------------------------------- correlation

def test_correlation_matrix_matches_pandas():
    rng = np.random.RandomState(2)
    data = rng.normal(size=(50, 4))
    for method in correlation.CORRELATION_METHODS:
        pd.testing.assert_frame_equal(correlation.correlation_matrix(data, method),
                                      pd.DataFrame(data).corr(method=method))
    with pytest.raises(ValueError):
        correlation.correlation_matrix(data, "cosine")


def test_linkages():
    rng = np.random.RandomState(3)
    data = rng.normal(size=(40, 5))
    link = correlation.agglomerative_linkage(data.T)
    assert link.shape == (4, 4)
    assert link[-1, 3] == 5
    corr = correlation.correlation_matrix(data)
    assert correlation.abs_correlation_ward_linkage(corr).shape == (4, 4)


# ------------------------------------------------------------------- display

def test_auto_contrast_matches_original_rgb_pipeline():
    from PIL import Image, ImageOps

    rng = np.random.RandomState(4)
    frame = rng.randint(30, 120, size=(32, 32)).astype(np.uint8)
    old = np.array(ImageOps.autocontrast(Image.fromarray(frame).convert("RGB"), cutoff=2, ignore=2).convert("L"))
    np.testing.assert_array_equal(display.auto_contrast(frame), old)


def test_display_conversion():
    frame8 = np.array([[0, 128, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(display.to_display_uint8(frame8), frame8)
    stack16 = np.array([[[1000, 2000]], [[3000, 5000]]], dtype=np.uint16)
    rng16 = display.display_range(stack16)
    assert rng16 == (1000.0, 5000.0)
    np.testing.assert_array_equal(display.to_display_uint8(stack16[1], rng16), [[127, 255]])


def test_display_functions_do_not_modify_input():
    frame = np.full((4, 4), 100, dtype=np.uint8)
    before = frame.copy()
    display.apply_brightness_contrast(frame, 50, 50)
    display.auto_contrast(frame)
    display.to_display_uint8(frame)
    np.testing.assert_array_equal(frame, before)


# --------------------------------------------- Multiple Files loader (needs Tk)

def test_multiple_files_loader_drops_frame_column(tmp_path):
    pytest.importorskip("tkinter")
    import multi_xls_helpers

    df = pd.DataFrame({"cell_1": [3.0, 4.0, 5.0], "cell_4": [9.0, 8.0, 7.0]})
    for ext in (".csv", ".xlsx", ".npy"):
        path = tmp_path / f"signals{ext}"
        formats.write_signals(df, path)
        sheet = multi_xls_helpers._read_sheet_names(str(path))[0]
        loaded = multi_xls_helpers._load_sheet_dataframe(str(path), sheet)
        assert list(loaded.columns) == ["0", "1"], ext
        np.testing.assert_allclose(loaded.to_numpy(), df.to_numpy())

    # A headerless file with no frame column is left as is.
    plain = tmp_path / "plain.csv"
    pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]).to_csv(plain, header=False, index=False)
    loaded = multi_xls_helpers._load_sheet_dataframe(str(plain), "(CSV)")
    np.testing.assert_allclose(loaded.to_numpy(), [[1, 2], [3, 4], [5, 7]])
