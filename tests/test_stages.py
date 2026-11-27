"""Tests for entering the pipeline at any stage: per-stage formats,
validation and the stages registry. Run with:  python -m pytest tests"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neclab_core import formats, segmentation, signals, stages, validation  # noqa: E402
from neclab_core.validation import ValidationError  # noqa: E402

SHAPE = (24, 32)


@pytest.fixture
def stack():
    rng = np.random.RandomState(0)
    return rng.randint(0, 200, size=(15,) + SHAPE).astype(np.uint16)


@pytest.fixture
def labels():
    lab = np.zeros(SHAPE, dtype=np.int32)
    lab[2:6, 3:8] = 1
    lab[10:14, 20:26] = 2
    return lab


# ---------------------------------------------------------- registry

def test_every_stage_and_item_is_registered():
    assert [s.number for s in stages.STAGES] == list(range(7))
    assert stages.stage("cells").number == 4 and stages.stage(4).key == "cells"
    for item in stages.ITEMS.values():
        assert item.stage in {s.key for s in stages.STAGES}
        assert stages.open_filetypes(item.key)[-1] == ("All files", "*.*")
    assert {i.key for i in stages.items_for_stage("signals")} == {"signals", "normalized_signals", "events"}
    assert stages.stage("correlation").number == 6


# ---------------------------------------------------------- stage 0/1

@pytest.mark.parametrize("name", ["rec.tif", "rec.npy", "rec.ome.tif"])
def test_stack_round_trip(tmp_path, stack, name):
    import tifffile

    path = tmp_path / name
    if name.endswith(".ome.tif"):
        tifffile.imwrite(path, stack, ome=True, metadata={"axes": "TYX"})
    else:
        stages.save_item("processed_stack", stack, path)
    loaded = stages.load_item("stack", path)
    np.testing.assert_array_equal(loaded.data, stack)


def test_stack_rejects_2d_and_wrong_frame_size(tmp_path, stack, labels):
    path = tmp_path / "single.npy"
    np.save(path, stack[0])
    with pytest.raises(ValidationError, match="single 2D image"):
        stages.load_item("stack", path)
    path = tmp_path / "rec.npy"
    np.save(path, stack[:, :20, :20])
    with pytest.raises(ValidationError, match="same size"):
        stages.load_item("stack", path, stages.Context(labels=labels))


def test_raw_stack_is_never_saved(stack, tmp_path):
    with pytest.raises(ValidationError):
        stages.save_item("stack", stack, tmp_path / "x.tif")


# ---------------------------------------------------------- stage 2

@pytest.mark.parametrize("name", ["maps.npz", "Population Variance.npy", "Population Variance.tif"])
def test_maps_round_trip(tmp_path, name):
    m = np.random.RandomState(1).rand(*SHAPE) * 500
    maps = {"Population Variance": m}
    path = tmp_path / name
    stages.save_item("maps", maps, path)
    loaded = stages.load_item("maps", path)
    assert list(loaded.data) == ["Population Variance"]
    np.testing.assert_allclose(loaded.data["Population Variance"], m, rtol=1e-6)


def test_several_maps_need_npz(tmp_path):
    maps = {"a": np.ones(SHAPE), "b": np.zeros(SHAPE)}
    with pytest.raises(ValidationError, match="npz"):
        stages.save_item("maps", maps, tmp_path / "m.npy")
    stages.save_item("maps", maps, tmp_path / "m.npz")
    assert set(stages.load_item("maps", tmp_path / "m.npz").data) == {"a", "b"}


def test_maps_validation(tmp_path, labels):
    with pytest.raises(ValidationError, match="different sizes"):
        validation.check_maps({"a": np.ones((4, 4)), "b": np.ones((5, 4))})
    with pytest.raises(ValidationError, match="same size"):
        validation.check_maps({"a": np.ones((4, 4))}, frame_shape=SHAPE)
    with pytest.raises(ValidationError, match="no finite"):
        validation.check_map(np.full(SHAPE, np.nan))
    m = np.ones(SHAPE)
    m[0, 0] = np.nan
    assert validation.check_map(m)  # warning about the NaN pixel


# ---------------------------------------------------------- stage 3

@pytest.mark.parametrize("name", ["mask.tif", "mask.npy"])
def test_mask_round_trip(tmp_path, name):
    mask = np.zeros(SHAPE, bool)
    mask[3:9, 4:10] = True
    stages.save_item("mask", mask, tmp_path / name)
    loaded = stages.load_item("mask", tmp_path / name)
    assert loaded.data.dtype == bool
    np.testing.assert_array_equal(loaded.data, mask)


def test_mask_accepts_0_1_and_0_255_but_not_maps(tmp_path):
    for values in ((0, 1), (0, 255)):
        arr = np.zeros(SHAPE, np.uint8)
        arr[1:3, 1:3] = values[1]
        np.save(tmp_path / "m.npy", arr)
        assert stages.load_item("mask", tmp_path / "m.npy").data.sum() == 4
    np.save(tmp_path / "map.npy", np.random.RandomState(0).rand(*SHAPE))
    with pytest.raises(ValidationError, match="stage 2"):
        stages.load_item("mask", tmp_path / "map.npy")
    assert validation.check_mask(np.zeros(SHAPE, bool))  # empty-mask warning


# ---------------------------------------------------------- stage 4

def test_cells_label_image_checks(tmp_path, labels):
    stages.save_item("cells", labels, tmp_path / "cells.tif")
    loaded = stages.load_item("cells", tmp_path / "cells.tif", stages.Context(stack=np.zeros((3,) + SHAPE)))
    np.testing.assert_array_equal(loaded.data, labels)
    with pytest.raises(ValidationError, match="mask"):
        validation.check_labels(labels > 0)
    with pytest.raises(ValidationError, match="negative"):
        validation.check_labels(labels - 5)
    assert validation.check_labels(np.zeros(SHAPE, np.int32))  # "no cells" warning


def test_imagej_rois(tmp_path):
    roifile = pytest.importorskip("roifile")
    rect = roifile.ImagejRoi()
    rect.roitype = roifile.ROI_TYPE.RECT
    rect.left, rect.top, rect.right, rect.bottom = 2, 3, 8, 6   # x 2..7, y 3..5
    poly = roifile.ImagejRoi.frompoints(np.array([[20, 10], [26, 10], [26, 16], [20, 16]], float))
    line = roifile.ImagejRoi()
    line.roitype = roifile.ROI_TYPE.LINE
    line.name = "a line"
    roifile.roiwrite(tmp_path / "RoiSet.zip", [rect, line, poly])

    labels, skipped = formats.rois_to_labels(tmp_path / "RoiSet.zip", SHAPE)
    assert skipped == ["a line"]
    assert segmentation.cell_ids(labels) == [1, 2]
    expected = np.zeros(SHAPE, bool)
    expected[3:6, 2:8] = True
    np.testing.assert_array_equal(labels == 1, expected)
    assert (labels == 2).sum() == 36  # 6 x 6 pixels
    loaded = stages.load_item("cells", tmp_path / "RoiSet.zip", stages.Context(mask=np.zeros(SHAPE, bool)))
    np.testing.assert_array_equal(loaded.data, labels)
    with pytest.raises(ValidationError, match="image size"):
        stages.load_item("cells", tmp_path / "RoiSet.zip")


# ---------------------------------------------------------- stage 5/6

def test_signals_validated_against_loaded_cells_and_frames(tmp_path, stack, labels):
    df = signals.extract_signals(stack, labels)
    stages.save_item("signals", df, tmp_path / "s.csv")
    ctx = stages.Context(stack=stack, labels=labels)
    loaded = stages.load_item("signals", tmp_path / "s.csv", ctx)
    assert list(loaded.data.columns) == ["cell_1", "cell_2"]

    with pytest.raises(ValidationError, match="frames"):
        stages.load_item("signals", tmp_path / "s.csv", stages.Context(stack=stack[:10]))
    other = np.where(labels == 2, 9, labels)
    with pytest.raises(ValidationError, match="cell_2"):
        stages.load_item("signals", tmp_path / "s.csv", stages.Context(labels=other))


def test_signals_from_other_programs(tmp_path):
    # Headerless, with a time column of seconds -> told it is time.
    raw = pd.DataFrame({"t": [0.0, 0.5, 1.0, 1.5], "a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    raw.to_csv(tmp_path / "noheader.csv", header=False, index=False)
    kept, _ = formats.read_signals(tmp_path / "noheader.csv")
    assert kept.shape == (4, 3)           # 0, .5, 1 isn't 0,1,2: kept by "auto"
    dropped, _ = formats.read_signals(tmp_path / "noheader.csv", time_column=True)
    assert list(dropped.columns) == ["column_1", "column_2"]

    # Header with "Time" -> dropped automatically; other names kept.
    raw.rename(columns={"t": "Time"}).to_csv(tmp_path / "header.csv", index=False)
    df, _ = formats.read_signals(tmp_path / "header.csv")
    assert list(df.columns) == ["a", "b"]
    all_cols, _ = formats.read_signals(tmp_path / "header.csv", time_column=False)
    assert list(all_cols.columns) == ["Time", "a", "b"]

    # Missing values -> warning, not error.
    raw.loc[1, "a"] = None
    raw.to_csv(tmp_path / "gap.csv", index=False)
    _, warnings = formats.read_signals(tmp_path / "gap.csv")
    assert any("missing" in w for w in warnings)


def test_signals_rejects_bad_tables(tmp_path):
    pd.DataFrame({"frame": [0], "cell_1": [1.0]}).to_csv(tmp_path / "one_row.csv", index=False)
    with pytest.raises(ValidationError, match="at least 2"):
        formats.read_signals(tmp_path / "one_row.csv")
    pd.DataFrame({"frame": [0, 1, 2], "name": ["a", "b", "c"]}).to_csv(tmp_path / "text.csv", index=False)
    with pytest.raises(ValidationError, match="no numbers"):
        formats.read_signals(tmp_path / "text.csv")


def test_normalized_signals_are_flagged(tmp_path):
    df = pd.DataFrame({"cell_1": [0.0, 0.5, 1.0]})
    stages.save_item("normalized_signals", df, tmp_path / "n.csv")
    assert stages.load_item("normalized_signals", tmp_path / "n.csv").metadata["normalized"] is True
    assert stages.load_item("signals", tmp_path / "n.csv").metadata["normalized"] is False


def test_events_round_trip_and_checks(tmp_path, labels):
    ev = pd.DataFrame({"cell_id": [1, 1, 2], "frame": [3, 9, 4], "amplitude": [1.5, 2.0, 0.7]})
    stages.save_item("events", ev, tmp_path / "events.csv")
    loaded = stages.load_item("events", tmp_path / "events.csv",
                              stages.Context(stack=np.zeros((10,) + SHAPE), labels=labels))
    pd.testing.assert_frame_equal(loaded.data, ev)
    with pytest.raises(ValidationError, match="frame 9"):
        stages.load_item("events", tmp_path / "events.csv", stages.Context(stack=np.zeros((5,) + SHAPE)))
    with pytest.raises(ValidationError, match="frame"):
        validation.check_events(pd.DataFrame({"cell_id": [1]}))


def test_events_from_multiple_files_peaks_export(tmp_path):
    legacy = pd.DataFrame({"Column": ["Column 1"] * 4, "Sheet": ["rec - (CSV)"] * 4,
                           "Sample_Index": [0, 1, 2, 3], "Peak_Flag": [0, 1, 0, 1]})
    legacy.to_csv(tmp_path / "peaks.csv", index=False)
    ev, _ = formats.read_events(tmp_path / "peaks.csv")
    assert list(ev.columns) == ["signal", "sheet", "frame"]
    assert list(ev["frame"]) == [1, 3]


# ---------------------------------------------------------- stage 7

@pytest.mark.parametrize("name", ["c.csv", "c.xlsx"])
def test_correlation_round_trip(tmp_path, name):
    data = np.random.RandomState(2).normal(size=(30, 3))
    corr = pd.DataFrame(data, columns=["cell_1", "cell_2", "cell_5"]).corr()
    stages.save_item("correlation", corr, tmp_path / name)
    loaded = stages.load_item("correlation", tmp_path / name)
    pd.testing.assert_frame_equal(loaded.data, corr)
    assert loaded.warnings == []


def test_correlation_checks():
    with pytest.raises(ValidationError, match="square"):
        validation.check_correlation(pd.DataFrame(np.ones((2, 3))))
    with pytest.raises(ValidationError, match="non-numeric"):
        validation.check_correlation(pd.DataFrame([[1, None], [None, 1]], index=["a", "b"], columns=["a", "b"]))
    warnings = validation.check_correlation(pd.DataFrame([[1, 3], [2, 1]], index=["a", "b"], columns=["a", "b"]))
    assert any("range" in w for w in warnings) and any("symmetric" in w for w in warnings)


# ---------------------------------------------------------- cross-stage

def test_check_compatibility(stack, labels):
    df = signals.extract_signals(stack, labels)
    validation.check_compatibility(stack=stack, mask=labels > 0, labels=labels, signals=df)
    with pytest.raises(ValidationError, match="rows"):
        validation.check_compatibility(stack=stack[:5], labels=labels, signals=df)
    with pytest.raises(ValidationError, match="same size"):
        validation.check_compatibility(stack=stack, mask=np.zeros((5, 5), bool))


def test_context_frame_shape_fallbacks(labels):
    assert stages.Context().frame_shape is None
    assert stages.Context(labels=labels).frame_shape == SHAPE
    assert stages.Context(maps={"m": np.ones(SHAPE)}).frame_shape == SHAPE
    assert stages.Context(labels=labels).cell_ids == [1, 2]


def test_detect_cells_from_loaded_mask(labels):
    mask = labels > 0
    out = segmentation.detect_cells_from_mask(mask, min_size=5, max_size=100)
    assert segmentation.cell_ids(out) == [1, 2]
    with pytest.raises(ValueError, match="variability map"):
        segmentation.detect_cells_from_mask(mask, split="kmeans")
