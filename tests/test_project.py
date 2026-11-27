"""Tests for the project file (neclab_core.project). Run with:  python -m pytest tests"""

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neclab_core import correlation, project as proj, segmentation, signals, variability  # noqa: E402
from neclab_core.project import DONE, LOADED, MISSING, NOT_RUN, OUT_OF_DATE, Project  # noqa: E402
from neclab_core.validation import ValidationError  # noqa: E402


def _recording(seed=0):
    rng = np.random.RandomState(seed)
    t, h, w = 40, 32, 32
    data = rng.normal(400, 20, size=(t, h, w))
    frames = np.arange(t)
    data[:, 5:11, 5:11] += (600 * (1 + np.sin(frames / 3.0)))[:, None, None]
    data[:, 20:27, 18:25] += (600 * (1 + np.cos(frames / 4.0)))[:, None, None]
    return np.clip(data, 0, 65535).astype(np.uint16)


@pytest.fixture
def rec_path(tmp_path):
    path = tmp_path / "data" / "recording.npy"
    path.parent.mkdir()
    np.save(path, _recording())
    return path


def run_pipeline(p, stack, path):
    """Fill a project the way the app does: recording -> map -> mask/cells -> signals -> correlation."""
    p.set_recording(stack, path)
    var_im, th, name = variability.calculate_variability(stack, 1)
    p.set_stage("features", {"maps": {name: var_im}}, params={"method": name})
    labels, mask = segmentation.detect_cells(var_im, variability.sharpen(var_im), 1000, 10, 200)
    p.set_stage("mask", {"mask": mask}, params={"threshold": 1000, "map": name})
    p.set_stage("cells", {"cells": labels}, params={"split": "none", "min_size": 10, "max_size": 200})
    df = signals.extract_signals(stack, labels)
    p.set_stage("signals", {"signals": df}, params={"cells": segmentation.cell_ids(labels)})
    p.set_stage("correlation", {"correlation": correlation.correlation_matrix(df)}, params={"method": "pearson"})
    p.ui["selected_cells"] = segmentation.cell_ids(labels)
    return labels, df


# --------------------------------------------------------------- status

def test_status_and_out_of_date_propagation(rec_path):
    stack = np.load(rec_path)
    p = Project()
    assert all(r["status"] == NOT_RUN for r in p.summary())
    labels, _ = run_pipeline(p, stack, rec_path)
    assert p.status("raw") == LOADED
    for key in ("features", "mask", "cells", "signals", "correlation"):
        assert p.status(key) == DONE, key
    assert p.modified

    # New detection: everything computed from the cells is out of date, earlier stages aren't.
    p.set_stage("cells", {"cells": np.where(labels == 2, 0, labels)}, params={"split": "none"})
    assert p.status("cells") == DONE
    assert p.status("signals") == OUT_OF_DATE
    assert p.status("correlation") == OUT_OF_DATE   # through signals
    assert p.status("mask") == DONE

    # Re-running a stage with the same data and params keeps downstream up to date.
    p2 = Project()
    run_pipeline(p2, stack, rec_path)
    var_im = p2.data("features")["Population Variance"]
    p2.set_stage("features", {"maps": {"Population Variance": var_im.copy()}}, params={"method": "Population Variance"})
    assert p2.status("cells") == DONE


def test_new_recording_marks_everything_out_of_date(rec_path, tmp_path):
    p = Project()
    run_pipeline(p, np.load(rec_path), rec_path)
    other = tmp_path / "other.npy"
    np.save(other, _recording(seed=1))
    p.set_recording(np.load(other), other)
    assert [p.status(k) for k in ("features", "cells", "signals", "correlation")] == [OUT_OF_DATE] * 4


def test_imported_stage_is_loaded_and_not_invalidated_by_later_inputs(rec_path):
    p = Project()
    labels = np.zeros((32, 32), np.int32)
    labels[1:4, 1:4] = 1
    p.set_stage("cells", {"cells": labels}, origin="imported", source={"path": "/x/cells.tif"})
    assert p.status("cells") == LOADED
    p.set_recording(np.load(rec_path), rec_path)     # attaching the recording later
    assert p.status("cells") == LOADED
    assert p.summary()[4]["source"] == "imported: cells.tif"


def test_set_stage_rejects_unknown_items():
    with pytest.raises(KeyError):
        Project().set_stage("cells", {"mask": np.zeros((2, 2), bool)})
    with pytest.raises(KeyError):
        Project().set_stage("nope", {})


# --------------------------------------------------------------- save / load

def test_save_and_load_round_trip(rec_path, tmp_path):
    stack = np.load(rec_path)
    p = Project()
    labels, df = run_pipeline(p, stack, rec_path)
    folder = tmp_path / "exp.neclab"
    p.save(folder)
    assert not p.modified and p.name == "exp"

    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["format"] == "neclab-project" and manifest["format_version"] == 2
    assert set(manifest["stages"]) == {"raw", "features", "mask", "cells", "signals", "correlation"}
    assert sorted(os.listdir(folder)) == sorted(["manifest.json", "s0_raw", "s2_features", "s3_mask",
                                                 "s4_cells", "s5_signals", "s6_correlation"])
    assert not (folder / "s0_raw" / "recording.npy").exists()      # the recording isn't copied
    assert manifest["stages"]["raw"]["source"]["relative_path"] == os.path.join("..", "data", "recording.npy")

    q, warnings = Project.load(folder)
    assert warnings == []
    assert not q.modified
    assert [r["status"] for r in q.summary()] == [r["status"] for r in p.summary()]
    np.testing.assert_array_equal(q.data("raw"), stack)
    np.testing.assert_array_equal(q.data("cells"), labels)
    np.testing.assert_allclose(q.data("signals").to_numpy(), df.to_numpy())
    assert q.get("mask").params["threshold"] == 1000
    assert q.ui["selected_cells"] == segmentation.cell_ids(labels)
    pd.testing.assert_frame_equal(q.data("correlation"), p.data("correlation"))


def test_project_and_recording_moved_together(rec_path, tmp_path):
    p = Project()
    run_pipeline(p, np.load(rec_path), rec_path)
    p.save(tmp_path / "exp.neclab")
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.move(str(tmp_path / "exp.neclab"), moved / "exp.neclab")
    shutil.move(str(tmp_path / "data"), moved / "data")
    q, warnings = Project.load(moved / "exp.neclab")
    assert warnings == [] and q.status("raw") == LOADED
    assert q.get("raw").source["path"] == str(moved / "data" / "recording.npy")


def test_missing_recording_keeps_the_other_stages(rec_path, tmp_path):
    p = Project()
    labels, _ = run_pipeline(p, np.load(rec_path), rec_path)
    p.save(tmp_path / "exp.neclab")
    hidden = tmp_path / "hidden.npy"
    shutil.move(str(rec_path), hidden)

    q, warnings = Project.load(tmp_path / "exp.neclab")
    assert q.status("raw") == MISSING and q.data("raw") is None
    assert any("not found" in w for w in warnings)
    assert q.status("cells") == DONE and q.status("signals") == DONE
    np.testing.assert_array_equal(q.data("cells"), labels)

    assert q.relocate_recording(str(hidden)) == []        # same file: nothing out of date
    assert q.status("raw") == LOADED and q.status("signals") == DONE


def test_changed_recording_is_detected(rec_path, tmp_path):
    p = Project()
    run_pipeline(p, np.load(rec_path), rec_path)
    p.save(tmp_path / "exp.neclab")
    np.save(rec_path, _recording(seed=3))                  # overwritten with different data
    q, warnings = Project.load(tmp_path / "exp.neclab")
    assert any("has changed" in w for w in warnings)
    assert q.status("raw") == LOADED
    assert q.status("features") == OUT_OF_DATE and q.status("signals") == OUT_OF_DATE


def test_a_broken_stage_file_does_not_block_the_project(rec_path, tmp_path):
    p = Project()
    run_pipeline(p, np.load(rec_path), rec_path)
    p.save(tmp_path / "exp.neclab")
    (tmp_path / "exp.neclab" / "s4_cells" / "cells.tif").write_bytes(b"not a tiff")
    q, warnings = Project.load(tmp_path / "exp.neclab")
    assert q.status("cells") == MISSING and q.get("cells").problem
    assert any("Stage 4" in w for w in warnings)
    assert q.status("mask") == DONE and q.status("signals") == DONE


def test_saving_replaces_previous_save(rec_path, tmp_path):
    p = Project()
    run_pipeline(p, np.load(rec_path), rec_path)
    folder = tmp_path / "exp.neclab"
    p.save(folder)
    p.clear_stage("correlation")
    p.save(folder)
    assert not (folder / "s6_correlation").exists()
    assert not (tmp_path / "exp.neclab.saving").exists() and not (tmp_path / "exp.neclab.old").exists()


def test_refuses_to_overwrite_a_folder_that_is_not_a_project(tmp_path):
    folder = tmp_path / "important"
    folder.mkdir()
    (folder / "thesis.docx").write_text("x")
    with pytest.raises(ValidationError, match="not a NecLab project"):
        Project().save(folder)
    assert (folder / "thesis.docx").exists()


def test_load_rejects_non_projects_and_newer_versions(tmp_path):
    with pytest.raises(ValidationError, match="isn't a NecLab project"):
        Project.load(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "neclab-project", "format_version": 99}))
    with pytest.raises(ValidationError, match="newer version"):
        Project.load(tmp_path)


# --------------------------------------------------------------- single stages

def test_save_and_load_a_single_stage(rec_path, tmp_path):
    stack = np.load(rec_path)
    p = Project()
    labels, _ = run_pipeline(p, stack, rec_path)
    p.save_stage("cells", tmp_path / "my_cells")
    assert sorted(os.listdir(tmp_path / "my_cells")) == ["cells.csv", "cells.tif", "stage.json"]

    q = Project()
    q.set_recording(stack, rec_path)
    key, warnings = q.load_stage(tmp_path / "my_cells")
    assert key == "cells" and warnings == []
    assert q.status("cells") == LOADED
    assert q.get("cells").params["min_size"] == 10           # settings travel with the stage
    np.testing.assert_array_equal(q.data("cells"), labels)


def test_load_a_stage_from_a_project_folder(rec_path, tmp_path):
    p = Project()
    run_pipeline(p, np.load(rec_path), rec_path)
    p.save(tmp_path / "exp.neclab")
    assert Project.stages_in_project_folder(tmp_path / "exp.neclab") == [
        "features", "mask", "cells", "signals", "correlation"]
    q = Project()
    with pytest.raises(ValidationError, match="choose which stage"):
        q.load_stage(tmp_path / "exp.neclab")
    key, _ = q.load_stage(tmp_path / "exp.neclab", "mask")
    assert key == "mask" and q.data("mask").dtype == bool


def test_loaded_stage_is_validated_against_the_project(rec_path, tmp_path):
    p = Project()
    p.set_stage("cells", {"cells": np.ones((10, 10), np.int32)})
    p.save_stage("cells", tmp_path / "small_cells")
    q = Project()
    q.set_recording(np.load(rec_path), rec_path)               # 32 x 32 frames
    with pytest.raises(ValidationError, match="same size"):
        q.load_stage(tmp_path / "small_cells")
    assert q.status("cells") == NOT_RUN


def test_single_stage_save_guards(rec_path, tmp_path):
    p = Project()
    p.set_recording(np.load(rec_path), rec_path)
    with pytest.raises(ValidationError, match="isn't copied"):
        p.save_stage("raw", tmp_path / "raw")
    with pytest.raises(ValidationError, match="no data"):
        p.save_stage("cells", tmp_path / "c")
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "other.txt").write_text("x")
    p.set_stage("cells", {"cells": np.ones((32, 32), np.int32)})
    with pytest.raises(ValidationError, match="isn't empty"):
        p.save_stage("cells", busy)


def test_quick_checksum_detects_changes(tmp_path):
    f = tmp_path / "big.bin"
    data = np.random.RandomState(0).bytes(5 << 20)
    f.write_bytes(data)
    a = proj.quick_checksum(f)
    f.write_bytes(data[:-1] + b"\x00" if data[-1:] != b"\x00" else data[:-1] + b"\x01")
    assert proj.quick_checksum(f) != a


def test_listeners_are_notified(rec_path):
    p = Project()
    calls = []
    p.add_listener(lambda: calls.append(1))
    p.set_recording(np.load(rec_path), rec_path)
    p.clear_stage("raw")
    assert len(calls) == 2


# ------------------------------------------------- projects saved with stages 5 and 6 apart

def _make_legacy(folder, p):
    """Rewrite a saved project the way NecLab saved it before stages 5 and 6
    were merged: s5_signals (raw signals only), s6_analysis (normalized
    signals + events), s7_correlation, format version 1."""
    manifest = json.loads((folder / "manifest.json").read_text())
    sig = manifest["stages"]["signals"]
    norm = p.data("signals", "normalized_signals")
    ev = p.data("signals", "events")
    analysis_dir = folder / "s6_analysis"
    analysis_dir.mkdir()
    shutil.move(str(folder / "s5_signals" / "signals_norm.csv"), analysis_dir / "signals_norm.csv")
    shutil.move(str(folder / "s5_signals" / "events.csv"), analysis_dir / "events.csv")
    sig["files"] = {"signals": sig["files"]["signals"]}
    analysis_params = {"method": "delta_f_over_f0", "smoothing_points": 0, "lowest_points": 6,
                       "subtract_baseline": False, "peak_method": "Peak Caller",
                       "peak_params": {"rise_percent": 5, "fall_percent": 5, "max_lookback": 10,
                                       "max_lookahead": 10}}
    sig["params"] = {"cells": sig["params"]["cells"]}
    manifest["stages"]["analysis"] = {
        "stage": "analysis", "number": 6, "origin": "computed", "params": analysis_params, "source": {},
        "inputs": {"signals": sig["fingerprint"]}, "fingerprint": "f6", "updated_at": "",
        "files": {"normalized_signals": ["signals_norm.csv"], "events": ["events.csv"]},
        "folder": "s6_analysis"}
    corr = manifest["stages"]["correlation"]
    shutil.move(str(folder / "s6_correlation"), folder / "s7_correlation")
    corr["folder"], corr["number"] = "s7_correlation", 7
    corr["inputs"] = {"signals": sig["fingerprint"], "analysis": "f6"}
    manifest["format_version"] = 1
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return norm, ev


def _pipeline_with_analysis(p, rec_path):
    from neclab_core import analysis

    _labels, df = run_pipeline(p, np.load(rec_path), rec_path)
    result = analysis.Analyzer().run(df, {**analysis.DEFAULT_SETTINGS, "method": "delta_f_over_f0",
                                          "peak_method": "Peak Caller"})
    p.set_stage("signals", {"signals": df, **result.items},
                params={"cells": p.get("signals").params["cells"], **result.params})
    p.set_stage("correlation", {"correlation": correlation.correlation_matrix(result.norm_df)},
                params={"method": "pearson"})


def test_project_saved_before_the_merge_opens_with_normalization_in_stage_5(rec_path, tmp_path):
    p = Project()
    _pipeline_with_analysis(p, rec_path)
    folder = tmp_path / "old.neclab"
    p.save(folder)
    norm, ev = _make_legacy(folder, p)

    q, warnings = Project.load(folder)
    assert warnings == []
    pd.testing.assert_frame_equal(q.data("signals", "normalized_signals"), norm, check_exact=False)
    assert q.data("signals", "events")[["cell_id", "frame"]].equals(ev[["cell_id", "frame"]])
    assert q.get("signals").params["method"] == "delta_f_over_f0"
    assert q.get("signals").params["peak_method"] == "Peak Caller"
    assert "analysis" not in q.records
    assert q.status("signals") == DONE and q.status("correlation") == DONE

    # Saved again, it is in the new layout.
    q.save(tmp_path / "new.neclab")
    assert sorted(os.listdir(tmp_path / "new.neclab")) == sorted(
        ["manifest.json", "s0_raw", "s2_features", "s3_mask", "s4_cells", "s5_signals", "s6_correlation"])


def test_out_of_date_legacy_normalization_is_skipped(rec_path, tmp_path):
    p = Project()
    _pipeline_with_analysis(p, rec_path)
    folder = tmp_path / "old.neclab"
    p.save(folder)
    _make_legacy(folder, p)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["stages"]["analysis"]["inputs"]["signals"] = "something else"
    (folder / "manifest.json").write_text(json.dumps(manifest))

    q, warnings = Project.load(folder)
    assert len(warnings) == 1 and "out of date" in warnings[0]
    assert q.data("signals", "normalized_signals") is None
    assert q.data("signals") is not None


def test_legacy_stage_6_folder_loads_into_stage_5(rec_path, tmp_path):
    p = Project()
    _pipeline_with_analysis(p, rec_path)
    folder = tmp_path / "old.neclab"
    p.save(folder)
    norm, _ev = _make_legacy(folder, p)
    manifest = json.loads((folder / "manifest.json").read_text())
    (folder / "s6_analysis" / "stage.json").write_text(json.dumps(manifest["stages"]["analysis"]))

    q = Project()
    with pytest.raises(ValidationError, match="load the signals"):
        q.load_stage(folder / "s6_analysis")
    q.set_stage("signals", {"signals": p.data("signals")}, params={"cells": [1, 2]})
    key, _w = q.load_stage(folder / "s6_analysis")
    assert key == "signals"
    pd.testing.assert_frame_equal(q.data("signals", "normalized_signals"), norm, check_exact=False)
    assert q.get("signals").params["peak_method"] == "Peak Caller"
    assert Project.stages_in_project_folder(folder) == ["features", "mask", "cells", "signals", "correlation"]
