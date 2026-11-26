"""
NecLab project: the state of the whole pipeline for one recording, saved
to and loaded from a folder.

Layout of a project folder (name.neclab/):

    manifest.json          format version, save time, one entry per stage
    s0_raw/stage.json      the recording is referenced, never copied:
                           path (absolute + relative to the project), size,
                           quick checksum, OME metadata
    s1_preprocess/         stage.json, processed.tif
    s2_features/           stage.json, maps.npz
    s3_mask/               stage.json, mask.tif
    s4_cells/              stage.json, cells.tif, cells.csv
    s5_signals/            stage.json, signals_raw.csv
    s6_analysis/           stage.json, signals_norm.csv, events.csv
    s7_correlation/        stage.json, correlation.csv

Only stages that have data get a folder. Each stage.json records:
    origin      "computed" (made in NecLab) or "imported" (loaded from a file)
    source      where an imported stage came from
    params      the settings it was computed with (threshold, method, ...)
    inputs      fingerprints of the stages it was computed from
    fingerprint a hash of its own data + params
    updated_at  when it was last set

Status of a stage (Project.status):
    "not run"      no data
    "done"         computed in NecLab and up to date
    "loaded"       imported from a file and up to date
    "out of date"  something it was computed from has changed since
    "missing"      its data couldn't be loaded (e.g. recording not found)

A stage folder is self-contained (stage.json + its files), so a single
stage can be saved on its own and loaded into another project
(save_stage / load_stage).
"""

import datetime
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import stages
from .validation import ValidationError

FORMAT = "neclab-project"
FORMAT_VERSION = 1
PROJECT_EXTENSION = ".neclab"
MANIFEST = "manifest.json"
STAGE_FILE = "stage.json"

NOT_RUN = "not run"
DONE = "done"
LOADED = "loaded"
OUT_OF_DATE = "out of date"
MISSING = "missing"

# Items held by each stage.
STAGE_ITEMS = {
    "raw": ("stack",),
    "preprocess": ("processed_stack",),
    "features": ("maps",),
    "mask": ("mask",),
    "cells": ("cells",),
    "signals": ("signals",),
    "analysis": ("normalized_signals", "events"),
    "correlation": ("correlation",),
}

# Stages each stage is computed from. A stage is out of date when one of
# the inputs that existed when it was set has changed since (inputs that
# appear later don't make it out of date).
DEPENDENCIES = {
    "raw": (),
    "preprocess": ("raw",),
    "features": ("preprocess", "raw"),
    "mask": ("features",),
    "cells": ("mask", "features"),
    "signals": ("cells", "preprocess", "raw"),
    "analysis": ("signals",),
    "correlation": ("analysis", "signals"),
}

STAGE_KEYS = tuple(s.key for s in stages.STAGES)


def stage_folder_name(key):
    return f"s{stages.stage(key).number}_{key}"


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------ hashing

def quick_checksum(path, chunk=1 << 20):
    """Fast checksum of a (possibly very large) file: SHA-256 of its size
    and of 1 MB from its start, middle and end. Detects a different or
    modified file without reading gigabytes."""
    size = os.path.getsize(path)
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        for offset in sorted({0, max(0, size // 2 - chunk // 2), max(0, size - chunk)}):
            f.seek(offset)
            h.update(f.read(chunk))
    return h.hexdigest()


def _hash_data(data):
    h = hashlib.sha1()
    if data is None:
        return ""
    if isinstance(data, np.ndarray):
        h.update(str((data.shape, data.dtype.str)).encode())
        h.update(np.ascontiguousarray(data).tobytes())
    elif isinstance(data, pd.DataFrame):
        h.update(json.dumps([str(c) for c in data.columns]).encode())
        h.update(json.dumps([str(i) for i in data.index]).encode())
        h.update(pd.util.hash_pandas_object(data, index=False).to_numpy().tobytes())
    elif isinstance(data, dict):
        for k in sorted(data):
            h.update(str(k).encode())
            h.update(_hash_data(data[k]).encode())
    else:
        h.update(repr(data).encode())
    return h.hexdigest()


def _fingerprint(item_hashes, params):
    return hashlib.sha1(json.dumps(
        _json_safe({"items": item_hashes, "params": params}), sort_keys=True).encode()).hexdigest()


def _json_safe(value):
    """Convert numpy scalars/arrays etc. so json.dumps never fails."""
    return json.loads(json.dumps(value, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


# ------------------------------------------------------------ records

@dataclass
class StageRecord:
    key: str
    origin: str                       # "computed" or "imported"
    items: dict                       # {item_key: data}
    params: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)   # {dep_key: fingerprint at set time}
    fingerprint: str = ""
    updated_at: str = ""
    available: bool = True            # False when the data couldn't be loaded
    problem: str = ""                 # why it isn't available

    def describe_source(self):
        if self.key == "raw" and self.source.get("path"):
            return os.path.basename(self.source["path"])
        if self.origin == "imported" and self.source.get("path"):
            return f"imported: {os.path.basename(str(self.source['path']).rstrip(os.sep))}"
        return self.origin

    def to_json(self):
        return _json_safe({
            "stage": self.key,
            "number": stages.stage(self.key).number,
            "origin": self.origin,
            "params": self.params,
            "source": self.source,
            "inputs": self.inputs,
            "fingerprint": self.fingerprint,
            "updated_at": self.updated_at,
        })


class Project:
    """In-memory pipeline state for one recording (see the module docstring)."""

    def __init__(self):
        self.records = {}
        self.ui = {}                  # view state worth keeping (e.g. selected cells)
        self.path = None              # folder the project was last saved to / loaded from
        self.modified = False
        self._listeners = []

    # ------------------------------------------------------- listeners

    def add_listener(self, fn):
        """fn() is called after every change (for views that show the status)."""
        self._listeners.append(fn)

    def remove_listener(self, fn):
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _changed(self):
        self.modified = True
        for fn in list(self._listeners):
            try:
                fn()
            except Exception:
                pass

    # ------------------------------------------------------- name/state

    @property
    def name(self):
        if not self.path:
            return "Untitled"
        base = os.path.basename(str(self.path).rstrip("/\\"))
        return base[: -len(PROJECT_EXTENSION)] if base.endswith(PROJECT_EXTENSION) else base

    def is_empty(self):
        return not self.records

    # ------------------------------------------------------- set / get

    def _current_fingerprint(self, key):
        rec = self.records.get(key)
        return rec.fingerprint if rec else None

    def set_stage(self, key, items, params=None, origin="computed", source=None):
        """Store a stage's data (replacing what was there) and record the
        fingerprints of its inputs, so later changes upstream mark it out
        of date. `items` is {item_key: data} for the stage's items."""
        if key not in STAGE_ITEMS:
            raise KeyError(f"Unknown stage {key!r}")
        unknown = set(items) - set(STAGE_ITEMS[key])
        if unknown:
            raise KeyError(f"Stage {key!r} has no item(s) {sorted(unknown)}")
        params = dict(params or {})
        source = dict(source or {})
        item_hashes = {k: _hash_data(v) for k, v in items.items()}
        if key == "raw" and source.get("quick_sha256"):
            item_hashes = {"stack": source["quick_sha256"]}  # the file identifies the recording
        fingerprint = _fingerprint(item_hashes, params)
        inputs = {dep: self._current_fingerprint(dep) for dep in DEPENDENCIES[key]
                  if self._current_fingerprint(dep)}
        self.records[key] = StageRecord(key, origin, dict(items), params, source, inputs,
                                        fingerprint, _now())
        self._changed()
        return self.records[key]

    def set_ui(self, key, value):
        """Store view state worth keeping with the project (e.g. the
        selected cells); counts as an unsaved change."""
        value = _json_safe(value)
        if self.ui.get(key) != value:
            self.ui[key] = value
            self._changed()

    def clear_stage(self, key):
        if self.records.pop(key, None) is not None:
            self._changed()

    def get(self, key):
        return self.records.get(key)

    def data(self, key, item=None):
        """A stage's data: the single item, or `item`; None if absent."""
        rec = self.records.get(key)
        if rec is None or not rec.available:
            return None
        if item is None:
            item = STAGE_ITEMS[key][0]
        return rec.items.get(item)

    def stack(self):
        """The recording to analyse: the processed one if any, else the raw one."""
        processed = self.data("preprocess")
        return processed if processed is not None else self.data("raw")

    def context(self):
        maps = self.data("features")
        return stages.Context(stack=self.stack(), maps=maps, mask=self.data("mask"),
                              labels=self.data("cells"))

    # ------------------------------------------------------- status

    def status(self, key, _seen=None):
        rec = self.records.get(key)
        if rec is None:
            return NOT_RUN
        if not rec.available:
            return MISSING
        _seen = _seen or set()
        for dep, fp in rec.inputs.items():
            dep_rec = self.records.get(dep)
            if dep_rec is None:
                continue  # input removed since: nothing to compare with
            if dep_rec.fingerprint != fp:
                return OUT_OF_DATE
            if dep not in _seen and self.status(dep, _seen | {key}) == OUT_OF_DATE:
                return OUT_OF_DATE
        return DONE if rec.origin == "computed" else LOADED

    def summary(self):
        """One row per stage: number, key, title, status, source, updated."""
        rows = []
        for s in stages.STAGES:
            rec = self.records.get(s.key)
            rows.append({
                "number": s.number, "key": s.key, "title": s.title,
                "status": self.status(s.key),
                "source": rec.describe_source() if rec else "",
                "updated": rec.updated_at.replace("T", " ") if rec else "",
                "problem": rec.problem if rec else "",
            })
        return rows

    # ------------------------------------------------------- raw file

    @staticmethod
    def raw_source(path, metadata=None):
        """`source` dict for set_stage("raw", ...) from the recording's file."""
        path = os.path.abspath(str(path))
        return {"path": path, "size": os.path.getsize(path),
                "quick_sha256": quick_checksum(path), "metadata": _json_safe(metadata or {})}

    def set_recording(self, stack, path, metadata=None, origin="imported"):
        return self.set_stage("raw", {"stack": stack}, origin=origin,
                              source=self.raw_source(path, metadata))

    # ------------------------------------------------------- save

    def _write_stage(self, key, folder):
        """Write one stage (stage.json + its files) into `folder`."""
        rec = self.records[key]
        os.makedirs(folder, exist_ok=True)
        entry = rec.to_json()
        files = {}
        if key != "raw" and rec.available:
            for item_key, data in rec.items.items():
                if data is None:
                    continue
                written = stages.save_item(item_key, data, os.path.join(folder, stages.item(item_key).default_name))
                files[item_key] = [os.path.basename(p) for p in written]
        entry["files"] = files
        if key == "raw":
            entry["source"] = dict(entry["source"])
            raw_path = rec.source.get("path")
            if raw_path:
                # Relative to the project folder (the parent of this stage folder),
                # so the project still finds the recording if both are moved together.
                project_dir = os.path.dirname(os.path.abspath(folder))
                try:
                    entry["source"]["relative_path"] = os.path.relpath(raw_path, project_dir)
                except ValueError:  # different drive on Windows
                    pass
        with open(os.path.join(folder, STAGE_FILE), "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
        return entry

    def save(self, folder):
        """Save the whole project to `folder` (created or replaced).

        Written to a temporary folder first and swapped in at the end, so a
        failure never leaves a half-written project behind.
        """
        folder = os.path.abspath(str(folder))
        if os.path.exists(folder) and not (os.path.isdir(folder) and
                                           (not os.listdir(folder) or
                                            os.path.exists(os.path.join(folder, MANIFEST)))):
            raise ValidationError(f"{folder} exists and is not a NecLab project; choose another name.")
        tmp = folder + ".saving"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        try:
            manifest = {"format": FORMAT, "format_version": FORMAT_VERSION, "saved_at": _now(),
                        "stages": {}, "ui": _json_safe(self.ui)}
            for key in STAGE_KEYS:
                if key not in self.records:
                    continue
                sub = stage_folder_name(key)
                # tmp sits next to the final folder, so paths relative to it are the same
                entry = self._write_stage(key, os.path.join(tmp, sub))
                entry["folder"] = sub
                manifest["stages"][key] = entry
            with open(os.path.join(tmp, MANIFEST), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        old = folder + ".old"
        if os.path.exists(folder):
            if os.path.exists(old):
                shutil.rmtree(old)
            os.rename(folder, old)
        os.rename(tmp, folder)
        shutil.rmtree(old, ignore_errors=True)
        self.path = folder
        self.modified = False
        for fn in list(self._listeners):
            fn()

    def save_stage(self, key, folder):
        """Save a single stage to `folder` (stage.json + its files)."""
        if key not in self.records:
            raise ValidationError(f"Stage {stages.stage(key).number} has no data to save.")
        if key == "raw":
            raise ValidationError("The recording isn't copied: save the project, or keep the recording file.")
        if os.path.exists(folder) and os.listdir(folder) and not os.path.exists(os.path.join(folder, STAGE_FILE)):
            raise ValidationError(f"{folder} isn't empty; choose a new folder.")
        self._write_stage(key, folder)

    # ------------------------------------------------------- load

    @staticmethod
    def _read_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ValidationError(f"Can't read {os.path.basename(path)}: {e}")

    def _load_items(self, key, folder, entry, context):
        items, warnings = {}, []
        for item_key, names in (entry.get("files") or {}).items():
            if item_key not in STAGE_ITEMS[key]:
                warnings.append(f"Unknown item '{item_key}' in stage {key} ignored.")
                continue
            loaded = stages.load_item(item_key, os.path.join(folder, names[0]), context)
            items[item_key] = loaded.data
            warnings += loaded.warnings
        return items, warnings

    def _locate_raw(self, entry, project_folder):
        src = entry.get("source") or {}
        candidates = []
        if src.get("relative_path"):
            candidates.append(os.path.normpath(os.path.join(project_folder, src["relative_path"])))
        if src.get("path"):
            candidates.append(src["path"])
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    def _restore_raw(self, entry, path, warnings):
        """Load the recording for a raw entry; returns the record."""
        src = dict(entry.get("source") or {})
        rec = StageRecord("raw", entry.get("origin", "imported"), {}, entry.get("params") or {}, src,
                          entry.get("inputs") or {}, entry.get("fingerprint", ""), entry.get("updated_at", ""))
        if path is None:
            rec.available = False
            rec.problem = f"Recording not found: {src.get('path', '?')}"
            warnings.append(rec.problem + ". Locate it to extract signals; the other stages are loaded.")
            return rec
        loaded = stages.load_item("stack", path)
        rec.items = {"stack": loaded.data}
        checksum = quick_checksum(path)
        if src.get("quick_sha256") and checksum != src["quick_sha256"]:
            warnings.append(f"The recording {os.path.basename(path)} has changed since the project was saved; "
                            "the stages computed from it are marked out of date.")
            rec.fingerprint = _fingerprint({"stack": checksum}, rec.params)
        rec.source = {**src, "path": os.path.abspath(path), "quick_sha256": checksum,
                      "size": os.path.getsize(path)}
        rec.source.pop("relative_path", None)
        return rec

    @classmethod
    def load(cls, folder):
        """Open a project folder. Returns (project, warnings).

        Stages whose files can't be loaded are kept as "missing" with the
        reason, so one bad file doesn't prevent opening the rest.
        """
        folder = os.path.abspath(str(folder))
        manifest_path = os.path.join(folder, MANIFEST)
        if not os.path.isfile(manifest_path):
            raise ValidationError(f"{os.path.basename(folder)} isn't a NecLab project (no {MANIFEST}).")
        manifest = cls._read_json(manifest_path)
        if manifest.get("format") != FORMAT:
            raise ValidationError(f"{MANIFEST} isn't a NecLab project manifest.")
        if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
            raise ValidationError("This project was saved by a newer version of NecLab; update NecLab to open it.")

        project = cls()
        warnings = []
        entries = manifest.get("stages") or {}
        for key in STAGE_KEYS:
            entry = entries.get(key)
            if entry is None:
                continue
            if key == "raw":
                try:
                    project.records["raw"] = project._restore_raw(entry, project._locate_raw(entry, folder), warnings)
                except (ValidationError, ValueError, OSError) as e:
                    rec = project._restore_raw(entry, None, [])
                    rec.problem = f"Recording can't be loaded: {e}"
                    project.records["raw"] = rec
                    warnings.append(rec.problem)
                continue
            sub = os.path.join(folder, entry.get("folder") or stage_folder_name(key))
            rec = StageRecord(key, entry.get("origin", "computed"), {}, entry.get("params") or {},
                              entry.get("source") or {}, entry.get("inputs") or {},
                              entry.get("fingerprint", ""), entry.get("updated_at", ""))
            try:
                rec.items, w = project._load_items(key, sub, entry, project.context())
                warnings += [f"Stage {stages.stage(key).number}: {x}" for x in w]
            except (ValidationError, ValueError, OSError) as e:
                rec.available = False
                rec.problem = str(e)
                warnings.append(f"Stage {stages.stage(key).number} ({stages.stage(key).title}) "
                                f"couldn't be loaded: {e}")
            project.records[key] = rec
        project.ui = manifest.get("ui") or {}
        project.path = folder
        project.modified = False
        return project, warnings

    def relocate_recording(self, path):
        """Point the project at the recording's new location (after it moved).
        Returns a list of warnings (e.g. if it's a different file)."""
        warnings = []
        entry = self.records["raw"].to_json() if "raw" in self.records else {"origin": "imported", "source": {}}
        rec = self._restore_raw(entry, path, warnings)
        self.records["raw"] = rec
        self._changed()
        return warnings

    @staticmethod
    def stages_in_project_folder(folder):
        """Stage keys saved in a project folder (for picking one to load)."""
        manifest = Project._read_json(os.path.join(folder, MANIFEST))
        return [k for k in STAGE_KEYS if k in (manifest.get("stages") or {}) and k != "raw"]

    def load_stage(self, folder, key=None):
        """Load one stage from a stage folder (one with stage.json) or from a
        project folder (then `key` says which stage) into this project, as
        an imported stage. Its data is validated against what is loaded
        here. Returns (stage key, warnings)."""
        folder = os.path.abspath(str(folder))
        if os.path.isfile(os.path.join(folder, MANIFEST)):
            if key is None:
                raise ValidationError("This is a project folder: choose which stage to load from it.")
            manifest = self._read_json(os.path.join(folder, MANIFEST))
            entry = (manifest.get("stages") or {}).get(key)
            if entry is None:
                raise ValidationError(f"The project has no stage {stages.stage(key).number}.")
            stage_dir = os.path.join(folder, entry.get("folder") or stage_folder_name(key))
        else:
            stage_dir = folder
        stage_json = os.path.join(stage_dir, STAGE_FILE)
        if not os.path.isfile(stage_json):
            raise ValidationError(f"{os.path.basename(stage_dir)} isn't a saved NecLab stage (no {STAGE_FILE}).")
        entry = self._read_json(stage_json)
        stage_key = entry.get("stage")
        if stage_key not in STAGE_ITEMS:
            raise ValidationError(f"Unknown stage {stage_key!r} in {STAGE_FILE}.")
        if key is not None and stage_key != key:
            raise ValidationError(f"This folder holds stage {entry.get('number')}, not stage {stages.stage(key).number}.")
        if stage_key == "raw":
            raise ValidationError("Open the recording itself (File → Open Recording).")
        ctx = self.context()
        # Data of this same stage doesn't constrain its replacement.
        if stage_key == "features":
            ctx.maps = None
        elif stage_key == "mask":
            ctx.mask = None
        elif stage_key == "cells":
            ctx.labels = None
        items, warnings = self._load_items(stage_key, stage_dir, entry, ctx)
        params = dict(entry.get("params") or {})
        self.set_stage(stage_key, items, params=params, origin="imported",
                       source={"path": stage_dir, "original_origin": entry.get("origin")})
        return stage_key, warnings
