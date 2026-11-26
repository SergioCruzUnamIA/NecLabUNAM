"""
The pipeline stages and the data each one produces.

Every stage can be entered by loading its data from a file instead of
computing it; everything after it then works as usual. This module is the
one place that says, for each piece of data ("item"):

    - which stage it belongs to,
    - which file types can hold it (for open/save dialogs),
    - how to read and write it (formats.py),
    - how to check it against what is already loaded (validation.py).

The UI and the project file (a later step) go through load_item() and
save_item() instead of calling the readers directly.

Stages:
    0 raw          recording as acquired
    1 preprocess   processed recording
    2 features     per-pixel variability maps
    3 mask         candidate-pixel mask
    4 cells        label image (permanent cell IDs)
    5 signals      raw signal of each cell
    6 analysis     normalized signals and detected events
    7 correlation  correlation matrix between signals
"""

from dataclasses import dataclass, field

import numpy as np

from . import formats, validation


@dataclass(frozen=True)
class Stage:
    number: int
    key: str
    title: str
    description: str


STAGES = (
    Stage(0, "raw", "Raw data", "The recording as acquired (frames x height x width)."),
    Stage(1, "preprocess", "Preprocessing", "The recording after corrections and filters."),
    Stage(2, "features", "Pixel feature maps", "Variability of every pixel over time."),
    Stage(3, "mask", "Candidate pixel mask", "Pixels that may belong to an active cell."),
    Stage(4, "cells", "Cell detection", "Label image: each cell's pixels hold its permanent ID."),
    Stage(5, "signals", "Signal extraction", "Mean intensity of each cell in every frame."),
    Stage(6, "analysis", "Normalization and events", "Normalized signals and detected events (peaks)."),
    Stage(7, "correlation", "Correlation", "Correlation between the signals."),
)
_STAGE_BY_KEY = {s.key: s for s in STAGES}


def stage(key_or_number):
    """Stage by key ("cells") or number (4)."""
    if isinstance(key_or_number, int):
        return STAGES[key_or_number]
    return _STAGE_BY_KEY[key_or_number]


@dataclass(frozen=True)
class Item:
    key: str
    stage: str
    title: str
    open_types: tuple            # ((label, "*.a *.b"), ...) for open dialogs
    save_types: tuple = ()       # same, for save dialogs; empty = can't be saved here
    default_name: str = ""       # suggested file name when saving
    notes: str = field(default="", compare=False)


_TABLE = ("Tables", "*.csv *.xlsx")

ITEMS = {
    "stack": Item(
        "stack", "raw", "Recording",
        open_types=(("Recordings", "*.ome.tif *.ome.tiff *.tif *.tiff *.npy"),
                    ("OME-TIFF", "*.ome.tif *.ome.tiff"), ("TIFF", "*.tif *.tiff"), ("NumPy", "*.npy")),
        notes="The raw file is referenced, not copied."),
    "processed_stack": Item(
        "processed_stack", "preprocess", "Processed recording",
        open_types=(("Recordings", "*.tif *.tiff *.npy"),),
        save_types=(("TIFF", "*.tif"), ("NumPy", "*.npy")),
        default_name="processed.tif"),
    "maps": Item(
        "maps", "features", "Variability map(s)",
        open_types=(("Maps", "*.npz *.npy *.tif *.tiff"), ("NumPy archive (several maps)", "*.npz"),
                    ("Single map", "*.npy *.tif *.tiff")),
        save_types=(("NumPy archive", "*.npz"), ("NumPy", "*.npy"), ("TIFF (32-bit float)", "*.tif")),
        default_name="maps.npz"),
    "mask": Item(
        "mask", "mask", "Candidate pixel mask",
        open_types=(("Masks", "*.tif *.tiff *.npy"),),
        save_types=(("TIFF (0/255)", "*.tif"), ("NumPy (bool)", "*.npy")),
        default_name="mask.tif"),
    "cells": Item(
        "cells", "cells", "Cells",
        open_types=(("Cells", "*.tif *.tiff *.npy *.roi *.zip *.txt"),
                    ("Label image", "*.tif *.tiff *.npy"),
                    ("ImageJ ROIs", "*.roi *.zip"),
                    ("Coordinates (older NecLab)", "*.txt")),
        save_types=(("Label image (TIFF)", "*.tif"), ("Label image (NumPy)", "*.npy")),
        default_name="cells.tif",
        notes="A cell table (.csv) with the same name is written next to the label image."),
    "signals": Item(
        "signals", "signals", "Signals",
        open_types=(("Signals", "*.csv *.xlsx *.npy"),),
        save_types=(("CSV", "*.csv"), ("Excel", "*.xlsx"), ("NumPy", "*.npy")),
        default_name="signals_raw.csv"),
    "normalized_signals": Item(
        "normalized_signals", "analysis", "Normalized signals",
        open_types=(("Signals", "*.csv *.xlsx *.npy"),),
        save_types=(("CSV", "*.csv"), ("Excel", "*.xlsx"), ("NumPy", "*.npy")),
        default_name="signals_norm.csv"),
    "events": Item(
        "events", "analysis", "Events (peaks)",
        open_types=(_TABLE,),
        save_types=(("CSV", "*.csv"), ("Excel", "*.xlsx")),
        default_name="events.csv"),
    "correlation": Item(
        "correlation", "correlation", "Correlation matrix",
        open_types=(_TABLE,),
        save_types=(("CSV", "*.csv"), ("Excel", "*.xlsx")),
        default_name="correlation.csv"),
}


def item(key):
    return ITEMS[key]


def items_for_stage(stage_key):
    return [it for it in ITEMS.values() if it.stage == stage_key]


def open_filetypes(key):
    """File-dialog filetypes for opening an item (+ "All files")."""
    return list(ITEMS[key].open_types) + [("All files", "*.*")]


def save_filetypes(key):
    return list(ITEMS[key].save_types) + [("All files", "*.*")]


# ------------------------------------------------------------- context

@dataclass
class Context:
    """What is already loaded, used to check new data against it.

    Any field can be None. frame_shape and n_frames are taken from the
    recording when there is one, otherwise from the other images.
    """
    stack: object = None
    maps: dict = None
    mask: object = None
    labels: object = None

    @property
    def frame_shape(self):
        if self.stack is not None:
            return tuple(np.asarray(self.stack).shape[1:])
        for image in (self.labels, self.mask):
            if image is not None:
                return tuple(np.asarray(image).shape)
        if self.maps:
            return tuple(np.asarray(next(iter(self.maps.values()))).shape)
        return None

    @property
    def n_frames(self):
        return None if self.stack is None else int(np.asarray(self.stack).shape[0])

    @property
    def cell_ids(self):
        if self.labels is None:
            return None
        return [int(i) for i in np.unique(self.labels) if i > 0]


@dataclass
class Loaded:
    """Result of load_item: the data, any extra information and warnings."""
    key: str
    data: object
    path: str
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def load_item(key, path, context=None, **options):
    """Read and validate one item; raises validation.ValidationError.

    options:
        signals / normalized_signals: time_column ("auto"/True/False), sheet_name
        correlation: sheet_name
    """
    ctx = context or Context()
    path = str(path)
    if key in ("stack", "processed_stack"):
        stack, metadata = formats.load_stack(path)
        if ctx.frame_shape is not None:
            validation.check_frame_shape(stack.shape[1:], ctx.frame_shape, "recording's frame size")
        return Loaded(key, stack, path, metadata=metadata)
    if key == "maps":
        maps, warnings = formats.load_maps(path, ctx.frame_shape)
        return Loaded(key, maps, path, warnings)
    if key == "mask":
        mask, warnings = formats.load_mask(path, ctx.frame_shape)
        return Loaded(key, mask, path, warnings)
    if key == "cells":
        labels = formats.load_cells(path, shape=ctx.frame_shape)
        warnings = validation.check_labels(labels, ctx.frame_shape)
        return Loaded(key, labels, path, warnings)
    if key in ("signals", "normalized_signals"):
        df, warnings = formats.read_signals(path, n_frames=ctx.n_frames, cell_ids=ctx.cell_ids, **options)
        return Loaded(key, df, path, warnings, metadata={"normalized": key == "normalized_signals"})
    if key == "events":
        df, warnings = formats.read_events(path, n_frames=ctx.n_frames, cell_ids=ctx.cell_ids)
        return Loaded(key, df, path, warnings)
    if key == "correlation":
        df, warnings = formats.read_correlation(path, **options)
        return Loaded(key, df, path, warnings)
    raise KeyError(f"Unknown item {key!r}")


def save_item(key, data, path):
    """Write one item in its stage's format. Returns the list of files written."""
    path = str(path)
    if key == "stack":
        raise validation.ValidationError("The raw recording is never copied; it is referenced by its path.")
    if key == "processed_stack":
        formats.save_stack(data, path)
        return [path]
    if key == "maps":
        formats.save_maps(data, path)
        return [path]
    if key == "mask":
        formats.save_mask(data, path)
        return [path]
    if key == "cells":
        return list(formats.save_cells(data, path))
    if key in ("signals", "normalized_signals"):
        validation.check_signals(data)
        formats.write_signals(data, path)
        return [path]
    if key == "events":
        formats.write_events(data, path)
        return [path]
    if key == "correlation":
        formats.write_correlation(data, path)
        return [path]
    raise KeyError(f"Unknown item {key!r}")
