"""
neclab_core - NecLab's analysis code, with no user interface.

Every function here takes plain data (NumPy arrays, pandas DataFrames,
file paths) and returns plain data. Nothing imports tkinter, opens a
dialog or draws a plot, so the same code runs from the app, a notebook,
a batch script or a test.

Modules, in pipeline order:
    formats       reading/writing files (OME-TIFF, signals tables, cells)
    display       display-only adjustments (never change the analysis data)
    variability   per-pixel variability maps (the 7 methods) and sharpening
    segmentation  candidate-pixel mask -> cells, as a label image
    signals       per-cell signal extraction from the stack
    correlation   correlation matrices and hierarchical-clustering linkage

Cells are represented everywhere by a *label image*: a 2D integer array the
size of one frame, where 0 is background and every other value is the
permanent ID of the cell that pixel belongs to.
"""

__all__ = [
    "formats",
    "display",
    "variability",
    "segmentation",
    "signals",
    "correlation",
]
