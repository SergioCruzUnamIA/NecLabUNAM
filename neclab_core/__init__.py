"""
neclab_core - NecLab's analysis code, with no user interface.

Every function here takes plain data (NumPy arrays, pandas DataFrames,
file paths) and returns plain data. Nothing imports tkinter, opens a
dialog or draws a plot, so the same code runs from the app, a notebook,
a batch script or a test.

Modules:
    stages        the pipeline stages, what each one reads/writes, and
                  load_item()/save_item() to enter or leave any stage
    formats       file readers/writers for every stage
    validation    checks on loaded data, alone and against other stages
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
    "stages",
    "formats",
    "validation",
    "display",
    "variability",
    "segmentation",
    "signals",
    "correlation",
]
