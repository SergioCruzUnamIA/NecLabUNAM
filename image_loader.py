# image_loader.py
"""File dialogs for loading recordings. Reading and validation are in
neclab_core (formats / validation / stages)."""

from tkinter.filedialog import askopenfilename
from PIL import Image

from neclab_core import stages


def ask_recording_path(parent=None, title="Open Recording"):
    """Dialog for a recording file (.ome.tif, .tif, .npy). Returns a path or ''."""
    return askopenfilename(parent=parent, title=title, filetypes=stages.open_filetypes("stack"))


def load_recording(path):
    """Read and validate a recording. Returns (stack, metadata).
    Raises neclab_core.validation.ValidationError for unusable files."""
    loaded = stages.load_item("stack", path)
    return loaded.data, loaded.metadata


def load_ometiff_image():
    """
    Opens a dialog to select a recording and returns (image, metadata, filename).
    Returns (None, None, None) if the user cancels.
    """
    filename = ask_recording_path()
    if not filename:
        return None, None, None
    img, metadata = load_recording(filename)
    return img, metadata, filename


def process_image_slice(img_array, slice_index, display_width, display_height):
    """
    Converts the given slice into a processed image for display.
    """
    pil_img = Image.fromarray(img_array[slice_index, :, :])
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    width_pil, height_pil = pil_img.size
    ratio = min(display_width / (width_pil * 1.5), display_height / (height_pil * 1.5))
    pil_img = pil_img.resize((int(width_pil * ratio), int(height_pil * ratio)), Image.LANCZOS)
    return pil_img
