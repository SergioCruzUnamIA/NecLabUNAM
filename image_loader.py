# image_loader.py
"""File dialogs for loading images. The reading itself is in neclab_core.formats."""

from tkinter.filedialog import askopenfilename
from PIL import Image

from neclab_core import formats


def load_ometiff_image():
    """
    Opens a dialog to select an OME-TIFF file and returns (image, metadata, filename).
    Returns (None, None, None) if the user cancels.
    """
    filename = askopenfilename(title="Open OME-TIFF file",
                               filetypes=[("OME-TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
    if not filename:
        return None, None, None
    img, metadata = formats.load_ometiff(filename)
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
