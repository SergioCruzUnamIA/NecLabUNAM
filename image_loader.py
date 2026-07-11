# image_loader.py

from tkinter.filedialog import askopenfilename
from pyometiff import OMETIFFReader
from PIL import Image

def pick_ometiff_file():
    """
    Abre un diálogo para seleccionar un archivo OME-TIFF y retorna su ruta
    (o None si se cancela). Separado de la lectura del archivo para que
    esta última pueda correr en un hilo aparte (ver read_ometiff_image).
    """
    return askopenfilename(title="Abrir archivo OME-TIFF", filetypes=[("OME-TIFF files", "*.tif"), ("All files", "*.*")])


def read_ometiff_image(filename):
    """
    Lee un archivo OME-TIFF ya elegido y retorna la imagen y su metadata.
    No toca ningún widget de Tk, por lo que es seguro llamarla desde un
    hilo aparte (ver progress_utils.run_with_progress_window).
    """
    reader = OMETIFFReader(fpath=filename)
    img, metadata, xml_metadata = reader.read()
    return img, metadata, xml_metadata


def load_ometiff_image():
    """
    Abre un diálogo para seleccionar un archivo OME-TIFF y retorna la imagen y su metadata.
    """
    filename = pick_ometiff_file()
    if not filename:
        return None, None, None
    return read_ometiff_image(filename)

def process_image_slice(img_array, slice_index, display_width, display_height):
    """
    Convierte la capa indicada en una imagen procesada para visualización.
    """
    pil_img = Image.fromarray(img_array[slice_index, :, :])
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    width_pil, height_pil = pil_img.size
    ratio = min(display_width / (width_pil * 1.5), display_height / (height_pil * 1.5))
    pil_img = pil_img.resize((int(width_pil * ratio), int(height_pil * ratio)), Image.LANCZOS)
    return pil_img
