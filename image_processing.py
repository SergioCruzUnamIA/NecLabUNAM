# image_processing.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def auto_contrast(img_array, cutoff=2, ignore=2):
    """
    Aplica autocontraste a cada capa de la pila de imágenes.
    Trabaja sobre una COPIA para no modificar los datos originales,
    ya que el contraste es solo para visualización, no procesamiento.

    Args:
        img_array (numpy.ndarray): Pila de imágenes con forma (num_slices, height, width).
        cutoff (int): Porcentaje de píxeles que se recortan de cada extremo del histograma.
        ignore (int): Número de píxeles a ignorar en los extremos del histograma.
    
    Returns:
        numpy.ndarray: Copia del array con el autocontraste aplicado.
    """
    result = img_array.copy()
    for i in range(result.shape[0]):
        im_pil = Image.fromarray(result[i, :, :])
        if im_pil.mode != 'RGB':
            im_pil = im_pil.convert('RGB')
        im2 = ImageOps.autocontrast(im_pil, cutoff=cutoff, ignore=ignore).convert('L')
        result[i, :, :] = np.array(im2)
    return result


def show_histogram(img_array):
    """
    Muestra el histograma de la varianza de la pila de imágenes.
    Calcula la varianza a lo largo del eje 0 (pila) y luego grafica el histograma en 2D.

    Args:
        img_array (numpy.ndarray): Pila de imágenes con forma (num_slices, height, width).
    """
    var_im = np.var(img_array, axis=0)
    plt.hist(var_im.ravel(), bins=50)
    plt.title("Histogram of Variance")
    plt.xlabel("Varianza de píxel (flattened)")
    plt.ylabel("Frecuencia")
    plt.show()


def binarize_variance(img_array, threshold=150):
    """
    Calcula la varianza a lo largo de la pila (eje 0) y crea una imagen binaria
    en 2D usando el umbral indicado.

    Args:
        img_array (numpy.ndarray): Pila de imágenes con forma (num_slices, height, width).
        threshold (int): Umbral para la binarización de la varianza.
    
    Returns:
        PIL.Image: Imagen binaria (RGB) resultante.
    """
    # Calcula la varianza por cada píxel a lo largo del eje 0
    var_im = np.var(img_array, axis=0)
    # Convierte a un vector 1D
    var_im_flat = var_im.ravel()
    # Genera la máscara binaria según el umbral
    bin_mask = (var_im_flat > threshold).astype(np.uint8) * 255
    # Reconvierte a 2D
    bin_mask_2d = bin_mask.reshape(var_im.shape)
    # Crea la imagen PIL
    pil_img = Image.fromarray(bin_mask_2d)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return pil_img

def threshold_image_pil(image_2d, threshold=200):
    """
    Aplica un umbral simple a una imagen 2D (una capa) y retorna la imagen binaria como PIL.Image.

    Args:
        image_2d (numpy.ndarray): Imagen 2D (altura x anchura).
        threshold (int): Valor del umbral. Los píxeles con valor > threshold se convierten en 255 (blanco)
                         y los demás en 0 (negro).

    Returns:
        PIL.Image: Imagen binaria en modo RGB.
    """
    # Aplana la imagen a un vector 1D
    arr_flat = image_2d.ravel()
    # Aplica el umbral y multiplica por 255 para obtener valores 0 o 255
    bin_mask = (arr_flat > threshold).astype(np.uint8) * 255
    # Reconstruye la imagen en su forma original (2D)
    bin_mask_2d = bin_mask.reshape(image_2d.shape)
    # Crea la imagen PIL
    pil_img = Image.fromarray(bin_mask_2d)
    # La imagen debe estar en modo RGB para la compatibilidad con Tkinter
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return pil_img
