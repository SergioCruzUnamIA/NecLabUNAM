import os
os.environ["OMP_NUM_THREADS"] = "1"  #limita num de threads
from pyometiff import OMETIFFReader
from image_loader import load_ometiff_image, process_image_slice
from tkinter import *
from visualization_helpers import *
from image_processing import *
import tkinter as tk
import numpy as np
from tkinter import PhotoImage, Grid, filedialog
from PIL import Image, ImageTk, ImageOps
import cv2
from tkinter import filedialog as fd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from functools import partial

# Initialize the main window
window = tk.Tk()
images = {}
data = None
canvas = None
corr = None

window.title("NecLab")
window.tk.call('tk', 'windowingsystem') 
window.option_add('*tearOff', FALSE)

# Get screen width and height of display and set window size
width = window.winfo_screenwidth() * 0.8
height = window.winfo_screenheight() * 0.8
window.geometry("%dx%d" % (width, height))

# Create menu bar
barra_menus = Menu(window)
window.config(menu=barra_menus)

# Create "Archivo" menu
menu_archivo = tk.Menu(barra_menus, tearoff=False)
barra_menus.add_cascade(menu=menu_archivo, label="Archivo")

# Create "Imagen" menu
menu_imagen = tk.Menu(barra_menus, tearoff=False)
barra_menus.add_cascade(menu=menu_imagen, label="Imagen")

# Create "Picos" menu
menu_picos = Menu(barra_menus)
barra_menus.add_cascade(menu=menu_picos, label="Visualización")

# Initialize global variables for image data
img_original = []
img_array = []

def archivo_nuevo_presionado(event=None):
    """
    Abre un archivo OME-TIFF y carga los datos usando el módulo image_loader.
    """
    global img_original, img_array
    # Llama a la función del módulo para cargar la imagen
    img_original, metadata, xml_metadata = load_ometiff_image()
    if img_original is None:
        return  # Si se canceló la selección

    img_array = img_original.copy()
    print(img_array.shape)
    scale1.configure(to=img_array.shape[0] - 1)
    num_im = scale1.get()
    # Usa la función del módulo para procesar la capa seleccionada
    pil_img = process_image_slice(img_array, num_im, width, height)
    image_ = ImageTk.PhotoImage(pil_img)
    label.configure(image=image_)
    label.image = image_
    label.update()

def auto_contraste_presionado(event=None):
    global img_array
    # Llamamos a la función auto_contrast 
    img_array = auto_contrast(img_array)
    update_image()


def Histogram():
    global img_original
    # Llamamos a show_histogram 
    show_histogram(img_original)


def Binarize():
    """
    Aplica binarización a la imagen.
    """
    global img_original
    # Llama a la función del nuevo módulo, pasando la pila completa (img_original)
    # y el umbral (150). Regresa una imagen PIL.
    pil_img = binarize_variance(img_original, 150)

    top2 = tk.Toplevel()
    top2.title("Binarize")

    frame = tk.Frame(master=top2, relief=tk.RAISED, borderwidth=1)
    frame.grid(row=0, column=0, sticky='nsew')
    frame.pack_propagate(0)
    Grid.rowconfigure(top2, 0, weight=1)
    Grid.columnconfigure(top2, 0, weight=1)
    
    image_ = ImageTk.PhotoImage(pil_img)
    label = tk.Label(master=frame, image=image_)
    label.image = image_

    frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth=1)
    frame.grid(row=0, column=1, sticky='nsew')
    frame.pack_propagate(0)
    Grid.rowconfigure(top2, 0, weight=1)
    Grid.columnconfigure(top2, 1, weight=1)
    
    inputtxt = tk.Text(top2, height=1, width=5)
    inputtxt.insert("1.0", "150")
    printButton = tk.Button(top2, text="Apply Binarization", command=partial(UpdateBinarization, label, inputtxt))

def UpdateBinarization(label, inputtxt):
    """
    Update the binarization of the image based on the input threshold.
    """
    global img_original
    inp = int(inputtxt.get(1.0, "end-1c"))
    var_im = np.var(img_original, axis=0)
    pil_img = Apply_Binarize(var_im, inp)
    image_ = ImageTk.PhotoImage(pil_img)
    label.configure(image=image_)
    label.image = image_
    label.update()

def Apply_Binarize(var_im, th):
    """
    Apply binarization to the image based on the threshold.
    """
    var_im2 = np.reshape(var_im, (var_im.shape[0] * var_im.shape[1]))
    res_labels = [int(var_im2[i] > th) for i in range(len(var_im2))]
    res_labels = np.reshape(res_labels, (var_im.shape[0], var_im.shape[1]))
    res_labels = res_labels * 255
    pil_img = Image.fromarray(res_labels)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return pil_img

def update_image():
    """
    Update the displayed image based on the current scale value.
    """
    global img_array
    num_im = scale1.get()
    pil_img = Image.fromarray(img_array[num_im,:,:])
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    width_pil, height_pil = pil_img.size
    ratio = min(width/(width_pil * 1.5), height/(height_pil * 1.5))
    pil_img = pil_img.resize((int(width_pil * ratio), int(height_pil * ratio)), Image.LANCZOS)
    image_ = ImageTk.PhotoImage(pil_img)
    label.configure(image=image_)
    label.image = image_
    label.update()

def slider_presionado(val):
    threshold_slider_changed(threshold_scale.get())


def threshold_slider_changed(val):
    """
    Actualiza la imagen en el label aplicando el umbral actual del slider de threshold.
    """
    global img_array
    umbral = int(val)          # Valor del umbral obtenido del slider
    num_im = scale1.get()      # Índice de la capa actual del slider de capas
    current_slice = img_array[num_im, :, :]  # Extrae la capa actual (2D)
    
    # Aplica la función de threshold (definida en image_processing.py)
    pil_img = threshold_image_pil(current_slice, threshold=umbral)
    
    # Actualiza el widget de imagen (label)
    image_ = ImageTk.PhotoImage(pil_img)
    label.configure(image=image_)
    label.image = image_
    label.update()




# Add commands to the "Archivo" menu
menu_archivo.add_command(
    label="Open Ome/Tiff file",
    accelerator="Ctrl+O",
    command=archivo_nuevo_presionado,
    compound=tk.LEFT
)

# Add commands to the "Imagen" menu

menu_imagen.add_command(
    label="Umbral",
    command=lambda: threshold_slider_changed(threshold_scale.get()),
    compound=tk.LEFT
)

menu_imagen.add_command(
    label="Auto Contraste",
    command=auto_contraste_presionado,
    compound=tk.LEFT
)
menu_imagen.add_command(
    label="Histogram",
    command=Histogram,
    compound=tk.LEFT
)
menu_imagen.add_command(
    label="Binarize",
    command=Binarize,
    compound=tk.LEFT
)

# Add commands to the "Picos" menu
menu_picos.add_command(
    label='Abrir Datos', 
    command=lambda:initialize_visualization(window, menu_picos, canvas), 
    state=NORMAL
)
menu_picos.add_separator()
menu_picos.add_command(
    label='Elliptic Envelope', 
    command=None, 
    state=DISABLED
)
menu_picos.add_command(
    label='Peak Caller', 
    command=None, 
    state=DISABLED
)
menu_picos.add_command(
    label='Local Outlier Factor', 
    command=None, 
    state=DISABLED
)
menu_picos.add_command(
    label='Pico 4', 
    command=None, 
    state=DISABLED
)
menu_picos.add_command(
    label='Isolation Forest', 
    command=None, 
    state=DISABLED
)
menu_picos.add_command(
    label='Linear Model', 
    command=None, 
    state=DISABLED
)
menu_picos.add_command(
    label='Pico 7', 
    command=None, 
    state=DISABLED
)
menu_picos.add_separator()
menu_picos.add_command(
    label='Correlation Pearson', 
    command=None, 
    state=DISABLED
)
menu_picos.add_command(
    label='Correlation Kendall', 
    command=None, 
    state=DISABLED
)
menu_picos.add_command(
    label='Correlation Spearman', 
    command=None, 
    state=DISABLED
)
menu_picos.add_separator()
menu_picos.add_command(
    label='Dendogram', 
    command=None, 
    state=DISABLED
)

# Create and configure the main frame
frame = tk.Frame(master=window, relief=tk.RAISED, borderwidth=1)
frame.grid(row=0, column=0, sticky='nsew')
frame.pack_propagate(0)
Grid.rowconfigure(window, 0, weight=1)
Grid.columnconfigure(window, 0, weight=1)
label = tk.Label(master=frame)
label.pack(fill=tk.BOTH, expand=True)
window.focus()

# Load and display the initial image
try:
    pil_img = Image.open('input_image_7.png')
except FileNotFoundError:
    # Cargar una imagen por defecto o simplemente crear una imagen vacía
    pil_img = Image.new('RGB', (400, 300), color='gray')

#SLIDER

# Crear y configurar el frame para los sliders
frame2 = tk.Frame(master=window, relief=tk.RAISED, borderwidth=1)
frame2.grid(row=1, column=0, sticky='nsew')
frame2.pack_propagate(0)
Grid.rowconfigure(window, 1, weight=1)
Grid.columnconfigure(window, 0, weight=1)

# Slider para seleccionar la capa (slice)
scale1 = tk.Scale(master=frame2, from_=0, to=500, orient="horizontal", length=500, command=slider_presionado)
scale1.pack()

# Slider para el umbral (threshold) con rango de 0 a 255 y valor inicial 200
threshold_scale = tk.Scale(master=frame2, from_=0, to=255, orient="horizontal", length=500, 
                            label="Threshold", command=threshold_slider_changed)
threshold_scale.set(200)
threshold_scale.pack()

########## 

width_pil, height_pil = pil_img.size
ratio = min(width/(width_pil * 1.5), height/(height_pil * 1.5))
pil_img = pil_img.resize((int(width_pil * ratio), int(height_pil * ratio)), Image.LANCZOS)
image_ = ImageTk.PhotoImage(pil_img)
label.configure(image=image_)
label.image = image_
label.update()

# Create and configure the second frame for the slider
frame2 = tk.Frame(master=window, relief=tk.RAISED, borderwidth=1)
frame2.grid(row=1, column=0, sticky='nsew')
frame2.pack_propagate(0)
Grid.rowconfigure(window, 1, weight=1)
Grid.columnconfigure(window, 0, weight=1)
scale1 = tk.Scale(master=frame2, from_=0, to=500, orient="horizontal", length=500, command=slider_presionado)
scale1.pack()
#button1 = tk.Button(master=frame2, text="Guardar", command=save, state=DISABLED)
#button1.pack()

# Start the main event loop
window.mainloop()