import os
os.environ["OMP_NUM_THREADS"] = "1"
from pyometiff import OMETIFFReader
import tkinter as tk
import numpy as np
from tkinter import PhotoImage, Grid
from PIL import Image, ImageTk, ImageOps
import cv2
from tkinter import filedialog as fd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from functools import partial

window = tk.Tk()
images = {}
#getting screen width and height of display
width= window.winfo_screenwidth() * 0.8
height= window.winfo_screenheight() * 0.8
#setting tkinter window size
window.geometry("%dx%d" % (width, height))

barra_menus = tk.Menu()
window.config(menu=barra_menus)

menu_archivo = tk.Menu(barra_menus, tearoff=False)
barra_menus.add_cascade(menu=menu_archivo, label="Archivo")
menu_imagen = tk.Menu(barra_menus, tearoff=False)
barra_menus.add_cascade(menu=menu_imagen, label="Imagen")

img_original = []
img_array = []

def archivo_nuevo_presionado(event=None):
    global img_array
    global img_original
    filename = fd.askopenfilename()
    reader = OMETIFFReader(fpath=filename)
    img_original, metadata, xml_metadata = reader.read()
    img_array = img_original.copy()
    print(img_array.shape)
    scale1.configure(to=img_array.shape[0] - 1)
    num_im = scale1.get()
    pil_img = Image.fromarray(img_array[num_im,:,:])

    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    width_pil, height_pil = pil_img.size
    ratio = min(width/(width_pil * 1.5), height/(height_pil * 1.5))
    pil_img = pil_img.resize((int(width_pil * ratio), int(height_pil * ratio)))
    image_ = ImageTk.PhotoImage(pil_img)
    label.configure(image=image_)
    label.image = image_
    label.update()

def auto_contraste_presionado(event=None):
    for i in range(img_array.shape[0]):
        im_pil = Image.fromarray(img_array[i,:,:])
        if im_pil.mode != 'RGB':
            im_pil = im_pil.convert('RGB')
        im2 = ImageOps.autocontrast(im_pil, cutoff = 2, ignore = 2).convert('L')
        img_array[i,:,:] = np.array(im2)
    update_image()

def Histogram():
    global img_original
    var_im = np.var(img_original, axis=0)
    histogram = plt.hist(var_im)
    imgplot = plt.show()

def Binarize():
    global img_original
    var_im = np.var(img_original, axis=0)
    pil_img = Apply_Binarize(var_im, 150)
    top2 = tk.Toplevel()
    top2.title("Binarize")

    frame = tk.Frame(
            master=top2,
            relief=tk.RAISED,
            borderwidth=1
        )
    frame.grid(row=0, column=0, sticky='nsew')
    frame.pack_propagate(0)
    Grid.rowconfigure(top2, 0, weight=1)
    Grid.columnconfigure(top2, 0, weight=1)
    
    image_ = ImageTk.PhotoImage(pil_img)
    label = tk.Label(master=frame, image=image_)
    label.image = image_

    frame = tk.Frame(
            master=window,
            relief=tk.RAISED,
            borderwidth=1
        )
    frame.grid(row=0, column=1, sticky='nsew')
    frame.pack_propagate(0)
    Grid.rowconfigure(top2, 0, weight=1)
    Grid.columnconfigure(top2, 1, weight=1)
    
    inputtxt = tk.Text(top2, height = 1, width = 5) 
    inputtxt.insert("1.0", "150")
    printButton = tk.Button(top2, text = "Apply Binarization", command=partial(UpdateBinarization, label, inputtxt))

def UpdateBinarization(label, inputtxt): 
    global img_original
    inp = int(inputtxt.get(1.0, "end-1c"))
    var_im = np.var(img_original, axis=0)
    pil_img = Apply_Binarize(var_im, inp)
    image_ = ImageTk.PhotoImage(pil_img)
    label.configure(image=image_)
    label.image = image_
    label.update()

def Apply_Binarize(var_im, th):
    var_im2 = np.reshape(var_im, (var_im.shape[0] * var_im.shape[1]))
    res_labels = [int(var_im2[i] > th) for i in range(len(var_im2))]
    res_labels = np.reshape(res_labels, (var_im.shape[0], var_im.shape[1]))
    res_labels = res_labels * 255
    pil_img = Image.fromarray(res_labels)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return pil_img
    

def update_image():
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
    update_image()

menu_archivo.add_command(
    label="Open Ome/Tiff file",
    accelerator="Ctrl+O",
    command=archivo_nuevo_presionado,
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

frame = tk.Frame(
            master=window,
            relief=tk.RAISED,
            borderwidth=1
        )
frame.grid(row=0, column=0, sticky='nsew')
frame.pack_propagate(0)
Grid.rowconfigure(window, 0, weight=1)
Grid.columnconfigure(window, 0, weight=1)
label = tk.Label(master=frame)
label.pack(fill=tk.BOTH, expand=True)

pil_img = Image.open('input_image_7.png')
width_pil, height_pil = pil_img.size
ratio = min(width/(width_pil * 1.5), height/(height_pil * 1.5))
pil_img = pil_img.resize((int(width_pil * ratio), int(height_pil * ratio)), Image.LANCZOS)
image_ = ImageTk.PhotoImage(pil_img)
label.configure(image=image_)
label.image = image_
label.update()

frame2 = tk.Frame(
            master=window,
            relief=tk.RAISED,
            borderwidth=1
        )
frame2.grid(row=1, column=0, sticky='nsew')
frame2.pack_propagate(0)
Grid.rowconfigure(window, 1, weight=1)
Grid.columnconfigure(window, 0, weight=1)
scale1 = tk.Scale(master=frame2, from_=0, to=500, orient="horizontal", length=500, command=slider_presionado)
scale1.pack()

# window.grid_columnconfigure(0, weight=1)
window.mainloop()
