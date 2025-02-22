from tkinter import *
from tkinter import ttk, filedialog
from pico_calculators import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#TODO ver como sobreescribir el canvas

data = None
canvas = None

root = Tk()
root.title("NecLab")

root.tk.call('tk', 'windowingsystem') 
root.option_add('*tearOff', FALSE)

def openfile():
    global data
    filename = filedialog.askopenfilename(parent=root,title="Abrir archivo",filetypes=[("Numpy files", "*.npy")])
    data = programflow(filename)
    plot_data(data)
    menu_pico.entryconfig("Pico 1", command=lambda:elliptic_envelope_peak(data, root, canvas), state=NORMAL)
    #menu_pico.entryconfig("Pico 2", command=lambda:peak_caller(data, root), state=NORMAL) TODO:ver como implementar
    menu_pico.entryconfig("Pico 3", command=lambda:smvr_peak(data, root, canvas), state=NORMAL)
    menu_pico.entryconfig("Pico 4", command=lambda:clf_peak(data, root, canvas), state=NORMAL)
    menu_pico.entryconfig("Pico 5", command=lambda:isolation_forest_peak(data, root, canvas), state=NORMAL)
    menu_pico.entryconfig("Pico 6", command=lambda:linear_model_peak(data, root, canvas), state=NORMAL)
    menu_pico.entryconfig("Pico 7", command=lambda:lasso_peak(data, root, canvas), state=NORMAL)

def plot_data(data):
    global canvas
    fig, ax = plt.subplots()
    plt.plot(np.array(range(len(data[:, 15]))).reshape(-1, 1), data[:, 15])
    ax = plt.gca()
    if canvas is not None:
        canvas.get_tk_widget().pack_forget()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

menubar = Menu(root)
root.config(menu=menubar)
menu_pico = Menu(menubar)
menubar.add_cascade(menu=menu_pico, label='Pico')
menu_pico.add_command(label='Abrir Datos', command=openfile)
menu_pico.add_separator()
menu_pico.add_command(label='Pico 1', command=None, state=DISABLED)
menu_pico.add_command(label='Pico 2', command=None, state=DISABLED)
menu_pico.add_command(label='Pico 3', command=None, state=DISABLED)
menu_pico.add_command(label='Pico 4', command=None, state=DISABLED)
menu_pico.add_command(label='Pico 5', command=None, state=DISABLED)
menu_pico.add_command(label='Pico 6', command=None, state=DISABLED)
menu_pico.add_command(label='Pico 7', command=None, state=DISABLED)
menubar.add_cascade(menu=menu_pico, label='Correlacion')

root.mainloop()