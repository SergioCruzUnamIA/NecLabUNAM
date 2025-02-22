from tkinter import *
from tkinter import ttk, filedialog
from pico_calculators import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

data = None

root = Tk()
root.title("NecLab")

root.tk.call('tk', 'windowingsystem') 
root.option_add('*tearOff', FALSE)

def openfile():
    global data
    filename = filedialog.askopenfilename(parent=root,title="Abrir archivo",filetypes=[("Numpy files", "*.npy")])
    data = programflow(filename)
    plot_data(data)
    menu_pico.entryconfig("Pico 1", state=NORMAL)
    menu_pico.entryconfig("Pico 1", )
    menu_pico.entryconfig("Pico 2", state=NORMAL)
    menu_pico.entryconfig("Pico 3", state=NORMAL)
    menu_pico.entryconfig("Pico 4", state=NORMAL)
    menu_pico.entryconfig("Pico 5", state=NORMAL)
    menu_pico.entryconfig("Pico 6", state=NORMAL)
    menu_pico.entryconfig("Pico 7", state=NORMAL)
    menu_pico.entryconfig("Pico 8", state=NORMAL)

def plot_data(data):
    fig, ax = plt.subplots()
    plt.plot(np.array(range(len(data[:, 15]))).reshape(-1, 1), data[:, 15])
    ax = plt.gca()
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
menu_pico.add_command(label='Pico 8', command=None, state=DISABLED)
menubar.add_cascade(menu=menu_pico, label='Correlacion')

root.mainloop()