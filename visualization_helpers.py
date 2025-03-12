from tkinter import *
from tkinter import filedialog
from peak_functions import *
from corr_dendo_functions import *

def initialize_visualization(window, menu_picos, canvas):
    filename = filedialog.askopenfilename(parent=window,title="Abrir archivo",filetypes=[("Numpy files", "*.npy")])
    data = normalize_data(filename)
    _plot_data(data, window, canvas)

    menu_picos.entryconfig("Elliptic Envelope", command=lambda:elliptic_envelope_peak(data), state=NORMAL)
    menu_picos.entryconfig("Peak Caller", command=lambda:actual_peak_caller(data), state=NORMAL)
    menu_picos.entryconfig("Local Outlier Factor", command=lambda:local_outlier_factor_peak(data), state=NORMAL)
    menu_picos.entryconfig("Pico 4", command=lambda:clf_peak(data), state=NORMAL)
    menu_picos.entryconfig("Isolation Forest", command=lambda:isolation_forest_peak(data), state=NORMAL)
    menu_picos.entryconfig("Linear Model", command=lambda:linear_model_peak(data), state=NORMAL)
    menu_picos.entryconfig("Pico 7", command=lambda:lasso_peak(data), state=NORMAL)
    menu_picos.entryconfig("Correlation Pearson", command=lambda:plot_correlation(data,correlation_pearson(data),window,canvas), state=NORMAL)
    menu_picos.entryconfig("Correlation Kendall", command=lambda:plot_correlation(data,correlation_kendall(data),window,canvas), state=NORMAL)
    menu_picos.entryconfig("Correlation Spearman", command=lambda:plot_correlation(data,correlation_spearman(data),window,canvas), state=NORMAL)
    menu_picos.entryconfig("Dendogram", command=lambda:plot_correlation(data, window, canvas), state=NORMAL)
    #button1.config(state=NORMAL)

def _plot_data(data, window, canvas):
    fig, ax = plt.subplots()
    plt.plot(np.array(range(len(data[:, 15]))).reshape(-1, 1), data[:, 15])
    ax = plt.gca()
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')