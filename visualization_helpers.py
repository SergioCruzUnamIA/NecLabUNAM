from tkinter import *
from tkinter import filedialog
from peak_functions import *
from corr_dendo_functions import *

def initialize_visualization(window, menu_picos, canvas):
    filename = filedialog.askopenfilename(parent=window,title="Open File",filetypes=[("Numpy files", "*.npy")])
    data = normalize_data(filename)
    canvas = _plot_data(data, window, canvas)

    menu_picos.entryconfig("Elliptic Envelope", command=lambda:elliptic_envelope_peak(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Peak Caller", command=lambda:actual_peak_caller(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Local Outlier Factor", command=lambda:local_outlier_factor_peak(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Peak Function 4", command=lambda:clf_peak(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Isolation Forest", command=lambda:isolation_forest_peak(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Linear Model", command=lambda:linear_model_peak(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Peak Function 7", command=lambda:lasso_peak(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Correlation Pearson", command=lambda:plot_correlation(data,correlation_pearson(data),window,canvas), state=NORMAL)
    menu_picos.entryconfig("Correlation Kendall", command=lambda:plot_correlation(data,correlation_kendall(data),window,canvas), state=NORMAL)
    menu_picos.entryconfig("Correlation Spearman", command=lambda:plot_correlation(data,correlation_spearman(data),window,canvas), state=NORMAL)
    menu_picos.entryconfig("Dendogram", command=lambda:plot_dendogram(data, window, canvas), state=NORMAL)

def _plot_data(data, window, canvas):
    fig, ax = plt.subplots()
    plt.plot(np.array(range(len(data[:, 15]))).reshape(-1, 1), data[:, 15])
    
    show_scale_widget(window)
    
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
    return canvas

def show_scale_widget(main_window):
    for widget in main_window.winfo_children():
        if isinstance(widget, tk.Frame):
            for child in widget.winfo_children():
                if isinstance(child, tk.Scale):
                    child.grid()
                    break