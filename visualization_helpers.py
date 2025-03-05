from tkinter import *
from tkinter import filedialog
from pico_calculators import *

def openfile(window, menu_picos, canvas, button1):
    filename = filedialog.askopenfilename(parent=window,title="Abrir archivo",filetypes=[("Numpy files", "*.npy")])
    data = programflow(filename)
    plot_data(data, window, canvas)

    rise = 5
    fall = 5
    max_lookahead = 10
    max_lookback = 10
    menu_picos.entryconfig("Elliptic Envelope", command=lambda:elliptic_envelope_peak(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Peak Caller", command=lambda:peak_caller(data, rise, fall, max_lookahead, max_lookback, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Local Outlier Factor", command=lambda:local_outlier_factor_peak(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Pico 4", command=lambda:clf_peak(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Isolation Forest", command=lambda:isolation_forest_peak(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Linear Model", command=lambda:linear_model_peak(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Pico 7", command=lambda:lasso_peak(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Correlation Pearson", command=lambda:actually_plot_corr(data,correlation_pearson(data),window,canvas), state=NORMAL)
    menu_picos.entryconfig("Correlation Kendall", command=lambda:actually_plot_corr(data,correlation_kendall(data),window,canvas), state=NORMAL)
    menu_picos.entryconfig("Correlation Spearman", command=lambda:actually_plot_corr(data,correlation_spearman(data),window,canvas), state=NORMAL)
    menu_picos.entryconfig("Dendogram", command=lambda:actually_plot_dendo(data, window, canvas), state=NORMAL)
    button1.config(state=NORMAL)

#def correlation(data, window, menu_picos, canvas):
#    menu_picos.entryconfig("Correlation Pearson", command=lambda:actually_plot_corr(data,correlation_pearson(data),window,canvas), state=NORMAL)
#    menu_picos.entryconfig("Correlation Kendall", command=lambda:actually_plot_corr(data,correlation_kendall(data),window,canvas), state=NORMAL)
#    menu_picos.entryconfig("Correlation Spearman", command=actually_plot_corr(data,correlation_spearman(data),window,canvas), state=NORMAL)

#def dendograms(data, window, menu_picos, canvas):
#    menu_picos.entryconfig("Dendogram", command=lambda:actually_plot_dendo(), state=NORMAL)

def plot_data(data, window, canvas):
    fig, ax = plt.subplots()
    plt.plot(np.array(range(len(data[:, 15]))).reshape(-1, 1), data[:, 15])
    ax = plt.gca()
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

def save():
    filename = asksaveasfilename(initialfile = 'Untitled.png',defaultextension=".png",filetypes=[("All Files","*.*"),("Portable Graphics Format","*.png")])
    plt.savefig(filename)