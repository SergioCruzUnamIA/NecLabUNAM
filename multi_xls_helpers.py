"""
Funciones auxiliares para cargar múltiples archivos Excel (.xls/.xlsx),
elegir qué hojas cargar, y preparar sus columnas de datos para la
pestaña "Datos Multiples".
"""
import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox


def _read_sheet_names(filepath):
    """Devuelve la lista de nombres de hojas (tabs) de un archivo Excel."""
    xl = pd.ExcelFile(filepath)
    return xl.sheet_names


def _select_sheets_dialog(parent, file_sheet_map):
    """
    Muestra un diálogo para elegir qué hojas de cada archivo Excel cargar.

    file_sheet_map: dict {filepath: [sheet_names]}
    Devuelve: lista de tuplas (filepath, sheet_name) seleccionadas, o None si se cancela.
    """
    dialog = tk.Toplevel(parent)
    dialog.title("Seleccionar Hojas a Cargar")
    dialog.geometry("480x520")
    dialog.transient(parent)
    dialog.grab_set()

    tk.Label(dialog, text="Selecciona las hojas (tabs) que deseas cargar:",
             font=('Arial', 11, 'bold')).pack(pady=(10, 4), padx=10, anchor='w')

    outer = tk.Frame(dialog)
    outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    canvas = tk.Canvas(outer, highlightthickness=0)
    scrollbar = tk.Scrollbar(outer, orient='vertical', command=canvas.yview)
    inner = tk.Frame(canvas)

    inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.create_window((0, 0), window=inner, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    check_vars = {}  # (filepath, sheet) -> BooleanVar, insertion order preserved

    for filepath, sheets in file_sheet_map.items():
        tk.Label(inner, text=os.path.basename(filepath), font=('Arial', 10, 'bold'),
                 anchor='w').pack(fill='x', pady=(8, 2))
        for sheet in sheets:
            var = tk.BooleanVar(value=True)
            check_vars[(filepath, sheet)] = var
            tk.Checkbutton(inner, text=sheet, variable=var, anchor='w'
                            ).pack(fill='x', padx=20, anchor='w')

    result = {'selection': None}

    def _set_all(value):
        for var in check_vars.values():
            var.set(value)

    btns_top = tk.Frame(dialog)
    btns_top.pack(fill='x', padx=10)
    tk.Button(btns_top, text="Seleccionar Todo",
              command=lambda: _set_all(True)).pack(side=tk.LEFT, padx=2, pady=4)
    tk.Button(btns_top, text="Deseleccionar Todo",
              command=lambda: _set_all(False)).pack(side=tk.LEFT, padx=2)

    def on_ok():
        selected = [key for key, var in check_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("Sin selección", "Selecciona al menos una hoja para cargar.")
            return
        result['selection'] = selected
        dialog.destroy()

    def on_cancel():
        result['selection'] = None
        dialog.destroy()

    btns = tk.Frame(dialog)
    btns.pack(pady=10)
    tk.Button(btns, text="OK", command=on_ok, width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(btns, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)

    parent.wait_window(dialog)
    return result['selection']


def _load_sheet_dataframe(filepath, sheet_name):
    """Lee una hoja de Excel sin fila de encabezado: las hojas no tienen
    nombres de columna, así que cada columna es una serie de datos
    identificada únicamente por su posición (0, 1, 2, ...). Como todas las
    hojas tienen el mismo número de columnas, esa posición es lo que hace
    corresponder una columna de una hoja con la misma columna en otra."""
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = [str(c) for c in df.columns]
    return df


def pick_files_and_sheets(parent):
    """
    Pide al usuario múltiples archivos .xls/.xlsx y qué hojas de cada uno
    quiere cargar.

    Devuelve: lista de tuplas (filepath, sheet_name) seleccionadas, o None
    si el usuario cancela o no hay hojas para elegir.
    """
    filenames = filedialog.askopenfilenames(
        parent=parent,
        title="Abrir Multiples Archivos (.xls)",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    if not filenames:
        return None

    file_sheet_map = {}
    for filepath in filenames:
        try:
            sheets = _read_sheet_names(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer '{os.path.basename(filepath)}':\n{e}")
            continue
        if sheets:
            file_sheet_map[filepath] = sheets
    if not file_sheet_map:
        return None

    return _select_sheets_dialog(parent, file_sheet_map)


def load_selected_sheets(selection, progress_callback=None):
    """
    Carga cada hoja (filepath, sheet_name) de 'selection' en un dataset:
        {'file': filepath, 'sheet': sheet_name, 'label': 'archivo - hoja',
         'df': DataFrame, 'column_names': [...]}

    progress_callback(done, total, filepath, sheet_name), si se da, se llama
    después de cargar cada hoja para poder actualizar una barra de progreso.
    """
    datasets = []
    total = len(selection)
    for i, (filepath, sheet) in enumerate(selection, start=1):
        try:
            df = _load_sheet_dataframe(filepath, sheet)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"No se pudo cargar la hoja '{sheet}' de '{os.path.basename(filepath)}':\n{e}"
            )
            df = None

        if df is not None and not df.empty:
            datasets.append({
                'file': filepath,
                'sheet': sheet,
                'label': f"{os.path.splitext(os.path.basename(filepath))[0]} - {sheet}",
                'df': df,
                'column_names': df.columns.tolist(),
            })

        if progress_callback:
            progress_callback(i, total, filepath, sheet)

    return datasets


def common_column_names(datasets):
    """Devuelve las columnas (identificadas por posición) comunes a todos los
    datasets, preservando el orden de aparición del primer dataset. Como las
    hojas no tienen encabezado, esto normalmente es solo 0..n-1 para el menor
    número de columnas compartido por todas las hojas cargadas."""
    if not datasets:
        return []
    common = set(datasets[0]['column_names'])
    for ds in datasets[1:]:
        common &= set(ds['column_names'])
    return [c for c in datasets[0]['column_names'] if c in common]
