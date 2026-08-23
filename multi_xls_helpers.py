"""
Helper functions to load one or more data files (.npy/.xls/.xlsx/.csv),
choose which sheets to load, and prepare their data columns for the
"Multiple Files" tab.
"""
import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox


_CSV_PSEUDO_SHEET = '(CSV)'
_NPY_PSEUDO_SHEET = '(NPY)'


def _is_csv_file(filepath):
    return filepath.lower().endswith('.csv')


def _is_npy_file(filepath):
    return filepath.lower().endswith('.npy')


def _read_sheet_names(filepath):
    """Returns the list of sheet (tab) names of an Excel file. A .csv or
    .npy file has no sheets, so each is assigned a single "pseudo" sheet
    so it fits into the same selection flow."""
    if _is_csv_file(filepath):
        return [_CSV_PSEUDO_SHEET]
    if _is_npy_file(filepath):
        return [_NPY_PSEUDO_SHEET]
    xl = pd.ExcelFile(filepath)
    return xl.sheet_names


def _select_sheets_dialog(parent, file_sheet_map):
    """
    Shows a dialog to choose which sheets from each Excel file to load.

    file_sheet_map: dict {filepath: [sheet_names]}
    Returns: list of selected (filepath, sheet_name) tuples, or None if cancelled.
    """
    dialog = tk.Toplevel(parent)
    dialog.title("Select Sheets to Load")
    dialog.geometry("480x520")
    dialog.transient(parent)
    dialog.grab_set()

    tk.Label(dialog, text="Select the sheets (tabs) you want to load:",
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
    tk.Button(btns_top, text="Select All",
              command=lambda: _set_all(True)).pack(side=tk.LEFT, padx=2, pady=4)
    tk.Button(btns_top, text="Deselect All",
              command=lambda: _set_all(False)).pack(side=tk.LEFT, padx=2)

    def on_ok():
        selected = [key for key, var in check_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("No Selection", "Select at least one sheet to load.")
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
    """Reads an Excel sheet (or a .csv/.npy file, each treated as its own
    single "sheet") with no header row: no column names are assumed, so
    each column is a data series identified solely by its position (0, 1,
    2, ...). Since all loaded files must have the same number of columns,
    that position is what makes a column in one file/sheet correspond to
    the same column in another."""
    if _is_csv_file(filepath):
        df = pd.read_csv(filepath, header=None)
    elif _is_npy_file(filepath):
        arr = np.load(filepath, allow_pickle=True)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        df = pd.DataFrame(arr)
    else:
        df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = [str(c) for c in df.columns]
    return df


def pick_files_and_sheets(parent):
    """
    Prompts the user for one or more .npy/.xls/.xlsx/.csv files and which
    sheets from each one to load (a single selected file works the same
    way - it's just a file_sheet_map with one entry).

    Returns: list of selected (filepath, sheet_name) tuples, or None if
    the user cancels or there are no sheets to choose from.
    """
    filenames = filedialog.askopenfilenames(
        parent=parent,
        title="Open Data Files (.npy / .xls / .csv)",
        filetypes=[("Data files", "*.npy *.xlsx *.xls *.csv"), ("Numpy files", "*.npy"),
                   ("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not filenames:
        return None

    file_sheet_map = {}
    for filepath in filenames:
        try:
            sheets = _read_sheet_names(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read '{os.path.basename(filepath)}':\n{e}")
            continue
        if sheets:
            file_sheet_map[filepath] = sheets
    if not file_sheet_map:
        return None

    return _select_sheets_dialog(parent, file_sheet_map)


def load_selected_sheets(selection, progress_callback=None, error_callback=None):
    """
    Loads each (filepath, sheet_name) sheet from 'selection' into a dataset:
        {'file': filepath, 'sheet': sheet_name, 'label': 'file - sheet',
         'df': DataFrame, 'column_names': [...]}

    progress_callback(done, total, filepath, sheet_name), if given, is called
    after loading each sheet so a progress bar can be updated.

    error_callback(filepath, sheet_name, exception), if given, is called
    instead of showing the error messagebox directly - for when this
    function runs on a thread other than the main one (where it isn't
    safe to create Tk windows directly) and the caller prefers to forward
    the error to the main thread itself.
    """
    datasets = []
    total = len(selection)
    for i, (filepath, sheet) in enumerate(selection, start=1):
        try:
            df = _load_sheet_dataframe(filepath, sheet)
        except Exception as e:
            if error_callback:
                error_callback(filepath, sheet, e)
            else:
                messagebox.showerror(
                    "Error",
                    f"Could not load sheet '{sheet}' from '{os.path.basename(filepath)}':\n{e}"
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
    """Returns the columns (identified by position) common to all
    datasets, preserving the order of appearance from the first dataset.
    Since the sheets have no header, this is normally just 0..n-1 for the
    smallest number of columns shared by all the loaded sheets."""
    if not datasets:
        return []
    common = set(datasets[0]['column_names'])
    for ds in datasets[1:]:
        common &= set(ds['column_names'])
    return [c for c in datasets[0]['column_names'] if c in common]
