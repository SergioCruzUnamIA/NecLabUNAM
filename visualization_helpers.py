from tkinter import *


def _show_sheet_selection_dialog(parent, sheet_names):
    """
    Show a dialog for the user to select which sheet of an Excel workbook to load.
    Returns the selected sheet name or None if cancelled.
    """
    dialog = Toplevel(parent)
    dialog.title("Select Sheet")
    dialog.geometry("400x500")
    dialog.transient(parent)
    dialog.grab_set()

    selected_sheet = None

    Label(dialog, text="Select which sheet to load:", font=('Arial', 12, 'bold')).pack(pady=10)

    list_frame = Frame(dialog)
    list_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

    scrollbar = Scrollbar(list_frame)
    scrollbar.pack(side=RIGHT, fill=Y)

    listbox = Listbox(list_frame, yscrollcommand=scrollbar.set, font=('Arial', 10))
    listbox.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    for name in sheet_names:
        listbox.insert(END, name)

    listbox.selection_set(0)
    listbox.activate(0)

    def on_ok():
        nonlocal selected_sheet
        selection = listbox.curselection()
        if selection:
            selected_sheet = sheet_names[selection[0]]
            dialog.destroy()
        else:
            from tkinter import messagebox
            messagebox.showwarning("No Selection", "Please select a sheet")

    def on_cancel():
        nonlocal selected_sheet
        selected_sheet = None
        dialog.destroy()

    def on_double_click(event):
        on_ok()

    listbox.bind('<Double-Button-1>', on_double_click)

    button_frame = Frame(dialog)
    button_frame.pack(pady=10)

    Button(button_frame, text="OK", command=on_ok, width=10).pack(side=LEFT, padx=5)
    Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side=LEFT, padx=5)

    parent.wait_window(dialog)

    return selected_sheet
