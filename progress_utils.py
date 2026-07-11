"""Shared helper to run a slow save/load operation on a background thread
while showing a responsive progress window, instead of freezing the Tk
main loop for the duration of the operation.

Used across the project's various save/load buttons (main data, image
stacks, correlation/dendrogram exports, multi-file Excel, etc.) so they
all get the same "fast background load + progress bar" behavior instead
of each reimplementing it.
"""

import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox


def run_with_progress_window(parent, title, message, maximum, worker_fn,
                              on_complete, on_error=None):
    """Runs worker_fn(report_progress, report_error) on a daemon thread,
    showing a progress window that stays responsive (can be moved,
    repainted, etc.) while the work happens - instead of blocking the
    main Tk thread for the whole operation, which is what made windows
    report as "Not Responding" on large files.

    'parent' is the Tk widget/window the progress dialog should be
    transient to (usually the app's root window).

    worker_fn must call report_progress(done, total, text) to update the
    bar (optional), and return its result, which is passed to
    on_complete(result) on the main thread when it finishes.
    report_error(text) shows an error messagebox without stopping the
    thread (for per-item errors that shouldn't abort the whole
    operation). An uncaught exception inside worker_fn is passed to
    on_error(exception) (or shown with a generic messagebox if on_error
    is None), also on the main thread.

    Tkinter is not thread-safe: worker_fn must never touch Tk widgets or
    variables directly, only call report_progress/report_error (which
    just enqueue a message) and work with plain Python/pandas/numpy
    data."""
    progress_win = tk.Toplevel(parent)
    progress_win.title(title)
    progress_win.transient(parent)
    progress_win.grab_set()
    progress_win.resizable(False, False)
    progress_win.protocol("WM_DELETE_WINDOW", lambda: None)

    tk.Label(progress_win, text=message, font=('Arial', 11, 'bold')).pack(
        pady=(15, 4), padx=15, anchor='w')
    status_label = tk.Label(progress_win, text="Preparando...", font=('Arial', 9),
                             anchor='w')
    status_label.pack(fill='x', padx=15, anchor='w')
    progress_bar = ttk.Progressbar(progress_win, orient='horizontal', length=390,
                                    mode='determinate', maximum=max(maximum, 1))
    progress_bar.pack(pady=(8, 15), padx=15)
    progress_win.update_idletasks()

    result_queue = queue.Queue()

    def report_progress(done, total, text):
        result_queue.put(('progress', done, total, text))

    def report_error(text):
        result_queue.put(('error', text))

    def run():
        try:
            result = worker_fn(report_progress, report_error)
            result_queue.put(('done', result))
        except Exception as exc:
            result_queue.put(('exception', exc))

    threading.Thread(target=run, daemon=True).start()

    def poll():
        try:
            while True:
                item = result_queue.get_nowait()
                kind = item[0]
                if kind == 'progress':
                    _, done, total, text = item
                    progress_bar['maximum'] = max(total, 1)
                    progress_bar['value'] = done
                    status_label.config(text=text)
                elif kind == 'error':
                    messagebox.showerror("Error", item[1], parent=progress_win)
                elif kind == 'done':
                    progress_win.destroy()
                    on_complete(item[1])
                    return
                elif kind == 'exception':
                    progress_win.destroy()
                    if on_error:
                        on_error(item[1])
                    else:
                        messagebox.showerror("Error", str(item[1]))
                    return
        except queue.Empty:
            pass
        progress_win.after(50, poll)

    progress_win.after(50, poll)


def run_save_with_progress(parent, title, message, worker_fn, success_message=None,
                            on_error=None):
    """Convenience wrapper around run_with_progress_window for the common
    "single indeterminate-ish step" save case: worker_fn() takes no
    arguments, does the (potentially slow) file write on the background
    thread and returns the destination filename (or any result). Shows a
    generic 'Guardando...' status and, on success, a Saved messagebox
    with success_message(result) if given, else a generic one."""
    def _worker(report_progress, report_error):
        report_progress(0, 1, "Guardando...")
        result = worker_fn()
        report_progress(1, 1, "Listo")
        return result

    def _on_complete(result):
        if success_message is not None:
            messagebox.showinfo("Guardado", success_message(result))

    run_with_progress_window(
        parent, title=title, message=message, maximum=1,
        worker_fn=_worker, on_complete=_on_complete, on_error=on_error)


def run_load_with_progress(parent, title, message, worker_fn, on_complete, on_error=None):
    """Convenience wrapper around run_with_progress_window for the common
    "single indeterminate-ish step" load case: worker_fn() takes no
    arguments, does the (potentially slow) file read on the background
    thread and returns the loaded result, which is handed to
    on_complete(result) on the main thread."""
    def _worker(report_progress, report_error):
        report_progress(0, 1, "Cargando...")
        result = worker_fn()
        report_progress(1, 1, "Listo")
        return result

    run_with_progress_window(
        parent, title=title, message=message, maximum=1,
        worker_fn=_worker, on_complete=on_complete, on_error=on_error)
