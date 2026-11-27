"""
Cell Signals: the per-cell view of stage 5 (user interface only).

A list of every cell in the project's stage 5 signals and, for the cell
selected in it, two plots that share the time axis:

    top     the raw signal, with its peaks (events) marked and, when the
            lowest-points baseline is taken on the raw signal, that
            baseline and the lowest points it goes through;
    bottom  the processed (normalized) signal the peaks were detected on,
            with the same peaks and the lowest points used.

The sidebar has the stage 5 settings (normalization and Peak Finder).
Changing them recomputes stage 5 through `apply_settings`, which the main
window routes to the cell detection tab that owns stage 5 (so its Signals
& Correlation tab stays in step) or, for signals loaded from a file,
recomputes it itself. The view follows the project: whenever stage 5
changes (other cells selected, other settings, a project opened) it
redraws.

This is the successor of the old Data Visualization tab's column list +
Peak Finder view, reading the pipeline's stage 5 instead of a loose file.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import asksaveasfilename

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from neclab_core import analysis as core_analysis
from neclab_core import events, normalization
from neclab_core import stages as core_stages
from neclab_core.project import OUT_OF_DATE
from neclab_core.signals import signal_label
from ui_tabs import add_tab

_PANEL = '#ffffff'
_CARD = '#f8fafc'
_ACC = '#2563eb'
_TEXT = '#1e293b'
_SUB = '#64748b'
_BORDER = '#e2e8f0'

_RAW_COLOR = 'steelblue'
_PROCESSED_COLOR = 'seagreen'
_PEAK_COLOR = 'crimson'
_BASELINE_COLOR = 'darkorange'

IMAGE_TYPES = [("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("TIFF", "*.tiff"),
               ("All files", "*.*")]


class CellSignalsView:
    """The Cell Signals tab. `apply_settings(settings)` recomputes stage 5
    with new normalization/peak settings (neclab_core.analysis.DEFAULT_SETTINGS
    keys); `on_closed()` runs once the tab is closed."""

    def __init__(self, notebook, parent, project, apply_settings, on_closed=None):
        self.project = project
        self.apply_settings = apply_settings
        self.on_closed = on_closed
        self.peak_params = {}        # {method: params} last chosen for each Peak Finder method
        self.columns = []            # stage 5 signal columns, in list order
        self.current = None          # column shown
        self.raw = self.norm = self.events = None
        self.peaks = {}              # {column: peak frames}
        self.settings = dict(core_analysis.DEFAULT_SETTINGS)
        self.description = "raw"
        self.identity = True
        self._lowest = {}            # {column: frames of the lowest points used}, filled as cells are shown
        self._shown_key = object()   # what stage 5 looked like when last drawn

        self.tab = add_tab(notebook, "Cell Signals", parent=parent, on_close=self.close)
        self._build()
        self.project.add_listener(self._on_project_changed)
        self.refresh(force=True)

    # ------------------------------------------------------------ tab

    def exists(self):
        return self.tab.exists()

    def show(self):
        self.tab.show()

    def close(self):
        self.project.remove_listener(self._on_project_changed)
        self.tab.close()
        if self.on_closed is not None:
            self.on_closed()

    def set_project(self, project):
        """Follow another project (e.g. one just opened)."""
        self.project.remove_listener(self._on_project_changed)
        self.project = project
        self.project.add_listener(self._on_project_changed)
        self.peak_params = {}
        self.refresh(force=True)

    # ------------------------------------------------------------ layout

    def _build(self):
        self.show_peaks_var = tk.BooleanVar(value=True)
        self.show_points_var = tk.BooleanVar(value=True)

        def build_view(m):
            m.add_checkbutton(label="Show peaks", variable=self.show_peaks_var, command=self.draw)
            m.add_checkbutton(label="Show lowest points & baseline", variable=self.show_points_var,
                              command=self.draw)
            m.add_separator()
            m.add_command(label="Previous Cell", accelerator="↑", command=lambda: self._step(-1))
            m.add_command(label="Next Cell", accelerator="↓", command=lambda: self._step(1))

        def build_save(m):
            m.add_command(label="Save Plot of This Cell...", command=self.save_plot)
            m.add_command(label="Save This Cell's Peaks...", command=self.save_cell_peaks)
            m.add_separator()
            m.add_command(label="Export All Events...", command=lambda: self._export("events"))
            m.add_command(label="Export Normalized Signals...",
                          command=lambda: self._export("normalized_signals"))

        self.tab.add_menu("View", build_view)
        self.save_menu = self.tab.add_menu("Save", build_save)

        body = self.tab.body
        body.rowconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        side = tk.Frame(body, bg=_PANEL, width=290, highlightbackground=_BORDER, highlightthickness=1)
        side.grid(row=0, column=0, sticky='nsew', padx=(8, 0), pady=8)
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)

        def section(text, row):
            tk.Label(side, text=text, font=('Arial', 8, 'bold'), bg=_PANEL, fg=_SUB).grid(
                row=row, column=0, sticky='w', padx=12, pady=(10, 2))
            tk.Frame(side, bg=_BORDER, height=1).grid(row=row + 1, column=0, sticky='ew', padx=10)
            return row + 2

        row = section("CELLS (STAGE 5)", 0)
        self.info_label = tk.Label(side, text="", bg=_PANEL, fg=_SUB, font=('Arial', 8),
                                   justify='left', anchor='w', wraplength=262)
        self.info_label.grid(row=row, column=0, sticky='ew', padx=12, pady=(2, 2))
        row += 1

        list_frame = tk.Frame(side, bg=_CARD, highlightbackground=_BORDER, highlightthickness=1)
        list_frame.grid(row=row, column=0, sticky='nsew', padx=10, pady=(2, 4))
        side.rowconfigure(row, weight=1)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        row += 1
        scroll = tk.Scrollbar(list_frame, relief='flat', width=10)
        scroll.grid(row=0, column=1, sticky='ns')
        self.listbox = tk.Listbox(list_frame, yscrollcommand=scroll.set, selectmode=tk.BROWSE,
                                  exportselection=False, font=('Arial', 10), bg=_CARD, fg=_TEXT,
                                  selectbackground=_ACC, selectforeground='white', relief='flat',
                                  bd=0, highlightthickness=0, activestyle='none')
        self.listbox.grid(row=0, column=0, sticky='nsew')
        scroll.config(command=self.listbox.yview)
        self.listbox.bind('<<ListboxSelect>>', self._on_list_select)

        row = section("SIGNAL PROCESSING (STAGE 5)", row)
        proc = tk.Frame(side, bg=_PANEL)
        proc.grid(row=row, column=0, sticky='ew', padx=10, pady=(4, 2))
        proc.columnconfigure(1, weight=1)
        row += 1

        self._norm_keys = list(normalization.NORMALIZATION_METHODS)
        self._norm_labels = list(normalization.NORMALIZATION_METHODS.values())
        tk.Label(proc, text="Normalization:", bg=_PANEL, fg=_TEXT, font=('Arial', 9)).grid(
            row=0, column=0, columnspan=2, sticky='w')
        self.norm_var = tk.StringVar(value=self._norm_labels[0])
        ttk.Combobox(proc, textvariable=self.norm_var, values=self._norm_labels, state='readonly',
                     width=34).grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 4))
        tk.Label(proc, text="Lowest points:", bg=_PANEL, fg=_TEXT, font=('Arial', 9)).grid(
            row=2, column=0, sticky='w')
        self.points_var = tk.IntVar(value=normalization.DEFAULT_LOWEST_POINTS)
        ttk.Spinbox(proc, from_=2, to=50, textvariable=self.points_var, width=5).grid(
            row=2, column=1, sticky='w', padx=(6, 0))
        self.smooth_var = tk.BooleanVar(value=False)
        tk.Checkbutton(proc, text="Smoothing", variable=self.smooth_var, bg=_PANEL,
                       activebackground=_PANEL).grid(row=3, column=0, columnspan=2, sticky='w')
        self.subtract_var = tk.BooleanVar(value=False)
        tk.Checkbutton(proc, text="Subtract lowest-points baseline", variable=self.subtract_var,
                       bg=_PANEL, activebackground=_PANEL).grid(row=4, column=0, columnspan=2, sticky='w')

        row = section("PEAK FINDER", row)
        peak = tk.Frame(side, bg=_PANEL)
        peak.grid(row=row, column=0, sticky='ew', padx=10, pady=(4, 2))
        peak.columnconfigure(0, weight=1)
        row += 1
        self.peak_var = tk.StringVar(value=events.NO_PEAKS)
        peak_box = ttk.Combobox(peak, textvariable=self.peak_var,
                                values=[events.NO_PEAKS] + list(events.PEAK_METHODS),
                                state='readonly', width=22)
        peak_box.grid(row=0, column=0, sticky='ew')
        peak_box.bind('<<ComboboxSelected>>', lambda e: self._on_peak_method(ask=True))
        tk.Button(peak, text="Parameters...", command=lambda: self._on_peak_method(ask=True)).grid(
            row=0, column=1, padx=(6, 0))

        self.apply_button = tk.Button(side, text="Apply", command=self.apply, bg=_ACC, fg='white',
                                      activebackground='#1d4ed8', activeforeground='white',
                                      relief='flat', padx=10, pady=3)
        self.apply_button.grid(row=row, column=0, sticky='ew', padx=10, pady=(8, 2))
        row += 1
        tk.Label(side, text="Recomputes stage 5 for every cell (the same settings as the "
                            "Signals & Correlation tab).",
                 bg=_PANEL, fg=_SUB, font=('Arial', 8), justify='left', wraplength=262).grid(
            row=row, column=0, sticky='ew', padx=12, pady=(0, 10))

        plot = tk.Frame(body, bg=_PANEL, highlightbackground=_BORDER, highlightthickness=1)
        plot.grid(row=0, column=1, sticky='nsew', padx=8, pady=8)
        self.fig = Figure(figsize=(9, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------ data

    def _on_project_changed(self):
        if self.exists():
            self.refresh()

    def _stage_key(self):
        rec = self.project.get("signals")
        if rec is None or not rec.available:
            return None
        return (rec.fingerprint, rec.updated_at)

    def refresh(self, force=False):
        """Reload stage 5 from the project if it changed (always if `force`)
        and redraw, keeping the same cell selected when it is still there."""
        key = self._stage_key()
        if key == self._shown_key and not force:
            return
        self._shown_key = key
        rec = self.project.get("signals")
        self.raw = self.project.data("signals")
        if self.raw is None:
            self.columns, self.current = [], None
            self.norm = self.events = None
            self.peaks, self._lowest = {}, {}
            self.listbox.delete(0, tk.END)
            self.info_label.config(text="No signals yet.")
            self._set_controls(core_analysis.settings_from_params(None))
            self.draw()
            return

        norm = self.project.data("signals", "normalized_signals")
        if norm is None or list(norm.columns) != list(self.raw.columns) or len(norm) != len(self.raw):
            norm = self.raw
        self.norm = norm
        self.identity = norm is self.raw or norm.equals(self.raw)
        self.events = self.project.data("signals", "events")
        self.settings = core_analysis.settings_from_params(rec.params)
        self.description = core_analysis.describe(self.settings)
        if self.settings["peak_method"] in events.PEAK_METHODS:
            self.peak_params[self.settings["peak_method"]] = dict(self.settings["peak_params"])
        self.columns = list(self.raw.columns)
        self.peaks = events.events_by_column(self.events, self.columns) if self.events is not None else {}
        self._lowest = {}

        self.listbox.delete(0, tk.END)
        for col in self.columns:
            text = signal_label(col)
            if self.events is not None:
                n = len(self.peaks.get(col, ()))
                text += f"   ·  {n} peak{'s' if n != 1 else ''}"
            self.listbox.insert(tk.END, text)
        if self.current not in self.columns:
            self.current = self.columns[0] if self.columns else None
        if self.current is not None:
            i = self.columns.index(self.current)
            self.listbox.selection_set(i)
            self.listbox.activate(i)
            self.listbox.see(i)

        info = f"{len(self.columns)} cells · {len(self.raw)} frames · {self.description}"
        if self.events is not None:
            with_peaks = sum(1 for c in self.columns if len(self.peaks.get(c, ())))
            info += (f"\n{len(self.events)} peaks in {with_peaks} cells "
                     f"({self.settings['peak_method']})")
        else:
            info += "\nNo peak method chosen"
        if self.project.status("signals") == OUT_OF_DATE:
            info += "\nOut of date: recompute it (select the cells again, or Apply)."
        self.info_label.config(text=info)
        self._set_controls(self.settings)
        self.draw()

    def _set_controls(self, settings):
        self.norm_var.set(normalization.NORMALIZATION_METHODS.get(settings["method"], self._norm_labels[0]))
        self.points_var.set(int(settings["lowest_points"]))
        self.smooth_var.set(bool(settings["smooth"]))
        self.subtract_var.set(bool(settings["subtract"]))
        self.peak_var.set(settings["peak_method"] if settings["peak_method"] in events.PEAK_METHODS
                          else events.NO_PEAKS)
        self.apply_button.config(state='normal' if self.raw is not None else 'disabled')

    def _read_controls(self):
        label = self.norm_var.get()
        method = self._norm_keys[self._norm_labels.index(label)] if label in self._norm_labels else "none"
        try:
            points = int(self.points_var.get())
        except (tk.TclError, ValueError):
            points = 0
        if points < 2:
            points = normalization.DEFAULT_LOWEST_POINTS
        peak_method = self.peak_var.get()
        if peak_method not in events.PEAK_METHODS:
            peak_method = events.NO_PEAKS
        return {"method": method, "lowest_points": points, "smooth": bool(self.smooth_var.get()),
                "subtract": bool(self.subtract_var.get()), "peak_method": peak_method,
                "peak_params": dict(self.peak_params.get(peak_method, {}))}

    # ------------------------------------------------------------ actions

    def apply(self):
        """Recompute stage 5 with the sidebar's settings."""
        if self.raw is None:
            return
        self.apply_settings(self._read_controls())

    def _on_peak_method(self, ask=False):
        """A Peak Finder method was chosen (or Parameters... clicked): ask
        for its parameters, then recompute stage 5."""
        method = self.peak_var.get()
        if method in events.PEAK_METHODS and (ask or method not in self.peak_params):
            from peak_functions import show_parameter_dialog
            title, specs = events.PEAK_METHODS[method]
            current = {**events.default_params(method), **self.peak_params.get(method, {})}
            specs = [{**spec, 'default': current[spec['key']]} for spec in specs]
            new_params = show_parameter_dialog(self.tab.frame, title, specs)
            if new_params:
                self.peak_params[method] = new_params
            elif method not in self.peak_params:     # cancelled before ever choosing parameters
                self.peak_var.set(self.settings["peak_method"])
                return
        self.apply()

    def _on_list_select(self, _event=None):
        sel = self.listbox.curselection()
        if not sel or sel[0] >= len(self.columns):
            return
        col = self.columns[sel[0]]
        if col != self.current:
            self.current = col
            self.draw()

    def _step(self, delta):
        if not self.columns:
            return
        i = self.columns.index(self.current) if self.current in self.columns else 0
        i = min(max(i + delta, 0), len(self.columns) - 1)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(i)
        self.listbox.activate(i)
        self.listbox.see(i)
        self._on_list_select()

    # ------------------------------------------------------------ drawing

    def _lowest_points(self, col):
        if col not in self._lowest:
            try:
                used = core_analysis.lowest_points_used(self.raw[[col]], self.settings)
            except ValueError:      # e.g. a flat signal
                used = {}
            self._lowest[col] = used.get(col)
        return self._lowest[col]

    def draw(self):
        self.fig.clear()
        if self.current is None:
            ax = self.fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, "No signals in stage 5.\n\n"
                              "Select cells in a cell detection tab (their signals are extracted\n"
                              "automatically), or use File → Open Pipeline Stage → 5 · Signals.",
                    ha='center', va='center', fontsize=11, color='gray', transform=ax.transAxes)
            self.canvas.draw_idle()
            return

        col = self.current
        label = signal_label(col)
        raw = self.raw[col].to_numpy(dtype=float)
        processed = self.norm[col].to_numpy(dtype=float)
        t = np.arange(len(raw))
        pk = np.asarray(self.peaks.get(col, ()), dtype=int)
        pk = pk[(pk >= 0) & (pk < len(raw))] if self.show_peaks_var.get() else np.array([], dtype=int)
        px = self._lowest_points(col) if self.show_points_var.get() else None
        if px is not None:
            px = np.asarray(px, dtype=int)
            px = px[(px >= 0) & (px < len(raw))]

        ax_raw, ax_proc = self.fig.subplots(2, 1, sharex=True)

        ax_raw.plot(t, raw, color=_RAW_COLOR, linewidth=0.9, label='Raw signal')
        if len(pk):
            ax_raw.scatter(pk, raw[pk], color=_PEAK_COLOR, s=28, zorder=5, label='Peaks')
        if px is not None and len(px) and core_analysis.baseline_on_raw(self.settings):
            baseline, _ = normalization.lowest_points_baseline(raw, int(self.settings["lowest_points"]))
            ax_raw.plot(t, baseline, color=_BASELINE_COLOR, linewidth=1.2, linestyle='--', label='Baseline')
            ax_raw.scatter(px, raw[px], color=_BASELINE_COLOR, s=25, zorder=5, label='Lowest points used')
        title = f"{label} — raw signal"
        if self.events is not None:
            n = len(self.peaks.get(col, ()))
            title += f"  ·  {n} peak{'s' if n != 1 else ''} ({self.settings['peak_method']})"
        ax_raw.set_title(title, fontsize=10)
        ax_raw.set_ylabel('Average intensity', fontsize=9)
        ax_raw.grid(True, alpha=0.3)

        ax_proc.plot(t, processed, color=_PROCESSED_COLOR, linewidth=0.9, label='Processed signal')
        if len(pk):
            ax_proc.scatter(pk, processed[pk], color=_PEAK_COLOR, s=28, zorder=5, label='Peaks')
        if px is not None and len(px):
            ax_proc.plot(px, processed[px], color='purple', linewidth=1.1, linestyle='--')
            ax_proc.scatter(px, processed[px], color='purple', s=25, zorder=5, label='Lowest points used')
        if self.identity:
            ax_proc.set_title(f"{label} — processed (no normalization: same as raw)", fontsize=10)
            ax_proc.set_ylabel('Average intensity', fontsize=9)
        else:
            ax_proc.set_title(f"{label} — processed signal\n{self.description}", fontsize=10)
            ax_proc.set_ylabel('Normalized value', fontsize=9)
        ax_proc.set_xlabel('Frame', fontsize=9)
        ax_proc.grid(True, alpha=0.3)

        for ax in (ax_raw, ax_proc):
            ax.tick_params(labelsize=8)
            if len(ax.get_legend_handles_labels()[0]) > 1:
                ax.legend(fontsize=7, loc='best', framealpha=0.8)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------ saving

    def save_plot(self):
        if self.current is None:
            return
        path = asksaveasfilename(parent=self.tab.frame, title="Save Plot of This Cell",
                                 defaultextension=".png", filetypes=IMAGE_TYPES,
                                 initialfile=f"{self.current}.png")
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches='tight')

    def cell_peaks_table(self, col=None):
        """The peaks of one cell: frame, raw value and processed value."""
        col = self.current if col is None else col
        pk = np.asarray(self.peaks.get(col, ()), dtype=int)
        raw = self.raw[col].to_numpy(dtype=float)
        processed = self.norm[col].to_numpy(dtype=float)
        pk = pk[(pk >= 0) & (pk < len(raw))]
        return pd.DataFrame({"frame": pk, "raw_value": raw[pk], "processed_value": processed[pk]})

    def save_cell_peaks(self):
        if self.current is None:
            return
        if self.events is None:
            messagebox.showwarning("No Peaks", "Choose a Peak Finder method first.", parent=self.tab.frame)
            return
        path = asksaveasfilename(parent=self.tab.frame, title="Save This Cell's Peaks",
                                 defaultextension=".csv", initialfile=f"{self.current}_peaks.csv",
                                 filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("All files", "*.*")])
        if not path:
            return
        table = self.cell_peaks_table()
        if path.lower().endswith('.xlsx'):
            table.to_excel(path, index=False)
        else:
            table.to_csv(path, index=False)

    def _export(self, item_key):
        data = self.events if item_key == "events" else self.norm
        if data is None:
            messagebox.showwarning("Nothing to Export",
                                   "Choose a Peak Finder method first." if item_key == "events"
                                   else "There are no signals in stage 5.", parent=self.tab.frame)
            return
        it = core_stages.item(item_key)
        path = asksaveasfilename(parent=self.tab.frame, title=f"Export {it.title}",
                                 defaultextension=".csv", initialfile=it.default_name,
                                 filetypes=core_stages.save_filetypes(item_key))
        if not path:
            return
        try:
            core_stages.save_item(item_key, data, path)
        except Exception as e:
            messagebox.showerror("Can't Save", str(e), parent=self.tab.frame)
