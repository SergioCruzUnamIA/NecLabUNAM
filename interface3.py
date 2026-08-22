"""
NecLab - Microscopy image analysis and data visualization tool
Main graphical interface (unified version)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # Limit number of threads

import tkinter as tk
from tkinter import Menu, Grid, filedialog, FALSE, DISABLED, NORMAL, ttk, messagebox
import tkinter.font as tkfont
import customtkinter as ctk
ctk.set_appearance_mode("light")
from PIL import Image, ImageTk, ImageOps
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import threading
import queue
import subprocess
import json
import urllib.request

# Axes margins (in inches, not a fixed fraction of the figure) for the Multiple
# Files top line plot and bottom heatmap. Left/right are shared by both
# so a sample's x-position lines up vertically between the two regardless of
# the heatmap's colorbar taking extra space on its side. Using inches instead
# of a fixed fraction keeps the margins just large enough for their content
# (axis label, title, colorbar) instead of eating a growing chunk of the
# figure as the panel gets wider.
_MULTI_XLS_LEFT_IN = 0.62             # y-axis label + tick numbers
_MULTI_XLS_RIGHT_IN_CBAR = 0.95       # reserved when the heatmap colorbar
                                       # will be drawn (kept on the line plot
                                       # too, which has none, so both x-axes
                                       # stay aligned)
_MULTI_XLS_RIGHT_IN_PLAIN = 0.15      # reserved when no colorbar will be
                                       # drawn this redraw (individual scale
                                       # per sheet): just a hair of breathing
                                       # room instead of a full colorbar's worth
_MULTI_XLS_TOP_IN = 0.35              # title
_MULTI_XLS_BOTTOM_IN_LABELS = 0.85    # rotated per-sheet name labels
_MULTI_XLS_BOTTOM_IN_NOLABELS = 0.22  # no labels: just the tick marks
_MULTI_XLS_CBAR_GAP_IN = 0.15         # gap between axes and colorbar
_MULTI_XLS_CBAR_WIDTH_IN = 0.22       # colorbar width

# 'jet' with black prepended at the very bottom of the range, so the lowest
# values in the Multiple Files heatmap render as black instead of jet's dark
# blue - a smooth gradient (black -> blue -> cyan -> green -> yellow -> red).
_MULTI_XLS_HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'jet_black', np.vstack(([[0, 0, 0, 1]], plt.cm.jet(np.linspace(0, 1, 256)))))


def _multi_xls_axes_margins(fig_width, fig_height, show_labels, reserve_colorbar):
    """(left, right, top, bottom) axes-position fractions for the Multiple
    Files line plot / heatmap figures, computed from constant margins in
    inches so the plotted area keeps expanding to fill the panel as it grows
    instead of leaving a fixed fraction of it unused. `reserve_colorbar`
    controls whether the right margin needs to fit the heatmap's colorbar
    (shared_scale on) or can shrink to a minimal margin (colorbar hidden)."""
    left = _MULTI_XLS_LEFT_IN / fig_width
    right_in = _MULTI_XLS_RIGHT_IN_CBAR if reserve_colorbar else _MULTI_XLS_RIGHT_IN_PLAIN
    right = 1 - right_in / fig_width
    top = 1 - _MULTI_XLS_TOP_IN / fig_height
    bottom_in = _MULTI_XLS_BOTTOM_IN_LABELS if show_labels else _MULTI_XLS_BOTTOM_IN_NOLABELS
    bottom = bottom_in / fig_height

    # Guard against degenerate figures (e.g. a panel dragged down to its
    # minimum height): if the fixed top/bottom margins would leave no room
    # for the plot itself, shrink them proportionally instead of handing
    # matplotlib an invalid (bottom >= top) subplot rectangle.
    min_plot_frac = 0.15
    if bottom >= top - min_plot_frac:
        total = bottom + (1 - top)
        scale = (1 - min_plot_frac) / total if total > 0 else 0
        bottom *= scale
        top = 1 - (1 - top) * scale

    return left, right, top, bottom

# ── L1 Sky Blue colour palette ────────────────────────────────────────────────
_C = {
    'bg':     '#f0f4f8',   # main background
    'panel':  '#ffffff',   # white sidebar / header panels
    'card':   '#f8fafc',   # listbox / inner card background
    'acc':    '#2563eb',   # blue accent (buttons, active tab)
    'acc2':   '#1d4ed8',   # darker blue (hover)
    'text':   '#1e293b',   # primary text
    'sub':    '#94a3b8',   # secondary / label text
    'border': '#e2e8f0',   # divider lines
}

# Local modules
from pyometiff import OMETIFFReader
from variability_functions import show_variability_analysis, get_variability_methods
from corr_dendo_functions import load_correlation_matrix
from multi_xls_helpers import (pick_files_and_sheets, load_selected_sheets, common_column_names,
                                load_single_data_file)

# Try to import image processing modules if they exist
try:
    from image_loader import load_ometiff_image, process_image_slice
    from image_processing import auto_contrast, threshold_image_pil
    HAS_IMAGE_MODULES = True
except ImportError:
    HAS_IMAGE_MODULES = False



class NecLabApp:
    """Main class of the NecLab application - Unified version."""

    # Parameter specs for each Peak Finder method (used by the parameter
    # dialog and the per-method params cache in the Multiple Files tab).
    _PEAK_PARAM_SPECS = {
        'Elliptic Envelope': ('Elliptic Envelope Parameters', [
            {'name': 'Contamination', 'key': 'contamination', 'default': 0.01, 'type': float},
        ]),
        'Peak Caller': ('Peak Caller Parameters', [
            {'name': 'Rise %', 'key': 'rise_percent', 'default': 5, 'type': int},
            {'name': 'Fall %', 'key': 'fall_percent', 'default': 5, 'type': int},
            {'name': 'Max Lookback', 'key': 'max_lookback', 'default': 10, 'type': int},
            {'name': 'Max Lookahead', 'key': 'max_lookahead', 'default': 10, 'type': int},
        ]),
        'Local Outlier Factor': ('Local Outlier Factor Parameters', [
            {'name': 'N Neighbors', 'key': 'n_neighbors', 'default': 20, 'type': int},
        ]),
        'Peak Function 4': ('Peak Function 4 (Elliptic Envelope + SVR) Parameters', [
            {'name': 'Contamination', 'key': 'contamination', 'default': 0.01, 'type': float},
        ]),
        'Isolation Forest': ('Isolation Forest Parameters', [
            {'name': 'Contamination', 'key': 'contamination', 'default': 0.05, 'type': float},
        ]),
        'Linear Model': ('Linear Model (SGDOneClassSVM) Parameters', [
            {'name': 'Nu', 'key': 'nu', 'default': 0.131, 'type': float},
        ]),
        'Peak Function 7': ('Peak Function 7 (Lasso + LOF) Parameters', [
            {'name': 'N Neighbors', 'key': 'n_neighbors', 'default': 20, 'type': int},
        ]),
    }

    def __init__(self, root):
        self.root = root
        self.root.title("NecLab - Image and Data Analysis")
        self.root.tk.call('tk', 'windowingsystem')
        self.root.option_add('*tearOff', FALSE)

        # Configure window size (90% of the screen)
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.width = int(self.screen_width * 0.9)
        self.height = int(self.screen_height * 0.9)
        self.root.geometry(f"{self.width}x{self.height}")

        # State variables - Images
        self.img_original = None  # Original, unmodified image
        self.img_array = None     # Working image (may have modifications)
        self.img_display = None   # Image for display (with contrast, etc.)

        # State variables - Multiple Files tab (multiple .xls files)
        self.multi_xls_tab = None
        self.multi_xls_datasets = []
        self.multi_xls_common_columns = []
        self.multi_xls_column_listbox = None
        self.multi_xls_show_labels_var = tk.BooleanVar(value=False)
        self.multi_xls_smoothing_var = tk.BooleanVar(value=True)
        self.multi_xls_smoothing_points_var = tk.IntVar(value=2)
        self.multi_xls_shared_scale_var = tk.BooleanVar(value=True)
        # None = auto (per-sheet or shared-scale range, per the checkbox
        # above); (min, max) = manual override set via "Color Limits
        # (Heatmap)...", applied to every sheet's heatmap image.
        self.multi_xls_heatmap_manual_range = None
        # 'local': min of this column, within each sheet (default). 'sheet':
        # min across all columns of each sheet (matches the heatmap's
        # normalization). 'column_global': min of this column pooled across
        # every loaded sheet, so all sheets share one divisor.
        self.multi_xls_norm_mode_var = tk.StringVar(value='local')
        self.multi_xls_xlim = None
        self.multi_xls_ylim = None
        self.multi_xls_plot_frame = None
        self.multi_xls_fig = None
        self.multi_xls_current_column = None
        self.multi_xls_current_index = None
        self.multi_xls_heatmap_frame = None
        self.multi_xls_heatmap_fig = None
        self._multi_xls_plot_resize_job = None
        self._multi_xls_heatmap_resize_job = None
        self._multi_xls_sash_dragging = False
        # Cache the expensive per-sheet data processing (interpolation,
        # normalization, smoothing / heatmap normalization) so a
        # resize-triggered redraw - which doesn't change any of that, only
        # the figure size - can skip straight to replotting instead of
        # redoing it from scratch. Keyed so loading new files, changing the
        # selected column, or toggling any relevant setting invalidates it.
        self._multi_xls_series_cache_key = None
        self._multi_xls_series_cache = None
        self._multi_xls_heatmap_matrices_cache_key = None
        self._multi_xls_heatmap_matrices_cache = None
        self.multi_xls_menu_grafica = None
        self.multi_xls_menu_datos = None
        self.multi_xls_class_row = None
        self.multi_xls_plot_placeholder = None
        self.multi_xls_heatmap_placeholder = None
        # Each plot's canvas/figure/axes are created only once and reused on
        # every redraw (instead of being destroyed and recreated each time),
        # and the size is updated with forward=False: this way Tk is never
        # told to resize the canvas widget, which is what triggered a new
        # <Configure> event on every redraw and caused an infinite redraw
        # loop.
        self._multi_xls_plot_canvas = None
        self._multi_xls_plot_ax = None
        self._multi_xls_heatmap_canvas = None
        self._multi_xls_heatmap_ax = None
        self._multi_xls_heatmap_colorbar = None
        self.multi_xls_classes = []           # names available for classifying sheets (cosmetic)
        self.multi_xls_sheet_class_var = {}    # sheet label -> tk.StringVar with the chosen classification
        self.multi_xls_class_combos = {}       # sheet label -> ttk.Combobox (positioned over its segment)
        self.multi_xls_classifications = {}    # col_index -> {sheet label -> saved classification}
        self._multi_xls_syncing_class_row = False

        # State variables - Multiple Files tab: Peak Finder / Smoothing points overlay
        self.multi_xls_peak_method_var = tk.StringVar(value='None')
        self.multi_xls_peak_method_combo = None
        self.multi_xls_peak_method_params = {}  # saved params per method name
        self.multi_xls_show_smoothing_points_var = tk.BooleanVar(value=True)

        # State variables - Multiple Files tab: Correlation + Selection
        self.multi_xls_corr_method_var = tk.StringVar(value='pearson')
        self.multi_xls_show_corr_labels_var = tk.BooleanVar(value=True)
        self.multi_xls_selection_listbox = None
        self.multi_xls_selection_indices = []
        self.multi_xls_correlation_frame = None
        self._multi_xls_corr_fig = None
        self._multi_xls_corr_df = None
        self.btn_multi_xls_add_sel = None
        self.btn_multi_xls_remove_sel = None
        self.multi_xls_right_paned = None
        # Middle (heatmap) and bottom (correlation) panes are hidden by
        # default - only the top line plot shows until one of these View
        # menu checkboxes is turned on.
        self.multi_xls_show_heatmap_var = tk.BooleanVar(value=False)
        self.multi_xls_show_correlation_var = tk.BooleanVar(value=False)

        # State variables - Dendrogram tab
        self.dendo_tab = None
        self.dendo_column_listbox = None
        self.dendo_selection_listbox = None
        self.dendo_selection_indices = []
        self.dendo_plot_frame = None
        self.dendo_top_frame = None
        self.dendo_bottom_frame = None
        self.dendo_fig = None
        self.dendo_signal_fig = None
        self.dendo_current_column = 0
        self._dendo_mouse_click = False
        self.btn_dendo_add_sel = None
        self.btn_dendo_remove_sel = None
        self.btn_dendo_save_img = None
        self.btn_dendo_save_csv = None

        # Build the interface
        self._create_menu()
        self._create_layout()

        # Handle window close to release the entire process
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.lift()
        self.root.focus_force()

    def _on_close(self):
        """Fully close the application, releasing all resources."""
        if messagebox.askokcancel("Exit", "Do you want to exit NecLab?"):
            plt.close('all')
            self.root.quit()
            self.root.destroy()
            sys.exit(0)

    # ==================== MENU ====================

    def _create_menu(self):
        """Create the unified menu bar."""
        self.menu_bar = Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # File menu
        self.menu_archivo = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_archivo, label="File")
        self.menu_archivo.add_command(
            label="Open OME-TIFF",
            accelerator="Ctrl+O",
            command=self.open_ometiff_file
        )
        self.menu_archivo.add_separator()
        self.menu_archivo.add_command(
            label='Open Data (.npy / .csv / .xlsx)',
            command=self.open_single_data_file,
            state=NORMAL
        )
        self.menu_archivo.add_command(
            label='Load Correlation Matrix',
            command=self.load_correlation_matrix_wrapper,
            state=NORMAL
        )
        self.menu_archivo.add_separator()
        self.menu_archivo.add_command(
            label='Open Multiple Files (.xls / .csv)',
            command=self.open_multiple_xls_files,
            state=NORMAL
        )
        self.menu_archivo.add_separator()
        self.menu_archivo.add_command(
            label="Exit",
            command=self._on_close
        )

        # Image menu
        self.menu_imagen = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_imagen, label="Image")
        self.menu_imagen.add_command(
            label="Auto Contrast",
            command=self.apply_auto_contrast,
            state=DISABLED
        )
        self.menu_imagen.add_command(
            label="Histogram",
            command=self.show_histogram,
            state=DISABLED
        )
        self.menu_imagen.add_command(
            label="Binarize",
            command=self.show_binarize,
            state=DISABLED
        )
        self.menu_imagen.add_separator()
        self.menu_imagen.add_command(
            label="Restore Original",
            command=self.restore_original,
            state=DISABLED
        )

        # Variability Analysis menu (top-level, between Image and Visualization)
        self.menu_variabilidad = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_variabilidad, label="Variability Analysis", state=DISABLED)

        # Add the 7 variability methods
        methods = get_variability_methods()
        for i, method_name in enumerate(methods):
            self.menu_variabilidad.add_command(
                label=method_name,
                command=lambda idx=i: self.show_variability_menu(idx)
            )

        # Visualization menu
        self.menu_visual = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_visual, label="Visualization")

        self.menu_visual.add_command(label='Dendrogram', command=None, state=DISABLED)
        self.menu_visual.add_separator()
        self.menu_visual.add_command(label='Time Series', command=None, state=DISABLED)

        # Help menu
        self.menu_ayuda = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_ayuda, label="Help")
        self.menu_ayuda.add_command(label='Check for Updates', command=self._check_for_updates)
        self.menu_ayuda.add_separator()
        self.menu_ayuda.add_command(label='About NecLab', command=self._show_about)

        # Keyboard shortcut
        self.root.bind('<Control-o>', lambda e: self.open_ometiff_file())

    # ==================== MAIN LAYOUT ====================

    def _create_layout(self):
        """Create the main layout with tabs for different modes."""
        self.root.configure(bg=_C['bg'])

        # Thin accent line across the top of the content area
        tk.Frame(self.root, bg=_C['acc'], height=2).pack(fill='x')

        # Style the ttk.Notebook
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure('L1.TNotebook',
                        background=_C['bg'], borderwidth=0, tabposition='n')
        style.configure('L1.TNotebook.Tab',
                        background=_C['card'], foreground=_C['sub'],
                        padding=[18, 8], font=('Arial', 10),
                        borderwidth=0, relief='flat')
        style.map('L1.TNotebook.Tab',
                  background=[('selected', _C['panel'])],
                  foreground=[('selected', _C['acc'])],
                  expand=[('selected', [0, 0, 0, 0])])

        self.notebook = ttk.Notebook(self.root, style='L1.TNotebook')
        self.notebook.pack(fill='both', expand=True)

        # Tab 1: Image Processing
        self.image_tab = tk.Frame(self.notebook, bg=_C['bg'])
        self.notebook.add(self.image_tab, text="  Image Processing  ")
        self._create_image_processing_layout()


    def _create_image_processing_layout(self):
        """Create the layout for image processing."""
        self.image_tab.columnconfigure(0, weight=3)
        self.image_tab.columnconfigure(1, weight=1)
        self.image_tab.rowconfigure(0, weight=1)

        self.image_frame = tk.Frame(self.image_tab, bg='#0f172a',
                                    highlightbackground=_C['border'],
                                    highlightthickness=1)
        self.image_frame.grid(row=0, column=0, sticky='nsew', padx=(8, 4), pady=8)

        self.image_label = tk.Label(self.image_frame, bg='#0f172a')
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self._create_image_controls_panel()
        self._load_default_image()

    def _create_image_controls_panel(self):
        """Create the right-side panel with image controls."""
        self.controls_panel = tk.Frame(self.image_tab, bg=_C['panel'],
                                       highlightbackground=_C['border'],
                                       highlightthickness=1)
        self.controls_panel.grid(row=0, column=1, sticky='nsew', padx=(4, 8), pady=8)

        tk.Label(self.controls_panel, text="Image Controls",
                 font=('Arial', 13, 'bold'), bg=_C['panel'], fg=_C['text']).pack(pady=12)

        tk.Frame(self.controls_panel, bg=_C['border'], height=1).pack(fill='x', padx=10)

        # ===== SECTION: Navigation =====
        self._create_section_navigation()

        # ===== SECTION: Image Adjustments =====
        self._create_section_image_adjustments()

        # ===== SECTION: Processing =====
        self._create_section_processing()

        # ===== SECTION: Information =====
        self._create_section_info()

    def _create_section_navigation(self):
        """Frame navigation section."""
        def _sec_label(parent, text):
            tk.Label(parent, text=text, font=('Arial', 8, 'bold'),
                     bg=_C['panel'], fg=_C['sub']).pack(anchor='w', padx=12, pady=(12, 2))
            tk.Frame(parent, bg=_C['border'], height=1).pack(fill='x', padx=10)

        _sec_label(self.controls_panel, "NAVIGATION")

        inner = tk.Frame(self.controls_panel, bg=_C['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        tk.Label(inner, text="Layer (Frame):", bg=_C['panel'],
                 fg=_C['text'], font=('Arial', 9)).pack(anchor='w')

        self.slice_slider = tk.Scale(inner, from_=0, to=0, orient="horizontal",
                                     command=self._on_slice_changed,
                                     bg=_C['panel'], fg=_C['text'],
                                     troughcolor=_C['card'], highlightthickness=0,
                                     relief='flat', sliderlength=16)
        self.slice_slider.pack(fill='x')

        self.frame_info_label = tk.Label(inner, text="Frame: 0 / 0",
                                          bg=_C['panel'], fg=_C['sub'], font=('Arial', 9))
        self.frame_info_label.pack(anchor='w')

    def _create_section_image_adjustments(self):
        """Image adjustments section."""
        def _sec_label(text):
            tk.Label(self.controls_panel, text=text, font=('Arial', 8, 'bold'),
                     bg=_C['panel'], fg=_C['sub']).pack(anchor='w', padx=12, pady=(12, 2))
            tk.Frame(self.controls_panel, bg=_C['border'], height=1).pack(fill='x', padx=10)

        _sec_label("IMAGE ADJUSTMENTS")
        inner = tk.Frame(self.controls_panel, bg=_C['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        tk.Label(inner, text="Brightness:", bg=_C['panel'], fg=_C['text'], font=('Arial', 9)).pack(anchor='w')
        self.brightness_slider = tk.Scale(inner, from_=-100, to=100, orient="horizontal",
                                          command=self._on_adjustment_changed,
                                          bg=_C['panel'], troughcolor=_C['card'],
                                          highlightthickness=0, relief='flat', sliderlength=16)
        self.brightness_slider.set(0)
        self.brightness_slider.pack(fill='x')

        tk.Label(inner, text="Contrast:", bg=_C['panel'], fg=_C['text'],
                 font=('Arial', 9)).pack(anchor='w', pady=(5, 0))
        self.contrast_slider = tk.Scale(inner, from_=-100, to=100, orient="horizontal",
                                        command=self._on_adjustment_changed,
                                        bg=_C['panel'], troughcolor=_C['card'],
                                        highlightthickness=0, relief='flat', sliderlength=16)
        self.contrast_slider.set(0)
        self.contrast_slider.pack(fill='x')

        ctk.CTkButton(inner, text="Auto Contrast", height=30, corner_radius=6,
                      fg_color=_C['acc'], hover_color=_C['acc2'], text_color='white',
                      font=ctk.CTkFont(size=11),
                      command=self.apply_auto_contrast).pack(fill='x', pady=(6, 2))
        ctk.CTkButton(inner, text="Reset Adjustments", height=30, corner_radius=6,
                      fg_color=_C['card'], hover_color=_C['border'],
                      text_color=_C['text'], border_width=1, border_color=_C['border'],
                      font=ctk.CTkFont(size=11),
                      command=self._reset_adjustments).pack(fill='x', pady=2)

    def _create_section_processing(self):
        """Processing section."""
        def _sec_label(text):
            tk.Label(self.controls_panel, text=text, font=('Arial', 8, 'bold'),
                     bg=_C['panel'], fg=_C['sub']).pack(anchor='w', padx=12, pady=(12, 2))
            tk.Frame(self.controls_panel, bg=_C['border'], height=1).pack(fill='x', padx=10)

        _sec_label("PROCESSING")
        inner = tk.Frame(self.controls_panel, bg=_C['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        row = tk.Frame(inner, bg=_C['panel'])
        row.pack(fill='x')
        tk.Label(row, text="Threshold:", bg=_C['panel'], fg=_C['text'], font=('Arial', 9)).pack(side='left')
        self.threshold_enabled = tk.BooleanVar(value=False)
        self.threshold_check = tk.Checkbutton(row, text="Apply", variable=self.threshold_enabled,
                                               command=self._update_image_display,
                                               bg=_C['panel'], fg=_C['text'],
                                               selectcolor=_C['card'], activebackground=_C['panel'])
        self.threshold_check.pack(side='right')

        self.threshold_slider = tk.Scale(inner, from_=0, to=255, orient="horizontal",
                                         command=self._on_threshold_changed,
                                         bg=_C['panel'], troughcolor=_C['card'],
                                         highlightthickness=0, relief='flat', sliderlength=16)
        self.threshold_slider.set(128)
        self.threshold_slider.pack(fill='x')

    def _create_section_info(self):
        """Image information section."""
        def _sec_label(text):
            tk.Label(self.controls_panel, text=text, font=('Arial', 8, 'bold'),
                     bg=_C['panel'], fg=_C['sub']).pack(anchor='w', padx=12, pady=(12, 2))
            tk.Frame(self.controls_panel, bg=_C['border'], height=1).pack(fill='x', padx=10)

        _sec_label("INFORMATION")
        inner = tk.Frame(self.controls_panel, bg=_C['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        self.info_text = tk.Label(inner, text="No image loaded", bg=_C['panel'],
                                   fg=_C['text'], justify='left', anchor='w', font=('Arial', 9))
        self.info_text.pack(fill='x')

    def open_single_data_file(self):
        """Open a single .npy/.csv/.xlsx/.xls file straight into the
        Multiple Files tab (as a one-sheet load), so a lone file uses the
        same pipeline/UI as multiple ones instead of a separate tab."""
        dataset = load_single_data_file(self.root)
        if not dataset:
            return
        self._on_multi_xls_load_complete(dataset)

    def _run_dendogram_on_selection(self):
        """Create the Dendrogram tab on first use, then switch to it."""
        if self.dendo_tab is None or not self.dendo_tab.winfo_exists():
            self.dendo_tab = tk.Frame(self.notebook, bg=_C['bg'])
            self.notebook.add(self.dendo_tab, text="Dendrogram")
            # Reset all dendo state so _create_dendogram_layout starts fresh
            self.dendo_column_listbox = None
            self.dendo_selection_listbox = None
            self.dendo_selection_indices = []
            self.dendo_plot_frame = None
            self.dendo_top_frame = None
            self.dendo_bottom_frame = None
            self.dendo_fig = None
            self.dendo_signal_fig = None
            self.dendo_current_column = 0
            self.btn_dendo_add_sel = None
            self.btn_dendo_remove_sel = None
            self.btn_dendo_save_img = None
            self.btn_dendo_save_csv = None
            self._create_dendogram_layout()
            self._dendo_populate_columns()
            if self.multi_xls_common_columns:
                self.btn_dendo_add_sel.configure(state='normal')
                self.btn_dendo_remove_sel.configure(state='normal')
                self.btn_dendo_save_img.configure(state='normal')
                self.btn_dendo_save_csv.configure(state='normal')
        self.notebook.select(self.dendo_tab)

    # ==================== DENDOGRAM TAB ====================

    def _create_dendogram_layout(self):
        """Build the permanent Dendrogram tab (sidebar + plot area)."""
        Grid.rowconfigure(self.dendo_tab, 0, weight=1)
        Grid.columnconfigure(self.dendo_tab, 0, weight=0)
        Grid.columnconfigure(self.dendo_tab, 1, weight=1)

        # ── Sidebar ──
        sidebar = tk.Frame(self.dendo_tab, bg=_C['panel'], width=250,
                           highlightbackground=_C['border'], highlightthickness=1)
        sidebar.grid(row=0, column=0, sticky='nsew', padx=(8, 0), pady=8)
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)

        def _dsec(text, row):
            tk.Label(sidebar, text=text, font=('Arial', 8, 'bold'),
                     bg=_C['panel'], fg=_C['sub']).grid(
                row=row, column=0, sticky='w', padx=12, pady=(12, 2))
            row += 1
            tk.Frame(sidebar, bg=_C['border'], height=1).grid(
                row=row, column=0, sticky='ew', padx=10)
            return row + 1

        drow = _dsec("DATA COLUMNS", 0)

        lb_frame = tk.Frame(sidebar, bg=_C['card'],
                            highlightbackground=_C['border'], highlightthickness=1)
        lb_frame.grid(row=drow, column=0, sticky='nsew', padx=10, pady=(2, 4))
        sidebar.rowconfigure(drow, weight=2)
        lb_frame.rowconfigure(0, weight=1)
        lb_frame.columnconfigure(0, weight=1)
        drow += 1

        lb_sb = tk.Scrollbar(lb_frame, relief='flat', width=10)
        lb_sb.grid(row=0, column=1, sticky='ns')
        self.dendo_column_listbox = tk.Listbox(
            lb_frame, yscrollcommand=lb_sb.set,
            selectmode=tk.EXTENDED, font=('Arial', 10),
            bg=_C['card'], fg=_C['text'],
            selectbackground=_C['acc'], selectforeground='white',
            relief='flat', bd=0, highlightthickness=0, activestyle='none'
        )
        self.dendo_column_listbox.grid(row=0, column=0, sticky='nsew')
        lb_sb.config(command=self.dendo_column_listbox.yview)

        drow = _dsec("SELECTION", drow)

        sel_frame = tk.Frame(sidebar, bg=_C['card'],
                             highlightbackground=_C['border'], highlightthickness=1)
        sel_frame.grid(row=drow, column=0, sticky='nsew', padx=10, pady=(2, 4))
        sidebar.rowconfigure(drow, weight=1)
        sel_frame.rowconfigure(0, weight=1)
        sel_frame.columnconfigure(0, weight=1)
        drow += 1

        sel_sb = tk.Scrollbar(sel_frame, relief='flat', width=10)
        sel_sb.grid(row=0, column=1, sticky='ns')
        self.dendo_selection_listbox = tk.Listbox(
            sel_frame, yscrollcommand=sel_sb.set,
            selectmode=tk.SINGLE, font=('Arial', 10),
            bg=_C['card'], fg=_C['text'],
            selectbackground=_C['acc'], selectforeground='white',
            relief='flat', bd=0, highlightthickness=0, activestyle='none'
        )
        self.dendo_selection_listbox.grid(row=0, column=0, sticky='nsew')
        sel_sb.config(command=self.dendo_selection_listbox.yview)

        self.btn_dendo_add_sel = ctk.CTkButton(
            sidebar, text="Add to Selection", height=28, corner_radius=6,
            fg_color=_C['acc'], hover_color=_C['acc2'], text_color='white',
            font=ctk.CTkFont(size=11), state='disabled',
            command=self._dendo_add_to_selection
        )
        self.btn_dendo_add_sel.grid(row=drow, column=0, sticky='ew', padx=10, pady=(2, 2))
        drow += 1

        self.btn_dendo_remove_sel = ctk.CTkButton(
            sidebar, text="Remove from Selection", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            state='disabled', command=self._dendo_remove_from_selection
        )
        self.btn_dendo_remove_sel.grid(row=drow, column=0, sticky='ew', padx=10, pady=(0, 4))
        drow += 1

        tk.Frame(sidebar, bg=_C['border'], height=1).grid(
            row=drow, column=0, sticky='ew', padx=10, pady=4)
        drow += 1

        self.btn_dendo_save_img = ctk.CTkButton(
            sidebar, text="Save Dendrogram Image", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            state='disabled', command=self._dendo_save_image
        )
        self.btn_dendo_save_img.grid(row=drow, column=0, sticky='ew', padx=10, pady=(2, 2))
        drow += 1

        self.btn_dendo_save_csv = ctk.CTkButton(
            sidebar, text="Save Dendrogram CSV", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            state='disabled', command=self._dendo_save_csv
        )
        self.btn_dendo_save_csv.grid(row=drow, column=0, sticky='ew', padx=10, pady=(0, 10))

        # ── Plot area (top: signal preview, bottom: dendrogram) ──
        self.dendo_plot_frame = tk.Frame(self.dendo_tab, bg=_C['panel'],
                                        highlightbackground=_C['border'], highlightthickness=1)
        self.dendo_plot_frame.grid(row=0, column=1, sticky='nsew', padx=8, pady=8)
        self.dendo_plot_frame.rowconfigure(0, weight=1)
        self.dendo_plot_frame.rowconfigure(1, weight=1)
        self.dendo_plot_frame.columnconfigure(0, weight=1)

        self.dendo_top_frame = tk.Frame(self.dendo_plot_frame, bg=_C['panel'])
        self.dendo_top_frame.grid(row=0, column=0, sticky='nsew')
        tk.Label(
            self.dendo_top_frame,
            text="Click a column to view its signal",
            font=('Arial', 14), bg=_C['panel'], fg=_C['sub']
        ).pack(fill=tk.BOTH, expand=True)

        tk.Frame(self.dendo_plot_frame, bg=_C['border'], height=1).grid(
            row=1, column=0, sticky='ew')

        self.dendo_bottom_frame = tk.Frame(self.dendo_plot_frame, bg=_C['panel'])
        self.dendo_bottom_frame.grid(row=2, column=0, sticky='nsew')
        self.dendo_plot_frame.rowconfigure(2, weight=1)
        tk.Label(
            self.dendo_bottom_frame,
            text="Add 2+ columns to Selection to see the dendrogram",
            font=('Arial', 14), bg=_C['panel'], fg=_C['sub']
        ).pack(fill=tk.BOTH, expand=True)

        # Bind column click to signal preview
        self.dendo_column_listbox.bind('<ButtonPress-1>', lambda e: setattr(self, '_dendo_mouse_click', True))
        self.dendo_column_listbox.bind('<<ListboxSelect>>', self._dendo_show_signal)
        self.dendo_column_listbox.bind('<ButtonRelease-1>', self._dendo_on_column_click)

    def _dendo_build_matrix(self, column_indices):
        """Build a (samples x columns) array by concatenating, for each
        column index in 'column_indices', its cross-sheet series in the
        same order/segments as the Multiple Files plot (see
        _multi_xls_concatenated_column). Columns are trimmed to the
        shortest one, which only differs when a column is missing from a
        sheet that contributes to another (both are 'common' columns
        normally, so this is a defensive minimum)."""
        vectors = [self._multi_xls_concatenated_column(self.multi_xls_common_columns[i])
                   for i in column_indices]
        if not vectors:
            return np.zeros((0, 0))
        min_len = min(len(v) for v in vectors)
        return np.column_stack([v[:min_len] for v in vectors])

    def _dendo_populate_columns(self):
        """Fill the Dendrogram tab column listbox with the same 'Column N'
        names shown in the Multiple Files tab's Data (Columns) list."""
        if self.dendo_column_listbox is None or not self.multi_xls_common_columns:
            return
        self.dendo_column_listbox.delete(0, tk.END)
        for i in range(self.multi_xls_column_listbox.size()):
            self.dendo_column_listbox.insert(tk.END, self.multi_xls_column_listbox.get(i))

    def _dendo_add_to_selection(self):
        """Add all highlighted columns to the dendrogram selection list, sorted."""
        sel = self.dendo_column_listbox.curselection()
        if not sel:
            return
        changed = False
        for idx in sel:
            if idx not in self.dendo_selection_indices:
                self.dendo_selection_indices.append(idx)
                changed = True
        if changed:
            self.dendo_selection_indices.sort()
            self.dendo_selection_listbox.delete(0, tk.END)
            for idx in self.dendo_selection_indices:
                self.dendo_selection_listbox.insert(tk.END, self.dendo_column_listbox.get(idx))
            self._dendo_update_plot()

    def _dendo_remove_from_selection(self):
        """Remove highlighted entry from the dendrogram selection list."""
        sel = self.dendo_selection_listbox.curselection()
        if not sel:
            return
        list_idx = sel[0]
        self.dendo_selection_listbox.delete(list_idx)
        self.dendo_selection_indices.pop(list_idx)
        self._dendo_update_plot()

    def _dendo_on_column_click(self, event):
        """Set dendo_current_column from the exact row under the mouse, then redraw."""
        self._dendo_mouse_click = False
        idx = self.dendo_column_listbox.nearest(event.y)
        if idx < 0 or not self.multi_xls_common_columns:
            return
        self.dendo_current_column = idx
        self._dendo_draw_signal(idx)

    def _dendo_show_signal(self, event=None):
        """Draw the clicked column's signal into the top frame (keyboard nav only)."""
        if self._dendo_mouse_click:
            return
        if not self.multi_xls_common_columns or self.dendo_top_frame is None:
            return
        sel = self.dendo_column_listbox.curselection()
        if not sel:
            return
        col_idx = sel[-1]
        self.dendo_current_column = col_idx
        self._dendo_draw_signal(col_idx)

    def _dendo_draw_signal(self, col_idx):
        """Render the (concatenated, cross-sheet) signal for col_idx into
        dendo_top_frame, one segment per loaded sheet like the Multiple
        Files tab's own plot."""
        if not self.multi_xls_common_columns or self.dendo_top_frame is None:
            return

        if self.dendo_signal_fig is not None:
            plt.close(self.dendo_signal_fig)
            self.dendo_signal_fig = None
        for w in list(self.dendo_top_frame.winfo_children()):
            w.destroy()

        col_label = self.dendo_column_listbox.get(col_idx)
        col_name = self.multi_xls_common_columns[col_idx]
        self.dendo_signal_fig, ax = plt.subplots()
        offset = 0
        for _, values in self._compute_multi_xls_series(col_name):
            n = len(values)
            ax.plot(np.arange(offset, offset + n), values, linewidth=0.8)
            if offset > 0:
                ax.axvline(offset, color=_C['sub'], linestyle='--', linewidth=1, alpha=0.6)
            offset += n
        ax.set_title(col_label)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        self.dendo_signal_fig.tight_layout()
        c = FigureCanvasTkAgg(self.dendo_signal_fig, master=self.dendo_top_frame)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _dendo_update_plot(self):
        """Render the dendrogram in the bottom frame once selection has 2+ columns."""
        if not self.multi_xls_common_columns or self.dendo_bottom_frame is None:
            return

        if self.dendo_fig is not None:
            plt.close(self.dendo_fig)
            self.dendo_fig = None
        for w in list(self.dendo_bottom_frame.winfo_children()):
            w.destroy()

        if len(self.dendo_selection_indices) < 2:
            tk.Label(
                self.dendo_bottom_frame,
                text="Add 2+ columns to Selection to see the dendrogram",
                font=('Arial', 14), bg=_C['panel'], fg=_C['sub']
            ).pack(fill=tk.BOTH, expand=True)
            return

        from corr_dendo_functions import AgglomerativeClustering, _plot_dendrogram_helper

        plot_data = self._dendo_build_matrix(self.dendo_selection_indices)
        clustering = AgglomerativeClustering(
            distance_threshold=0, n_clusters=None
        ).fit(plot_data.T)
        self.dendo_fig, ax = plt.subplots()
        plt.sca(ax)
        _plot_dendrogram_helper(
            clustering, truncate_mode="none", count_sort='none', show_contracted='true'
        )
        ax.set_title(f'Dendrogram ({len(self.dendo_selection_indices)} signals)')
        self.dendo_fig.tight_layout()

        canvas = FigureCanvasTkAgg(self.dendo_fig, master=self.dendo_bottom_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)



    def _dendo_save_image(self):
        """Save the current dendrogram figure to a file."""
        if self.dendo_fig is None:
            messagebox.showwarning("No Plot", "Generate a dendrogram first.")
            return
        from tkinter.filedialog import asksaveasfilename
        filename = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"),
                       ("TIFF Image", "*.tiff"), ("SVG Vector", "*.svg"),
                       ("EPS Vector", "*.eps"), ("All Files", "*.*")],
            title="Save Dendrogram Image"
        )
        if filename:
            self.dendo_fig.savefig(filename, dpi=300, bbox_inches='tight')

    def _dendo_save_csv(self):
        """Save dendrogram clustering data (labels + linkage matrix) to a CSV or Excel file."""
        if not self.multi_xls_common_columns:
            return
        from corr_dendo_functions import AgglomerativeClustering
        from tkinter.filedialog import asksaveasfilename
        import pandas as pd

        if len(self.dendo_selection_indices) >= 2:
            plot_data = self._dendo_build_matrix(self.dendo_selection_indices)
        else:
            plot_data = self._dendo_build_matrix(range(len(self.multi_xls_common_columns)))

        filename = asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All Files", "*.*")],
            title="Save Dendrogram CSV"
        )
        if not filename:
            return

        clustering = AgglomerativeClustering(
            distance_threshold=0, n_clusters=None
        ).fit(plot_data.T)

        df_labels = pd.DataFrame({
            'Sample_Index': list(range(len(clustering.labels_))),
            'Cluster_Label': clustering.labels_
        })
        linkage_rows = [
            {'Merge_Step': i, 'Child_1': int(c1), 'Child_2': int(c2),
             'Distance': clustering.distances_[i]}
            for i, (c1, c2) in enumerate(clustering.children_)
        ]
        df_linkage = pd.DataFrame(linkage_rows)

        if filename.lower().endswith(('.xlsx', '.xls')):
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                df_labels.to_excel(writer, sheet_name='Cluster_Labels', index=False)
                df_linkage.to_excel(writer, sheet_name='Linkage_Matrix', index=False)
        else:
            with open(filename, 'w', newline='') as f:
                f.write("# Cluster Labels\n")
                df_labels.to_csv(f, index=False)
                f.write("\n# Linkage Matrix\n")
                df_linkage.to_csv(f, index=False)

        messagebox.showinfo("Saved", f"Dendrogram data saved to:\n{filename}")

    def load_correlation_matrix_wrapper(self):
        """Open a precomputed correlation matrix file into its own window
        (a dendrogram + heatmap of the loaded matrix), independent of
        whichever tab is currently selected."""
        win = tk.Toplevel(self.root)
        win.title("Loaded Correlation Matrix")
        win.geometry("700x900")
        frame = tk.Frame(win, bg=_C['panel'])
        frame.pack(fill=tk.BOTH, expand=True)
        load_correlation_matrix(self.root, None, target_frame=frame)

    # ==================== MULTIPLE FILES (XLS) TAB ====================

    def open_multiple_xls_files(self):
        """Open multiple .xls/.xlsx/.csv files, let the user choose which
        sheets to load (a .csv counts as its own single sheet), and show
        their data columns in the 'Multiple Files' tab."""
        selection = pick_files_and_sheets(self.root)
        if not selection:
            return
        self._load_xls_with_progress(selection, self._on_multi_xls_load_complete)

    def _on_multi_xls_load_complete(self, datasets):
        """Continue where open_multiple_xls_files left off, once the
        background loading has finished (see _run_with_progress_window)."""
        if not datasets:
            messagebox.showwarning("No Data", "Could not load any of the selected sheets.")
            return

        self.multi_xls_datasets = datasets

        if self.multi_xls_tab is None or not self.multi_xls_tab.winfo_exists():
            self.multi_xls_tab = tk.Frame(self.notebook, bg=_C['bg'])
            self.notebook.add(self.multi_xls_tab, text="  Multiple Files  ")
            self._create_multi_xls_layout()

        self._populate_multi_xls_columns()
        self.notebook.select(self.multi_xls_tab)

        self.menu_visual.entryconfig(
            "Dendrogram", command=self._run_dendogram_on_selection, state=NORMAL)
        if self.dendo_tab is not None and self.dendo_tab.winfo_exists():
            self._dendo_populate_columns()

    def _run_with_progress_window(self, title, message, maximum, worker_fn,
                                   on_complete, on_error=None):
        """Run 'worker_fn(report_progress, report_error)' in a separate
        thread, showing a progress window that stays responsive (it can be
        moved, repainted, etc.) in the meantime - instead of blocking Tk's
        main thread for the whole operation, which is what made the window
        get reported as "not responding" with large files.

        'worker_fn' should call report_progress(done, total, text) to
        update the bar (optional), and return its result, which is passed
        to on_complete(result) on the main thread when it finishes.
        report_error(text) shows an error messagebox without stopping the
        thread (for per-file errors that shouldn't abort the whole
        process). An uncaught exception inside worker_fn is passed to
        on_error(exception) (or shown with a generic messagebox if
        on_error is None), also on the main thread.

        Tkinter is not safe to use from any thread other than the main
        one: worker_fn must never touch Tk widgets or variables directly,
        only call report_progress/report_error (which just queue a
        message) and work with plain Python/pandas data."""
        progress_win = tk.Toplevel(self.root)
        progress_win.title(title)
        progress_win.transient(self.root)
        progress_win.grab_set()
        progress_win.resizable(False, False)
        progress_win.protocol("WM_DELETE_WINDOW", lambda: None)

        tk.Label(progress_win, text=message, font=('Arial', 11, 'bold')).pack(
            pady=(15, 4), padx=15, anchor='w')
        status_label = tk.Label(progress_win, text="Preparing...", font=('Arial', 9),
                                 fg=_C['sub'], anchor='w')
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

    def _load_xls_with_progress(self, selection, on_complete):
        """Load the selected sheets in a separate thread (see
        _run_with_progress_window), showing a progress bar that stays
        responsive while large files are being read. Calls
        on_complete(datasets) on the main thread when it finishes."""
        def worker(report_progress, report_error):
            def _on_progress(done, total, filepath, sheet):
                report_progress(done, total,
                                 f"{os.path.basename(filepath)} — {sheet}  ({done}/{total})")

            def _on_error(filepath, sheet, exc):
                report_error(
                    f"Could not load sheet '{sheet}' from '{os.path.basename(filepath)}':\n{exc}")

            return load_selected_sheets(
                selection, progress_callback=_on_progress, error_callback=_on_error)

        self._run_with_progress_window(
            title="Loading Files", message="Loading Excel sheets...",
            maximum=len(selection), worker_fn=worker, on_complete=on_complete)

    def _create_multi_xls_layout(self):
        """Build the 'Multiple Files' tab: a local menu ('View', 'Plot',
        'Data') with the controls that used to be buttons/checkboxes in the
        sidebar, a resizable column list on the left (by dragging the
        divider) and the combined plot on the right."""
        Grid.rowconfigure(self.multi_xls_tab, 0, weight=0)
        Grid.rowconfigure(self.multi_xls_tab, 1, weight=1)
        Grid.columnconfigure(self.multi_xls_tab, 0, weight=1)

        # Local menu bar, docked at the top of the tab (same look as the
        # window's top 'File' menu), grouping the controls that used to
        # be individual checkboxes/buttons stacked in the sidebar.
        menubar = tk.Frame(self.multi_xls_tab, bg='#f5f5f5', height=26,
                           highlightbackground=_C['border'], highlightthickness=1)
        menubar.grid(row=0, column=0, sticky='ew')
        menubar.grid_propagate(False)

        def add_tab_menu(label, build):
            mb = tk.Menubutton(menubar, text=label, font=('Segoe UI', 9),
                                bg='#f5f5f5', activebackground='#dbeafe',
                                relief='flat', bd=0, padx=8, pady=3)
            menu = tk.Menu(mb, tearoff=0, font=('Segoe UI', 9))
            build(menu)
            mb['menu'] = menu
            mb.pack(side='left')
            return menu

        def build_vista_menu(m):
            m.add_checkbutton(label="Show Middle Plot (Heatmap)",
                               variable=self.multi_xls_show_heatmap_var,
                               command=self._on_multi_xls_show_heatmap_toggle)
            m.add_checkbutton(label="Show Bottom Plot (Correlation)",
                               variable=self.multi_xls_show_correlation_var,
                               command=self._on_multi_xls_show_correlation_toggle)
            m.add_separator()
            m.add_checkbutton(label="Show Data Names",
                               variable=self.multi_xls_show_labels_var,
                               command=self._on_multi_xls_show_labels_toggle)
            m.add_checkbutton(label="Smoothing",
                               variable=self.multi_xls_smoothing_var,
                               command=self._on_multi_xls_smoothing_toggle)
            m.add_command(label="Smoothing Points...",
                          command=self._open_multi_xls_smoothing_points_dialog)
            m.add_checkbutton(label="Show smoothing points",
                               variable=self.multi_xls_show_smoothing_points_var,
                               command=self._on_multi_xls_smoothing_toggle)
            m.add_separator()
            m.add_checkbutton(label="Show Labels (Correlation)",
                               variable=self.multi_xls_show_corr_labels_var,
                               command=self._update_multi_xls_correlation_display)
            m.add_separator()
            m.add_checkbutton(label="Shared Color Scale (heatmap)",
                               variable=self.multi_xls_shared_scale_var,
                               command=self._on_multi_xls_shared_scale_toggle)
            m.add_separator()
            norm_menu = tk.Menu(m, tearoff=0, font=('Segoe UI', 9))
            norm_menu.add_radiobutton(
                label="By Column (minimum of that column in each sheet)",
                variable=self.multi_xls_norm_mode_var, value='local',
                command=self._on_multi_xls_norm_mode_toggle)
            norm_menu.add_radiobutton(
                label="By Sheet (minimum of the entire sheet)",
                variable=self.multi_xls_norm_mode_var, value='sheet',
                command=self._on_multi_xls_norm_mode_toggle)
            norm_menu.add_radiobutton(
                label="By Column Across All Sheets (shared minimum)",
                variable=self.multi_xls_norm_mode_var, value='column_global',
                command=self._on_multi_xls_norm_mode_toggle)
            norm_menu.add_radiobutton(
                label="Global (minimum of all columns and sheets)",
                variable=self.multi_xls_norm_mode_var, value='all_global',
                command=self._on_multi_xls_norm_mode_toggle)
            m.add_cascade(label="Normalization", menu=norm_menu)

        def build_grafica_menu(m):
            m.add_command(label="Axis Limits (Top Plot)...",
                          command=self._open_multi_xls_axis_limits_dialog)
            m.add_command(label="Color Limits (Heatmap)...",
                          command=self._open_multi_xls_heatmap_range_dialog)
            m.add_separator()
            m.add_command(label="Save Plot Image...", state='disabled',
                          command=self._save_multi_xls_plot_image)
            m.add_command(label="Save Heatmap Image...", state='disabled',
                          command=self._save_multi_xls_heatmap_image)
            m.add_command(label="Save Correlation Image...", state='disabled',
                          command=self._save_multi_xls_correlation_image)
            m.add_separator()
            m.add_command(label="Save Smoothed Data (XLSX/CSV)...", state='disabled',
                          command=self._save_multi_xls_smoothed_data)
            m.add_command(label="Save Correlation Data...", state='disabled',
                          command=self._save_multi_xls_correlation_data)

        def build_datos_menu(m):
            m.add_command(label="Edit Classifications...",
                          command=self._open_multi_xls_class_editor)
            m.add_separator()
            m.add_command(label="Save Classifications...", state='disabled',
                          command=self._save_multi_xls_classifications)
            m.add_command(label="Load Classifications...", state='disabled',
                          command=self._load_multi_xls_classifications)
            m.add_separator()
            m.add_command(label="Save Peaks CSV...", state='disabled',
                          command=self._save_multi_xls_peaks_csv)

        add_tab_menu("View", build_vista_menu)
        self.multi_xls_menu_grafica = add_tab_menu("Plot", build_grafica_menu)
        self.multi_xls_menu_datos = add_tab_menu("Data", build_datos_menu)

        self.multi_xls_smoothing_points_var.trace_add(
            'write', lambda *args: self._on_multi_xls_smoothing_toggle())

        # Sidebar (column list) and plot area, in a draggable-sash
        # PanedWindow so the list can be resized by dragging the divider
        # instead of being stuck at a fixed width.
        paned = ttk.PanedWindow(self.multi_xls_tab, orient='horizontal')
        paned.grid(row=1, column=0, sticky='nsew', padx=8, pady=8)

        sidebar = tk.Frame(paned, bg=_C['panel'],
                           highlightbackground=_C['border'], highlightthickness=1)
        sidebar.columnconfigure(0, weight=1)

        def _sec(text, row):
            tk.Label(sidebar, text=text, font=('Arial', 8, 'bold'),
                     bg=_C['panel'], fg=_C['sub']).grid(row=row, column=0, sticky='w',
                                                         padx=12, pady=(12, 2))
            row += 1
            tk.Frame(sidebar, bg=_C['border'], height=1).grid(row=row, column=0,
                                                               sticky='ew', padx=10)
            return row + 1

        row = _sec("DATA (COLUMNS)", 0)

        lb_frame = tk.Frame(sidebar, bg=_C['card'],
                            highlightbackground=_C['border'], highlightthickness=1)
        lb_frame.grid(row=row, column=0, sticky='nsew', padx=10, pady=(6, 10))
        sidebar.rowconfigure(row, weight=2)
        lb_frame.rowconfigure(0, weight=1)
        lb_frame.columnconfigure(0, weight=1)
        row += 1

        scrollbar = tk.Scrollbar(lb_frame, relief='flat', width=10)
        scrollbar.grid(row=0, column=1, sticky='ns')

        self.multi_xls_column_listbox = tk.Listbox(
            lb_frame, yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE, font=('Arial', 10),
            bg=_C['card'], fg=_C['text'],
            selectbackground=_C['acc'], selectforeground='white',
            relief='flat', bd=0, highlightthickness=0, activestyle='none'
        )
        self.multi_xls_column_listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar.config(command=self.multi_xls_column_listbox.yview)
        self.multi_xls_column_listbox.bind('<<ListboxSelect>>', self._on_multi_xls_column_select)

        # ── Peak Finder ──
        row = _sec("PEAK FINDER", row)

        self.multi_xls_peak_method_combo = ttk.Combobox(
            sidebar, textvariable=self.multi_xls_peak_method_var,
            values=['None', 'Elliptic Envelope', 'Peak Caller', 'Local Outlier Factor',
                    'Peak Function 4', 'Isolation Forest', 'Linear Model', 'Peak Function 7'],
            state='readonly', width=22
        )
        self.multi_xls_peak_method_combo.grid(row=row, column=0, padx=10, pady=(2, 10), sticky='ew')
        self.multi_xls_peak_method_combo.bind(
            '<<ComboboxSelected>>', lambda e: self._on_multi_xls_peak_method_change(show_dialog=True))
        self.multi_xls_peak_method_combo.config(state=DISABLED)
        row += 1

        # ── Correlation ──
        row = _sec("CORRELATION", row)

        multi_xls_corr_method_combo = ttk.Combobox(
            sidebar, textvariable=self.multi_xls_corr_method_var,
            values=['pearson', 'kendall', 'spearman'], state='readonly', width=15
        )
        multi_xls_corr_method_combo.grid(row=row, column=0, padx=10, pady=(2, 10), sticky='w')
        multi_xls_corr_method_combo.bind(
            '<<ComboboxSelected>>', lambda e: self._update_multi_xls_correlation_display())
        row += 1

        # ── Selection ──
        row = _sec("SELECTION", row)

        sel_lb_frame = tk.Frame(sidebar, bg=_C['card'],
                                highlightbackground=_C['border'], highlightthickness=1)
        sel_lb_frame.grid(row=row, column=0, sticky='nsew', padx=10, pady=(6, 4))
        sidebar.rowconfigure(row, weight=1)
        sel_lb_frame.rowconfigure(0, weight=1)
        sel_lb_frame.columnconfigure(0, weight=1)
        row += 1

        sel_scrollbar = tk.Scrollbar(sel_lb_frame, relief='flat', width=10)
        sel_scrollbar.grid(row=0, column=1, sticky='ns')

        self.multi_xls_selection_listbox = tk.Listbox(
            sel_lb_frame, yscrollcommand=sel_scrollbar.set,
            selectmode=tk.SINGLE, font=('Arial', 10),
            bg=_C['card'], fg=_C['text'],
            selectbackground=_C['acc'], selectforeground='white',
            relief='flat', bd=0, highlightthickness=0, activestyle='none'
        )
        self.multi_xls_selection_listbox.grid(row=0, column=0, sticky='nsew')
        sel_scrollbar.config(command=self.multi_xls_selection_listbox.yview)

        self.btn_multi_xls_add_sel = ctk.CTkButton(
            sidebar, text="Add to Selection", height=28, corner_radius=6,
            fg_color=_C['acc'], hover_color=_C['acc2'], text_color='white',
            font=ctk.CTkFont(size=11), state='disabled',
            command=self._add_to_multi_xls_selection
        )
        self.btn_multi_xls_add_sel.grid(row=row, column=0, sticky='ew', padx=10, pady=(2, 2))
        row += 1

        self.btn_multi_xls_remove_sel = ctk.CTkButton(
            sidebar, text="Remove from Selection", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            state='disabled', command=self._remove_from_multi_xls_selection
        )
        self.btn_multi_xls_remove_sel.grid(row=row, column=0, sticky='ew', padx=10, pady=(0, 10))
        row += 1

        # Right side - stacked plot areas (line plot on top, heatmap below),
        # in their own vertical PanedWindow so that split can be resized by
        # dragging too, instead of being stuck at a fixed 3:2 ratio. Both
        # panels are redrawn to exactly fill their frame on every resize, so
        # the whole graph and every sheet image are always visible without
        # needing to scroll.
        right_paned = ttk.PanedWindow(paned, orient='vertical')

        paned.add(sidebar, weight=0)
        paned.add(right_paned, weight=1)
        # Give the sidebar a sensible starting width (same as the old fixed
        # width); the user can drag the sash to resize it from here.
        self.root.after(50, lambda: paned.sashpos(0, 250))

        # ttk.PanedWindow has no built-in minsize per pane, so enforce one by
        # snapping the sash back whenever the sidebar is dragged narrower
        # than what "Column 999" needs to display without truncating.
        sidebar_min_w = tkfont.Font(family='Arial', size=10).measure('Column 999') + 60

        def _enforce_sidebar_min_width(event=None):
            w = sidebar.winfo_width()
            if 0 < w < sidebar_min_w:
                paned.sashpos(0, sidebar_min_w)

        sidebar.bind('<Configure>', _enforce_sidebar_min_width)

        # While a sash is actively being dragged, suppress the debounced
        # resize-redraws entirely (see _on_multi_xls_plot_frame_resize /
        # _on_multi_xls_heatmap_frame_resize) - they're expensive (reprocess
        # every sheet's data from scratch), and a plain human drag has
        # natural pauses long enough for the debounce timer to fire mid-drag
        # anyway, which reads as stutter. Redraw once, immediately, on
        # release instead.
        def _on_sash_press(event=None):
            self._multi_xls_sash_dragging = True
            # Also cancel any job already scheduled from a resize that
            # happened just before the drag started (e.g. the initial
            # layout) - otherwise it could still fire mid-drag, since only
            # *new* scheduling is blocked while dragging.
            if self._multi_xls_plot_resize_job is not None:
                self.root.after_cancel(self._multi_xls_plot_resize_job)
                self._multi_xls_plot_resize_job = None
            if self._multi_xls_heatmap_resize_job is not None:
                self.root.after_cancel(self._multi_xls_heatmap_resize_job)
                self._multi_xls_heatmap_resize_job = None

        paned.bind('<ButtonPress-1>', _on_sash_press)
        right_paned.bind('<ButtonPress-1>', _on_sash_press)

        # Redraw both plots immediately when the user releases either sash,
        # instead of waiting for the debounce timer used for window-border
        # resizes (which can't be tied to a mouse-release event, since that
        # drag is owned by the OS window manager, not Tk).
        paned.bind('<ButtonRelease-1>', self._on_multi_xls_sash_release)
        right_paned.bind('<ButtonRelease-1>', self._on_multi_xls_sash_release)

        self.multi_xls_plot_frame = tk.Frame(right_paned, bg=_C['panel'],
                                              highlightbackground=_C['border'],
                                              highlightthickness=1)
        self.multi_xls_plot_frame.bind('<Configure>', self._on_multi_xls_plot_frame_resize)

        # Row of per-sheet classification dropdowns, pixel-aligned above the
        # segment each sheet occupies in the plot below (cosmetic only).
        self.multi_xls_class_row = tk.Frame(self.multi_xls_plot_frame, bg=_C['panel'], height=30)
        self.multi_xls_class_row.pack(side=tk.TOP, fill=tk.X)
        self.multi_xls_class_row.pack_propagate(False)

        # Small "next column" button docked to the top-right corner, above
        # the classification row. Created after (and kept raised over) that
        # row so it stays a fixed, stable control regardless of how many
        # per-sheet dropdowns get rebuilt into the row below it.
        self.btn_multi_xls_next_column = tk.Button(
            self.multi_xls_plot_frame, text="▶", font=('Arial', 10, 'bold'),
            command=self._on_multi_xls_next_column, state=DISABLED
        )
        self.btn_multi_xls_next_column.place(relx=1.0, rely=0, anchor='ne', width=26, height=26)
        self.btn_multi_xls_next_column.lift()

        self.multi_xls_plot_placeholder = tk.Label(
            self.multi_xls_plot_frame, text="Select a column to plot",
            font=('Arial', 14), bg=_C['panel'], fg=_C['sub'])
        self.multi_xls_plot_placeholder.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Bottom: heatmap images (one per sheet, columns=data, rows=samples)
        self.multi_xls_heatmap_frame = tk.Frame(right_paned, bg=_C['panel'],
                                                 highlightbackground=_C['border'],
                                                 highlightthickness=1)
        self.multi_xls_heatmap_frame.bind('<Configure>', self._on_multi_xls_heatmap_frame_resize)

        self.multi_xls_heatmap_placeholder = tk.Label(
            self.multi_xls_heatmap_frame, text="Sheet images will appear here",
            font=('Arial', 14), bg=_C['panel'], fg=_C['sub'])
        self.multi_xls_heatmap_placeholder.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Bottom-most pane: Pearson/Kendall/Spearman correlation matrix
        # across the columns added to Selection, same role as Data
        # Visualization's plot_bottom_frame.
        self.multi_xls_correlation_frame = tk.Frame(right_paned, bg=_C['panel'],
                                                     highlightbackground=_C['border'],
                                                     highlightthickness=1)
        self.multi_xls_correlation_placeholder = tk.Label(
            self.multi_xls_correlation_frame,
            text="Add 2+ columns to Selection to see correlation",
            font=('Arial', 14), bg=_C['panel'], fg=_C['sub'])
        self.multi_xls_correlation_placeholder.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.multi_xls_right_paned = right_paned
        # Only the top (line plot) pane is shown by default; the heatmap
        # and correlation panes are added on demand by their View-menu
        # checkboxes (see _on_multi_xls_show_heatmap_toggle /
        # _on_multi_xls_show_correlation_toggle), each unchecked by
        # default so a fresh load shows the upper plot alone.
        right_paned.add(self.multi_xls_plot_frame, weight=3)

        # Keep every visible panel at a legible minimum height, same idea
        # as the sidebar's minimum width above.
        right_min_h = 160

        def _enforce_right_paned_min_height(event=None):
            total = right_paned.winfo_height()
            n_sashes = len(right_paned.panes()) - 1
            if total <= 1 or n_sashes < 1:
                return
            positions = [right_paned.sashpos(i) for i in range(n_sashes)]
            prev = 0
            for i, pos in enumerate(positions):
                if pos - prev < right_min_h:
                    pos = prev + right_min_h
                    right_paned.sashpos(i, pos)
                positions[i] = pos
                prev = pos
            if total - prev < right_min_h and n_sashes >= 1:
                right_paned.sashpos(n_sashes - 1, max(prev, total - right_min_h))

        self._enforce_multi_xls_right_paned_min_height = _enforce_right_paned_min_height
        self.multi_xls_plot_frame.bind('<Configure>', _enforce_right_paned_min_height, add='+')

    def _populate_multi_xls_columns(self):
        """Fill the sidebar list with generic 'Column N' names (one for
        each column common to all loaded sheets) and plot the first one
        by default. These names don't change with the 'Show Data Names'
        checkbox, which only affects the plot's labels."""
        if self.multi_xls_column_listbox is None:
            return
        old_columns = self.multi_xls_common_columns
        old_classifications = self.multi_xls_classifications
        self.multi_xls_common_columns = common_column_names(self.multi_xls_datasets)

        # Remap saved classifications from old column indices to new ones by
        # matching column name, so (re)loading files doesn't discard
        # classification work already done -- only columns that no longer
        # exist are dropped.
        name_to_old_index = {name: i for i, name in enumerate(old_columns)}
        self.multi_xls_classifications = {}
        for new_idx, name in enumerate(self.multi_xls_common_columns):
            old_idx = name_to_old_index.get(name)
            if old_idx is not None and old_idx in old_classifications:
                self.multi_xls_classifications[new_idx] = old_classifications[old_idx]

        self.multi_xls_column_listbox.delete(0, tk.END)
        for i in range(len(self.multi_xls_common_columns)):
            self.multi_xls_column_listbox.insert(tk.END, f"Column {i + 1}")

        self._rebuild_multi_xls_class_row()

        if self.multi_xls_common_columns:
            self.multi_xls_column_listbox.selection_set(0)
            self._draw_multi_xls_plot(0)
        else:
            messagebox.showwarning(
                "No Common Columns",
                "The selected sheets do not share any data column with the same name."
            )

        self._draw_multi_xls_heatmap()

        if self.multi_xls_datasets:
            self.multi_xls_menu_grafica.entryconfigure("Save Plot Image...", state='normal')
            self.multi_xls_menu_grafica.entryconfigure("Save Heatmap Image...", state='normal')
            self.multi_xls_menu_grafica.entryconfigure("Save Smoothed Data (XLSX/CSV)...", state='normal')
            self.multi_xls_menu_grafica.entryconfigure("Save Correlation Image...", state='normal')
            self.btn_multi_xls_next_column.configure(state='normal')
            self.multi_xls_menu_datos.entryconfigure("Save Classifications...", state='normal')
            self.multi_xls_menu_datos.entryconfigure("Load Classifications...", state='normal')
            self.multi_xls_menu_datos.entryconfigure("Save Peaks CSV...", state='normal')
            self.multi_xls_peak_method_combo.config(state='readonly')
            self.btn_multi_xls_add_sel.configure(state='normal')
            self.btn_multi_xls_remove_sel.configure(state='normal')

    def _rebuild_multi_xls_class_row(self):
        """Recreate the classification combobox for each loaded sheet (one
        per sheet). The default options are the names of every loaded
        sheet; each sheet starts out classified with its own name. They
        are positioned on the next redraw of the line plot. The chosen
        classification is saved separately for each data column
        (self.multi_xls_classifications), so switching columns in the
        list updates the comboboxes with whatever was chosen for that
        particular column."""
        if self.multi_xls_class_row is None:
            return

        for w in list(self.multi_xls_class_row.winfo_children()):
            w.destroy()
        self.multi_xls_class_combos = {}

        # Merge in any new sheet names rather than overwriting the list, so
        # custom classification options (renamed via the editor, or loaded
        # from a saved file) survive (re)loading more/different files.
        for label in (ds['label'] for ds in self.multi_xls_datasets):
            if label not in self.multi_xls_classes:
                self.multi_xls_classes.append(label)

        self.multi_xls_sheet_class_var = {
            ds['label']: tk.StringVar(value=ds['label']) for ds in self.multi_xls_datasets
        }
        for label, var in self.multi_xls_sheet_class_var.items():
            var.trace_add('write', lambda *_a, sheet_label=label: self._on_multi_xls_sheet_class_changed(sheet_label))

        for ds in self.multi_xls_datasets:
            label = ds['label']
            combo = ttk.Combobox(
                self.multi_xls_class_row, textvariable=self.multi_xls_sheet_class_var[label],
                values=self.multi_xls_classes, state='readonly', font=('Arial', 8)
            )
            self.multi_xls_class_combos[label] = combo

    def _on_multi_xls_sheet_class_changed(self, sheet_label):
        """Save the chosen classification for 'sheet_label' under the
        currently selected column."""
        if self._multi_xls_syncing_class_row or self.multi_xls_current_index is None:
            return
        value = self.multi_xls_sheet_class_var[sheet_label].get()
        self.multi_xls_classifications.setdefault(self.multi_xls_current_index, {})[sheet_label] = value

    def _sync_multi_xls_class_row_values(self, index):
        """Update the comboboxes to show the saved classification of each
        sheet for column 'index' (or the sheet's name as the default
        value if that column+sheet combination hasn't been classified
        yet)."""
        saved = self.multi_xls_classifications.get(index, {})
        self._multi_xls_syncing_class_row = True
        try:
            for label, var in self.multi_xls_sheet_class_var.items():
                var.set(saved.get(label, label))
        finally:
            self._multi_xls_syncing_class_row = False

    def _position_multi_xls_class_row(self, ax, bounds):
        """Position each classification combobox pixel-aligned with the
        VISIBLE portion (according to the X axis's current limits, which
        may have been set manually via 'Axis Limits') of its sheet's
        (start_offset, end_offset) segment in the just-drawn plot 'ax'.
        Sheets that fall completely outside the visible range are hidden
        instead of being placed off-canvas."""
        if self.multi_xls_class_row is None:
            return
        xlim = ax.get_xlim()
        placed = set()
        for ds, (x0_data, x1_data) in zip(self.multi_xls_datasets, bounds):
            label = ds['label']
            combo = self.multi_xls_class_combos.get(label)
            if combo is None:
                continue

            vis_x0 = max(x0_data, xlim[0])
            vis_x1 = min(x1_data, xlim[1])
            if vis_x1 <= vis_x0:
                continue

            x0_px = ax.transData.transform((vis_x0, 0))[0]
            x1_px = ax.transData.transform((vis_x1, 0))[0]
            width_px = max(x1_px - x0_px, 30)
            combo.place(x=x0_px, y=2, width=width_px - 2, height=26)
            placed.add(label)

        for label, combo in self.multi_xls_class_combos.items():
            if label not in placed:
                combo.place_forget()

    def _open_multi_xls_class_editor(self):
        """Dialog to rename, delete, or add classification options. On
        apply, updates every combobox and preserves (or remaps) the
        classification already chosen for each sheet."""
        if not self.multi_xls_classes:
            messagebox.showinfo("No Data", "Load .xls files first.")
            return

        import uuid as _uuid

        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Classifications")
        dialog.geometry("360x440")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Classification options:", font=('Arial', 11, 'bold')).pack(
            pady=(10, 4), padx=10, anchor='w')

        list_frame = tk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                              selectmode=tk.SINGLE, font=('Arial', 10))
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        original_entries = [{'id': str(_uuid.uuid4()), 'name': c} for c in self.multi_xls_classes]
        entries = [dict(e) for e in original_entries]
        for e in entries:
            listbox.insert(tk.END, e['name'])

        edit_var = tk.StringVar()
        edit_frame = tk.Frame(dialog)
        edit_frame.pack(fill=tk.X, padx=10, pady=(4, 2))
        tk.Entry(edit_frame, textvariable=edit_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        def on_select(event=None):
            sel = listbox.curselection()
            if sel:
                edit_var.set(entries[sel[0]]['name'])
        listbox.bind('<<ListboxSelect>>', on_select)

        def add_option():
            name = edit_var.get().strip()
            if not name:
                return
            if any(e['name'] == name for e in entries):
                messagebox.showwarning("Duplicate", "That option already exists.", parent=dialog)
                return
            entries.append({'id': str(_uuid.uuid4()), 'name': name})
            listbox.insert(tk.END, name)

        def rename_option():
            sel = listbox.curselection()
            if not sel:
                messagebox.showinfo("No Selection", "Select an option to rename.", parent=dialog)
                return
            name = edit_var.get().strip()
            if not name:
                return
            idx = sel[0]
            if any(i != idx and e['name'] == name for i, e in enumerate(entries)):
                messagebox.showwarning("Duplicate", "That option already exists.", parent=dialog)
                return
            entries[idx]['name'] = name
            listbox.delete(idx)
            listbox.insert(idx, name)
            listbox.selection_set(idx)

        def delete_option():
            sel = listbox.curselection()
            if not sel:
                messagebox.showinfo("No Selection", "Select an option to delete.", parent=dialog)
                return
            if len(entries) <= 1:
                messagebox.showwarning("Not Allowed", "At least one option must remain.", parent=dialog)
                return
            idx = sel[0]
            del entries[idx]
            listbox.delete(idx)
            edit_var.set('')

        btn_row = tk.Frame(dialog)
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 6))
        tk.Button(btn_row, text="Add", command=add_option).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Rename", command=rename_option).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(btn_row, text="Delete", command=delete_option).pack(side=tk.LEFT, padx=(6, 0))

        def on_apply():
            self._apply_multi_xls_class_changes(original_entries, entries)
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ok_row = tk.Frame(dialog)
        ok_row.pack(pady=10)
        tk.Button(ok_row, text="Apply", command=on_apply, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(ok_row, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)

    def _apply_multi_xls_class_changes(self, original_entries, final_entries):
        """Apply the classification editor's changes: propagate renames to
        the sheets that had that option chosen, reassign the sheets that
        had a deleted option, and refresh the available options in every
        combobox."""
        original_by_id = {e['id']: e['name'] for e in original_entries}
        final_by_id = {e['id']: e['name'] for e in final_entries}

        rename_map = {}
        deleted_names = set()
        for id_, old_name in original_by_id.items():
            if id_ in final_by_id:
                new_name = final_by_id[id_]
                if new_name != old_name:
                    rename_map[old_name] = new_name
            else:
                deleted_names.add(old_name)

        self.multi_xls_classes = [e['name'] for e in final_entries]
        fallback = self.multi_xls_classes[0] if self.multi_xls_classes else ''

        # Propagate the rename/delete to every column's saved classification,
        # not just the one currently shown in the combobox.
        for col_classifications in self.multi_xls_classifications.values():
            for sheet_label, current in list(col_classifications.items()):
                if current in rename_map:
                    col_classifications[sheet_label] = rename_map[current]
                elif current in deleted_names:
                    col_classifications[sheet_label] = fallback

        for combo in self.multi_xls_class_combos.values():
            combo.configure(values=self.multi_xls_classes)

        if self.multi_xls_current_index is not None:
            self._sync_multi_xls_class_row_values(self.multi_xls_current_index)

    def _open_multi_xls_axis_limits_dialog(self):
        """Dialog to manually set the X/Y axis limits of the 'Multiple
        Files' top plot, or revert to auto-scaling."""
        if self._multi_xls_plot_ax is None:
            messagebox.showinfo("No Data", "Load and plot a column first.")
            return

        cur_xlim = self.multi_xls_xlim or self._multi_xls_plot_ax.get_xlim()
        cur_ylim = self.multi_xls_ylim or self._multi_xls_plot_ax.get_ylim()

        dialog = tk.Toplevel(self.root)
        dialog.title("Axis Limits - Top Plot")
        dialog.geometry("300x230")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        tk.Label(dialog, text="X Axis", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky='w', padx=10, pady=(12, 2))
        tk.Label(dialog, text="Min:").grid(row=1, column=0, sticky='e', padx=(10, 4))
        x_min_var = tk.StringVar(value=f"{cur_xlim[0]:.4g}")
        tk.Entry(dialog, textvariable=x_min_var, width=12).grid(row=1, column=1, sticky='w', padx=(0, 10), pady=2)
        tk.Label(dialog, text="Max:").grid(row=2, column=0, sticky='e', padx=(10, 4))
        x_max_var = tk.StringVar(value=f"{cur_xlim[1]:.4g}")
        tk.Entry(dialog, textvariable=x_max_var, width=12).grid(row=2, column=1, sticky='w', padx=(0, 10), pady=2)

        tk.Label(dialog, text="Y Axis", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, columnspan=2, sticky='w', padx=10, pady=(12, 2))
        tk.Label(dialog, text="Min:").grid(row=4, column=0, sticky='e', padx=(10, 4))
        y_min_var = tk.StringVar(value=f"{cur_ylim[0]:.4g}")
        tk.Entry(dialog, textvariable=y_min_var, width=12).grid(row=4, column=1, sticky='w', padx=(0, 10), pady=2)
        tk.Label(dialog, text="Max:").grid(row=5, column=0, sticky='e', padx=(10, 4))
        y_max_var = tk.StringVar(value=f"{cur_ylim[1]:.4g}")
        tk.Entry(dialog, textvariable=y_max_var, width=12).grid(row=5, column=1, sticky='w', padx=(0, 10), pady=2)

        def apply_limits():
            try:
                xmin, xmax = float(x_min_var.get()), float(x_max_var.get())
                ymin, ymax = float(y_min_var.get()), float(y_max_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "All limits must be numbers.", parent=dialog)
                return
            if xmin >= xmax or ymin >= ymax:
                messagebox.showerror("Invalid Range", "The minimum must be less than the maximum.", parent=dialog)
                return
            self.multi_xls_xlim = (xmin, xmax)
            self.multi_xls_ylim = (ymin, ymax)
            dialog.destroy()
            if self.multi_xls_current_index is not None:
                self._draw_multi_xls_plot(self.multi_xls_current_index)

        def reset_auto():
            self.multi_xls_xlim = None
            self.multi_xls_ylim = None
            dialog.destroy()
            if self.multi_xls_current_index is not None:
                self._draw_multi_xls_plot(self.multi_xls_current_index)

        btns = tk.Frame(dialog)
        btns.grid(row=6, column=0, columnspan=2, pady=18)
        tk.Button(btns, text="Auto", command=reset_auto, width=8).pack(side='left', padx=4)
        tk.Button(btns, text="Cancel", command=dialog.destroy, width=8).pack(side='left', padx=4)
        tk.Button(btns, text="Apply", command=apply_limits, width=8).pack(side='left', padx=4)

    def _open_multi_xls_heatmap_range_dialog(self):
        """Dialog to manually set the color range (min/max) of the
        'Multiple Files' heatmap, showing the range currently in use, or
        revert to auto-scaling (per sheet, or shared if 'Shared Color
        Scale' is on)."""
        if not self.multi_xls_datasets:
            messagebox.showinfo("No Data", "Load files first.")
            return

        matrices, _vmin, _vmax, _per_sheet_ranges = self._compute_multi_xls_heatmap_matrices()
        if not matrices:
            messagebox.showinfo("No Data", "There is no numeric data to compute a range from.")
            return

        # Combined range across all sheets, over the same normalized/smoothed
        # data the heatmap draws - a "current range" reference even when
        # each sheet uses its own scale.
        all_finite = np.concatenate([m[np.isfinite(m)].ravel() for m in matrices])
        auto_min, auto_max = float(all_finite.min()), float(all_finite.max())

        if self.multi_xls_heatmap_manual_range is not None:
            cur_min, cur_max = self.multi_xls_heatmap_manual_range
        else:
            cur_min, cur_max = auto_min, auto_max

        dialog = tk.Toplevel(self.root)
        dialog.title("Color Limits - Heatmap")
        dialog.geometry("320x210")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        tk.Label(dialog, text="Current range in use:", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky='w', padx=10, pady=(12, 0))
        tk.Label(dialog, text=f"{cur_min:.4g}  —  {cur_max:.4g}", font=('Arial', 9)).grid(
            row=1, column=0, columnspan=2, sticky='w', padx=10, pady=(0, 10))

        tk.Label(dialog, text="Min:").grid(row=2, column=0, sticky='e', padx=(10, 4))
        min_var = tk.StringVar(value=f"{cur_min:.4g}")
        tk.Entry(dialog, textvariable=min_var, width=12).grid(row=2, column=1, sticky='w', padx=(0, 10), pady=2)
        tk.Label(dialog, text="Max:").grid(row=3, column=0, sticky='e', padx=(10, 4))
        max_var = tk.StringVar(value=f"{cur_max:.4g}")
        tk.Entry(dialog, textvariable=max_var, width=12).grid(row=3, column=1, sticky='w', padx=(0, 10), pady=2)

        def apply_range():
            try:
                new_min, new_max = float(min_var.get()), float(max_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Min and Max must be numbers.", parent=dialog)
                return
            if new_min >= new_max:
                messagebox.showerror("Invalid Range", "The minimum must be less than the maximum.", parent=dialog)
                return
            self.multi_xls_heatmap_manual_range = (new_min, new_max)
            dialog.destroy()
            self._draw_multi_xls_heatmap()
            if self.multi_xls_current_index is not None:
                self._draw_multi_xls_plot(self.multi_xls_current_index)

        def reset_auto():
            self.multi_xls_heatmap_manual_range = None
            dialog.destroy()
            self._draw_multi_xls_heatmap()
            if self.multi_xls_current_index is not None:
                self._draw_multi_xls_plot(self.multi_xls_current_index)

        btns = tk.Frame(dialog)
        btns.grid(row=4, column=0, columnspan=2, pady=18)
        tk.Button(btns, text="Auto", command=reset_auto, width=8).pack(side='left', padx=4)
        tk.Button(btns, text="Cancel", command=dialog.destroy, width=8).pack(side='left', padx=4)
        tk.Button(btns, text="Apply", command=apply_range, width=8).pack(side='left', padx=4)

    def _on_multi_xls_show_heatmap_toggle(self):
        """Add or remove the middle (heatmap) pane from the vertical
        PanedWindow, inserting it before the correlation pane if that one
        is already shown so the top-to-bottom order (plot, heatmap,
        correlation) stays fixed regardless of toggle order."""
        paned = self.multi_xls_right_paned
        if paned is None:
            return
        panes = paned.panes()
        shown = str(self.multi_xls_heatmap_frame) in panes
        if self.multi_xls_show_heatmap_var.get() and not shown:
            if str(self.multi_xls_correlation_frame) in panes:
                paned.insert(self.multi_xls_correlation_frame, self.multi_xls_heatmap_frame, weight=2)
            else:
                paned.add(self.multi_xls_heatmap_frame, weight=2)
            self._enforce_multi_xls_right_paned_min_height()
            self._draw_multi_xls_heatmap()
        elif not self.multi_xls_show_heatmap_var.get() and shown:
            paned.forget(self.multi_xls_heatmap_frame)

    def _on_multi_xls_show_correlation_toggle(self):
        """Add or remove the bottom (correlation) pane from the vertical
        PanedWindow - always goes at the end, so plain .add() is enough
        regardless of whether the heatmap pane is shown."""
        paned = self.multi_xls_right_paned
        if paned is None:
            return
        shown = str(self.multi_xls_correlation_frame) in paned.panes()
        if self.multi_xls_show_correlation_var.get() and not shown:
            paned.add(self.multi_xls_correlation_frame, weight=2)
            self._enforce_multi_xls_right_paned_min_height()
            self._update_multi_xls_correlation_display()
        elif not self.multi_xls_show_correlation_var.get() and shown:
            paned.forget(self.multi_xls_correlation_frame)

    def _on_multi_xls_show_labels_toggle(self):
        """Redraw both plots to show or hide each sheet's label under the
        X axis, without touching the data list's names."""
        if self.multi_xls_current_index is not None:
            self._draw_multi_xls_plot(self.multi_xls_current_index)
        self._draw_multi_xls_heatmap()

    def _on_multi_xls_smoothing_toggle(self):
        """Redraw both plots (line and heatmap) with Convex Envelope
        smoothing applied (or removed), or with the updated number of
        points."""
        if self.multi_xls_current_index is not None:
            self._draw_multi_xls_plot(self.multi_xls_current_index)
        self._draw_multi_xls_heatmap()

    def _on_multi_xls_peak_method_change(self, show_dialog=False):
        """Show the parameter dialog for the newly chosen Peak Finder
        method (reusing the same _PEAK_PARAM_SPECS/show_parameter_dialog
        Data Visualization used), cache the params, and redraw the top
        plot with peaks overlaid on every sheet's segment."""
        method = self.multi_xls_peak_method_var.get()
        if method != 'None':
            if show_dialog or method not in self.multi_xls_peak_method_params:
                from peak_functions import show_parameter_dialog
                spec = self._PEAK_PARAM_SPECS.get(method)
                if spec:
                    title, param_list = spec
                    new_params = show_parameter_dialog(self.root, title, param_list)
                    if new_params is None:
                        self.multi_xls_peak_method_var.set('None')
                    else:
                        self.multi_xls_peak_method_params[method] = new_params
        if self.multi_xls_current_index is not None:
            self._draw_multi_xls_plot(self.multi_xls_current_index)

    def _open_multi_xls_smoothing_points_dialog(self):
        """Dialog to adjust the number of points used by the Convex
        Envelope smoothing of the 'Multiple Files' top plot."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Smoothing Points")
        dialog.geometry("260x120")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        row = tk.Frame(dialog)
        row.pack(padx=16, pady=(20, 8), fill='x')
        tk.Label(row, text="Number of points:").pack(side='left')
        points_var = tk.StringVar(value=str(self.multi_xls_smoothing_points_var.get()))
        ttk.Spinbox(row, from_=2, to=50, textvariable=points_var, width=6).pack(side='left', padx=(8, 0))

        def apply_points():
            try:
                n = int(points_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Must be an integer.", parent=dialog)
                return
            if not (2 <= n <= 50):
                messagebox.showerror("Invalid Value", "Must be between 2 and 50.", parent=dialog)
                return
            self.multi_xls_smoothing_points_var.set(n)
            dialog.destroy()

        btns = tk.Frame(dialog)
        btns.pack(side='bottom', pady=16)
        tk.Button(btns, text="Cancel", command=dialog.destroy, width=8).pack(side='left', padx=4)
        tk.Button(btns, text="Apply", command=apply_points, width=8).pack(side='left', padx=4)

    def _on_multi_xls_shared_scale_toggle(self):
        """Redraw the heatmap with a color scale shared across all sheets,
        or an independent scale for each sheet. Also redraws the line
        plot: its right margin depends on whether the heatmap will show a
        colorbar, so both stay aligned."""
        self._draw_multi_xls_heatmap()
        if self.multi_xls_current_index is not None:
            self._draw_multi_xls_plot(self.multi_xls_current_index)

    def _multi_xls_effective_shared_scale(self):
        """True when every sheet's heatmap image is drawn with the same
        color range - either because 'Shared Color Scale' is on, or
        because a manual range override is set (Color Limits). Used to
        decide whether to reserve/draw the heatmap colorbar and to keep
        the line plot's right margin aligned with it."""
        return self.multi_xls_shared_scale_var.get() or self.multi_xls_heatmap_manual_range is not None

    def _smooth_multi_xls_signal(self, values):
        """Apply Convex Envelope smoothing to a single sheet's column values
        when the 'Smoothing' checkbox is on."""
        if not self.multi_xls_smoothing_var.get():
            return values
        try:
            n_points = self.multi_xls_smoothing_points_var.get()
        except tk.TclError:
            return values
        from peak_functions import _convex_envelope_detrend_signal
        return _convex_envelope_detrend_signal(values, n_points=n_points)

    def _smooth_multi_xls_matrix(self, m):
        """Apply _smooth_multi_xls_signal to every data column of a
        normalized sheet matrix (samples x columns), for the heatmap - so
        it matches the top line plot when 'Smoothing' is on. NaNs (missing
        samples) are interpolated before smoothing, since the Convex
        Envelope algorithm needs a finite series, then restored afterward."""
        if not self.multi_xls_smoothing_var.get():
            return m
        import pandas as pd
        out = np.array(m, dtype=float, copy=True)
        for c in range(m.shape[1]):
            col = m[:, c]
            nan_mask = ~np.isfinite(col)
            if nan_mask.all():
                continue
            filled = pd.Series(col).interpolate(limit_direction='both').to_numpy() if nan_mask.any() else col
            smoothed = self._smooth_multi_xls_signal(filled)
            smoothed[nan_mask] = np.nan
            out[:, c] = smoothed
        return out

    def _on_multi_xls_norm_mode_toggle(self):
        """Redraw both plots with the chosen normalization mode: by column
        (local, default), by sheet, by column across all sheets, or
        global (all columns and sheets). Both plots use the same mode, so
        both are redrawn."""
        if self.multi_xls_current_index is not None:
            self._draw_multi_xls_plot(self.multi_xls_current_index)
        self._draw_multi_xls_heatmap()

    def _compute_multi_xls_series(self, col_name):
        """For each loaded sheet that has column 'col_name', return
        (label, values) already interpolated, normalized and smoothed -
        the same processing that draws the top plot, factored out so the
        cache can reuse it. The normalization divisor depends on
        multi_xls_norm_mode_var:
        - 'local': minimum of that column, within each sheet (each sheet
          uses its own minimum).
        - 'sheet': minimum of the entire sheet (all its columns), within
          each sheet (same criterion as the heatmap).
        - 'column_global': minimum of that column, pooling every loaded
          sheet - a single value shared by all of them.
        - 'all_global': minimum of all columns and all sheets - a single
          value for absolutely everything loaded, regardless of the
          selected column.

        The result is cached (by column + normalization mode + smoothing),
        because a redraw triggered only by a panel resize doesn't change
        any of this data - this avoids reprocessing every sheet on every
        resize."""
        import pandas as pd

        mode = self.multi_xls_norm_mode_var.get()
        smoothing_on = self.multi_xls_smoothing_var.get()
        try:
            smoothing_points = self.multi_xls_smoothing_points_var.get()
        except tk.TclError:
            smoothing_points = None
        cache_key = (id(self.multi_xls_datasets), col_name, mode, smoothing_on, smoothing_points)
        if self._multi_xls_series_cache_key == cache_key:
            return self._multi_xls_series_cache

        column_global_min = None
        if mode == 'column_global':
            pooled = []
            for ds in self.multi_xls_datasets:
                if col_name in ds['df'].columns:
                    v = ds['df'][col_name].to_numpy(dtype=float)
                    pooled.append(v[np.isfinite(v)])
            if pooled:
                pooled = np.concatenate(pooled)
                if pooled.size:
                    column_global_min = pooled.min()

        all_global_min = None
        if mode == 'all_global':
            pooled = [ds['df'].to_numpy(dtype=float) for ds in self.multi_xls_datasets]
            finite = np.concatenate([m[np.isfinite(m)].ravel() for m in pooled]) if pooled else np.array([])
            if finite.size:
                all_global_min = finite.min()

        series = []
        for ds in self.multi_xls_datasets:
            if col_name not in ds['df'].columns:
                continue
            values = ds['df'][col_name].to_numpy(dtype=float)
            values = pd.Series(values).interpolate(limit_direction='both').to_numpy()
            if len(values) == 0:
                continue

            if mode == 'sheet':
                all_values = ds['df'].to_numpy(dtype=float)
                finite = all_values[np.isfinite(all_values)]
                divisor = finite.min() if finite.size else None
            elif mode == 'column_global':
                divisor = column_global_min
            elif mode == 'all_global':
                divisor = all_global_min
            else:  # 'local'
                finite = values[np.isfinite(values)]
                divisor = finite.min() if finite.size else None

            if divisor is not None and divisor != 0:
                values = values / divisor
            values = self._smooth_multi_xls_signal(values)
            series.append((ds['label'], values))

        self._multi_xls_series_cache_key = cache_key
        self._multi_xls_series_cache = series
        return series

    def _on_multi_xls_column_select(self, event=None):
        """Callback for when the user chooses a column in the sidebar list."""
        sel = self.multi_xls_column_listbox.curselection()
        if not sel or sel[0] >= len(self.multi_xls_common_columns):
            return
        self._draw_multi_xls_plot(sel[0])

    def _on_multi_xls_next_column(self):
        """Advance to the next column (e.g. Column 6 -> Column 7), wrapping
        back to the first after the last, and redraw both plots for it."""
        count = self.multi_xls_column_listbox.size()
        if count == 0:
            return
        current = self.multi_xls_current_index if self.multi_xls_current_index is not None else -1
        next_index = (current + 1) % count
        self.multi_xls_column_listbox.selection_clear(0, tk.END)
        self.multi_xls_column_listbox.selection_set(next_index)
        self.multi_xls_column_listbox.see(next_index)
        self._draw_multi_xls_plot(next_index)

    def _on_multi_xls_plot_frame_resize(self, event=None):
        """Redraw the line plot (with a small delay, to avoid redrawing on
        every pixel while the window is being dragged) so it keeps
        exactly filling the panel after a resize. While a sash is being
        dragged, nothing is scheduled: the (expensive, since it
        reprocesses every sheet's data) redraw happens only once, on
        release (see _on_multi_xls_sash_release), instead of risking the
        timer firing mid-drag, which would feel stuck."""
        if self._multi_xls_sash_dragging:
            return
        if self._multi_xls_plot_resize_job is not None:
            self.root.after_cancel(self._multi_xls_plot_resize_job)
        self._multi_xls_plot_resize_job = self.root.after(200, self._redraw_multi_xls_plot_for_resize)

    def _redraw_multi_xls_plot_for_resize(self):
        self._multi_xls_plot_resize_job = None
        if self.multi_xls_current_index is not None:
            self._draw_multi_xls_plot(self.multi_xls_current_index)

    def _on_multi_xls_heatmap_frame_resize(self, event=None):
        """Redraw the image panel (with a small delay) so it keeps
        exactly filling the panel after a resize. Same as in
        _on_multi_xls_plot_frame_resize, nothing is scheduled while a
        sash is being dragged."""
        if self._multi_xls_sash_dragging:
            return
        if self._multi_xls_heatmap_resize_job is not None:
            self.root.after_cancel(self._multi_xls_heatmap_resize_job)
        self._multi_xls_heatmap_resize_job = self.root.after(200, self._draw_multi_xls_heatmap)

    def _on_multi_xls_sash_release(self, event=None):
        """Redraw both plots immediately when the mouse is released after
        dragging the divider between the list and the plots, instead of
        waiting for the debounce timer used for window resizing (which
        does need that delay, since there is no "mouse released" event to
        hook into when the drag is controlled by the OS window manager
        instead of Tk)."""
        self._multi_xls_sash_dragging = False
        if self._multi_xls_plot_resize_job is not None:
            self.root.after_cancel(self._multi_xls_plot_resize_job)
            self._multi_xls_plot_resize_job = None
        if self._multi_xls_heatmap_resize_job is not None:
            self.root.after_cancel(self._multi_xls_heatmap_resize_job)
            self._multi_xls_heatmap_resize_job = None
        if self.multi_xls_current_index is not None:
            self._draw_multi_xls_plot(self.multi_xls_current_index)
        self._draw_multi_xls_heatmap()

    def _draw_multi_xls_plot(self, index):
        """Plot the column at position 'index' of each loaded sheet, side
        by side in the same figure, separated by vertical dashed lines.
        The plot's title uses the same label selected in the data list,
        and each sheet's labels are shown or hidden according to the
        'Show Data Names' checkbox. The canvas is created only once and
        reused on every redraw (see comment in __init__), resizing to
        exactly fit the panel so it's fully visible without scrolling."""
        if not self.multi_xls_datasets or self.multi_xls_plot_frame is None:
            return
        if index >= len(self.multi_xls_common_columns):
            return

        frame = self.multi_xls_plot_frame
        frame.update_idletasks()
        w_px, h_px = frame.winfo_width(), frame.winfo_height()
        if w_px <= 1 or h_px <= 1:
            # The panel isn't visible yet (e.g. the tab was just created).
            # Remember which column to plot; the <Configure> that fires
            # once the panel gets its real size will do the drawing.
            self.multi_xls_current_index = index
            self.multi_xls_current_column = self.multi_xls_common_columns[index]
            return

        col_name = self.multi_xls_common_columns[index]
        display_label = self.multi_xls_column_listbox.get(index)
        show_labels = self.multi_xls_show_labels_var.get()
        reserve_colorbar = self._multi_xls_effective_shared_scale()

        self.multi_xls_current_column = col_name
        self.multi_xls_current_index = index
        self._sync_multi_xls_class_row_values(index)

        class_row_h_in = (self.multi_xls_class_row.winfo_height() or 30) / 100
        fig_width = w_px / 100
        fig_height = max(h_px / 100 - class_row_h_in, 2.0)

        if self._multi_xls_plot_canvas is None:
            if self.multi_xls_plot_placeholder is not None and self.multi_xls_plot_placeholder.winfo_exists():
                self.multi_xls_plot_placeholder.destroy()
                self.multi_xls_plot_placeholder = None
            self.multi_xls_fig, self._multi_xls_plot_ax = plt.subplots(figsize=(fig_width, fig_height))
            self._multi_xls_plot_canvas = FigureCanvasTkAgg(self.multi_xls_fig, master=frame)
            self._multi_xls_plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self._multi_xls_plot_ax.clear()
            # forward=False: only updates the figure's internal size,
            # without telling Tk to resize the canvas widget.
            self.multi_xls_fig.set_size_inches(fig_width, fig_height, forward=False)

        peak_method = self.multi_xls_peak_method_var.get()
        peak_params = (self.multi_xls_peak_method_params.get(peak_method)
                        if peak_method != 'None' else None)
        show_smoothing_points = (self.multi_xls_smoothing_var.get()
                                  and self.multi_xls_show_smoothing_points_var.get())
        try:
            smoothing_points_n = self.multi_xls_smoothing_points_var.get()
        except tk.TclError:
            smoothing_points_n = None

        ax = self._multi_xls_plot_ax
        offset = 0
        tick_positions = []
        tick_labels = []
        bounds = []
        peak_legend_added = False
        baseline_legend_added = False
        for label, values in self._compute_multi_xls_series(col_name):
            n = len(values)
            x = np.arange(offset, offset + n)
            ax.plot(x, values, linewidth=0.8)
            if offset > 0:
                ax.axvline(offset, color=_C['sub'], linestyle='--', linewidth=1, alpha=0.6)
            tick_positions.append(offset + n / 2)
            tick_labels.append(label)
            bounds.append((offset, offset + n))

            if peak_params is not None:
                from peak_functions import compute_peaks
                peaks = compute_peaks(values.reshape(-1, 1), 0, peak_method, peak_params)
                if peaks:
                    peaks = np.asarray(peaks, dtype=int)
                    ax.scatter(offset + peaks, values[peaks], color='crimson', s=20, zorder=5,
                               label=None if peak_legend_added else 'Peaks')
                    peak_legend_added = True

            if show_smoothing_points and smoothing_points_n is not None and n >= 2:
                from peak_functions import convex_envelope_lowest_points
                try:
                    px, py = convex_envelope_lowest_points(values, n_points=smoothing_points_n)
                except Exception:
                    px = None
                if px is not None and len(px) > 0:
                    baseline = np.interp(np.arange(n), px, py)
                    first = not baseline_legend_added
                    ax.plot(x, baseline, color='darkorange', linewidth=1.1, linestyle='--',
                            label='Baseline' if first else None)
                    ax.scatter(offset + px, py, color='darkorange', s=18, zorder=5,
                               label='Lowest points used' if first else None)
                    baseline_legend_added = True

            offset += n

        if peak_legend_added or baseline_legend_added:
            ax.legend(fontsize=7, loc='upper right')

        ax.set_xticks(tick_positions)
        if show_labels:
            ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.set_title(display_label)
        norm_mode = self.multi_xls_norm_mode_var.get()
        if norm_mode == 'sheet':
            ax.set_ylabel('Value / minimum of the entire sheet')
        elif norm_mode == 'column_global':
            ax.set_ylabel('Value / minimum of the column (all sheets)')
        elif norm_mode == 'all_global':
            ax.set_ylabel('Value / global minimum (all columns and sheets)')
        else:
            ax.set_ylabel('Value / minimum of the column')

        # Manual axis limits set via "Axis Limits", if any; otherwise fit
        # tightly to the data range (matching the heatmap below) instead of
        # matplotlib's default 5% autoscale padding on each side.
        if self.multi_xls_xlim is not None:
            ax.set_xlim(self.multi_xls_xlim)
        else:
            ax.set_xlim(0, offset)
        if self.multi_xls_ylim is not None:
            ax.set_ylim(self.multi_xls_ylim)

        # Margins are computed in inches (not tight_layout's auto-padding or a
        # fixed fraction) so the axes always use as much of the panel as
        # possible: the bottom margin shrinks when the rotated sheet-name
        # labels are hidden, and the shared left/right bounds keep this plot
        # aligned with the heatmap below regardless of panel width.
        left, right, top, bottom = _multi_xls_axes_margins(
            fig_width, fig_height, show_labels, reserve_colorbar)
        self.multi_xls_fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

        self._multi_xls_plot_canvas.draw()

        self._position_multi_xls_class_row(ax, bounds)
        self.btn_multi_xls_next_column.lift()

    def _compute_multi_xls_heatmap_matrices(self):
        """Return (matrices, vmin, vmax, per_sheet_ranges) for the
        heatmap: each normalized and transposed matrix (rows=data
        columns, columns=samples); vmin/vmax (or None, None if a shared
        scale doesn't apply); and a list of per-sheet (min, max) for when
        it doesn't. Returns (None, None, None, None) if there is no
        finite data.

        Uses the same normalization mode (multi_xls_norm_mode_var) as the
        line plot above, adapted to the fact that here each sheet is
        shown in full (all its columns at once, one per row):
        - 'local': each row (data column) is normalized against its own
          minimum, computed only within that sheet.
        - 'sheet': the entire sheet (all its columns together) is
          normalized against a single minimum, within that sheet
          (this heatmap's original behavior).
        - 'column_global': each row (data column, by position) is
          normalized against the minimum of that same column position,
          pooling every loaded sheet.
        - 'all_global': a single minimum for absolutely everything loaded.

        If 'Smoothing' is on, each row (data column) also goes through
        the same Convex Envelope used by the line plot above
        (_smooth_multi_xls_signal), so the heatmap matches what that plot
        shows.

        Cached by (datasets, normalization mode, shared scale, smoothing,
        smoothing points) - none of this changes with the panel's size,
        so a redraw triggered only by a resize reuses the result instead
        of reprocessing every sheet."""
        mode = self.multi_xls_norm_mode_var.get()
        shared_scale = self.multi_xls_shared_scale_var.get()
        smoothing_on = self.multi_xls_smoothing_var.get()
        try:
            smoothing_points = self.multi_xls_smoothing_points_var.get()
        except tk.TclError:
            smoothing_points = None
        cache_key = (id(self.multi_xls_datasets), mode, shared_scale, smoothing_on, smoothing_points)
        if self._multi_xls_heatmap_matrices_cache_key == cache_key:
            return self._multi_xls_heatmap_matrices_cache

        raw_matrices = [ds['df'].to_numpy(dtype=float) for ds in self.multi_xls_datasets]
        finite_vals = (np.concatenate([m[np.isfinite(m)].ravel() for m in raw_matrices])
                       if raw_matrices else np.array([]))
        if finite_vals.size == 0:
            result = (None, None, None, None)
        else:
            matrices = []
            if mode == 'local':
                # Each data column normalized against its own minimum,
                # within that sheet only.
                for m in raw_matrices:
                    with np.errstate(invalid='ignore'):
                        col_mins = np.where(np.isfinite(m), m, np.nan)
                        col_mins = np.nanmin(col_mins, axis=0, keepdims=True)
                    col_mins = np.where(np.isfinite(col_mins) & (col_mins != 0), col_mins, 1.0)
                    matrices.append(self._smooth_multi_xls_matrix(m / col_mins).T)
            elif mode == 'column_global':
                # Each data column (by position) normalized against the
                # minimum of that same column position, pooled across
                # every loaded sheet.
                n_cols = max((m.shape[1] for m in raw_matrices), default=0)
                pooled_min = np.full(n_cols, np.nan)
                for m in raw_matrices:
                    for c in range(m.shape[1]):
                        col = m[:, c]
                        finite = col[np.isfinite(col)]
                        if finite.size:
                            v = finite.min()
                            pooled_min[c] = v if np.isnan(pooled_min[c]) else min(pooled_min[c], v)
                pooled_min = np.where(np.isfinite(pooled_min) & (pooled_min != 0), pooled_min, 1.0)
                for m in raw_matrices:
                    matrices.append(self._smooth_multi_xls_matrix(m / pooled_min[:m.shape[1]]).T)
            elif mode == 'all_global':
                all_min = finite_vals.min()
                if all_min == 0:
                    all_min = 1.0
                matrices = [self._smooth_multi_xls_matrix(m / all_min).T for m in raw_matrices]
            else:  # 'sheet' (default / previous behavior)
                for m in raw_matrices:
                    sheet_min = float(m[np.isfinite(m)].min())
                    matrices.append(self._smooth_multi_xls_matrix(m / sheet_min).T)

            per_sheet_ranges = []
            for m in matrices:
                finite = m[np.isfinite(m)]
                per_sheet_ranges.append((float(finite.min()), float(finite.max())))

            vmin = vmax = None
            if shared_scale:
                norm_finite = np.concatenate([m[np.isfinite(m)].ravel() for m in matrices])
                vmin, vmax = float(norm_finite.min()), float(norm_finite.max())

            result = (matrices, vmin, vmax, per_sheet_ranges)

        self._multi_xls_heatmap_matrices_cache_key = cache_key
        self._multi_xls_heatmap_matrices_cache = result
        return result

    def _draw_multi_xls_heatmap(self):
        """Draw, for each loaded sheet, an image where the X axis is the
        samples (the same axis as the line plot above) and the Y axis is
        the columns (the data series); each pixel's color represents its
        value divided by a minimum, computed according to the mode chosen
        in the 'Normalization' submenu (see
        _compute_multi_xls_heatmap_matrices) - the same mode used by the
        line plot above. If a manual range is set
        (multi_xls_heatmap_manual_range, see
        _open_multi_xls_heatmap_range_dialog), every sheet uses that
        range; otherwise, if 'Shared Color Scale' is on, every sheet
        shares the same automatic scale; otherwise, each sheet uses its
        own automatic range. Each sheet's images are placed side by side.
        The canvas is created only once and reused on every redraw (see
        comment in __init__), resizing to exactly fit the panel."""
        if self.multi_xls_heatmap_frame is None:
            return

        if not self.multi_xls_datasets:
            if self.multi_xls_heatmap_fig is not None:
                plt.close(self.multi_xls_heatmap_fig)
                self.multi_xls_heatmap_fig = None
            self._multi_xls_heatmap_canvas = None
            self._multi_xls_heatmap_ax = None
            self._multi_xls_heatmap_colorbar = None
            for w in list(self.multi_xls_heatmap_frame.winfo_children()):
                w.destroy()
            self.multi_xls_heatmap_placeholder = tk.Label(
                self.multi_xls_heatmap_frame, text="Sheet images will appear here",
                font=('Arial', 14), bg=_C['panel'], fg=_C['sub'])
            self.multi_xls_heatmap_placeholder.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            return

        frame = self.multi_xls_heatmap_frame
        frame.update_idletasks()
        w_px, h_px = frame.winfo_width(), frame.winfo_height()
        if w_px <= 1 or h_px <= 1:
            return

        shared_scale = self.multi_xls_shared_scale_var.get()
        manual_range = self.multi_xls_heatmap_manual_range
        effective_shared = shared_scale or manual_range is not None
        matrices, vmin, vmax, per_sheet_ranges = self._compute_multi_xls_heatmap_matrices()
        if matrices is None:
            return

        show_labels = self.multi_xls_show_labels_var.get()
        fig_width = w_px / 100
        fig_height = h_px / 100

        if self._multi_xls_heatmap_canvas is None:
            if self.multi_xls_heatmap_placeholder is not None and self.multi_xls_heatmap_placeholder.winfo_exists():
                self.multi_xls_heatmap_placeholder.destroy()
                self.multi_xls_heatmap_placeholder = None
            self.multi_xls_heatmap_fig, self._multi_xls_heatmap_ax = plt.subplots(figsize=(fig_width, fig_height))
            self._multi_xls_heatmap_canvas = FigureCanvasTkAgg(self.multi_xls_heatmap_fig, master=frame)
            self._multi_xls_heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            # Clear the whole figure (not just the axes) so the colorbar's
            # axes from the previous draw is discarded too - trying to
            # remove() just the old colorbar while reusing the main axes
            # left it in a broken state on the next redraw.
            self.multi_xls_heatmap_fig.clear()
            self._multi_xls_heatmap_ax = self.multi_xls_heatmap_fig.add_subplot(111)
            self._multi_xls_heatmap_colorbar = None
            # forward=False: only updates the figure's internal size,
            # without telling Tk to resize the canvas widget (that used
            # to trigger a new <Configure> on every redraw and caused an
            # infinite redraw loop).
            self.multi_xls_heatmap_fig.set_size_inches(fig_width, fig_height, forward=False)

        ax = self._multi_xls_heatmap_ax
        offset = 0
        tick_positions = []
        tick_labels = []
        im = None
        for ds, matrix, sheet_range in zip(self.multi_xls_datasets, matrices, per_sheet_ranges):
            n_cols, n_samples = matrix.shape
            if manual_range is not None:
                im_vmin, im_vmax = manual_range
            elif shared_scale:
                im_vmin, im_vmax = vmin, vmax
            else:
                im_vmin, im_vmax = sheet_range
            im = ax.imshow(matrix, aspect='auto', cmap=_MULTI_XLS_HEATMAP_CMAP, vmin=im_vmin, vmax=im_vmax,
                           extent=(offset, offset + n_samples, n_cols, 0))
            if offset > 0:
                ax.axvline(offset, color='white', linewidth=1.5)
            tick_positions.append(offset + n_samples / 2)
            tick_labels.append(ds['label'])
            offset += n_samples

        ax.set_xlim(0, offset)
        ax.set_xticks(tick_positions)
        if show_labels:
            ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel('Column')
        norm_mode_desc = {
            'local': 'minimum of each column in each sheet',
            'sheet': 'minimum of each sheet',
            'column_global': 'minimum of each column (all sheets)',
            'all_global': 'global minimum (all columns and sheets)',
        }.get(self.multi_xls_norm_mode_var.get(), 'minimum of each sheet')
        if manual_range is not None:
            ax.set_title(f'All sheets (value / {norm_mode_desc}, manual range)')
        elif shared_scale:
            ax.set_title(f'All sheets (value / {norm_mode_desc})')
        else:
            ax.set_title(f'All sheets (value / {norm_mode_desc}, per-sheet individual scale)')
        # Same shared left/right bounds as the top line plot (see
        # _multi_xls_axes_margins), so a sample's x-position lines up
        # vertically between the two. The colorbar (when shown) goes in its
        # own explicit axes carved out of the reserved right margin, instead
        # of letting fig.colorbar() shrink ax to make room for it.
        left, right, top, bottom = _multi_xls_axes_margins(
            fig_width, fig_height, show_labels, effective_shared)
        self.multi_xls_heatmap_fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
        pos = ax.get_position()
        if effective_shared and im is not None:
            cax_left = right + _MULTI_XLS_CBAR_GAP_IN / fig_width
            cax_width = _MULTI_XLS_CBAR_WIDTH_IN / fig_width
            cax = self.multi_xls_heatmap_fig.add_axes([cax_left, pos.y0, cax_width, pos.height])
            self._multi_xls_heatmap_colorbar = self.multi_xls_heatmap_fig.colorbar(im, cax=cax)

        self._multi_xls_heatmap_canvas.draw()

    def _save_multi_xls_plot_image(self):
        """Save the combined line plot of 'Multiple Files' to a file."""
        if self.multi_xls_fig is None:
            return
        from tkinter.filedialog import asksaveasfilename
        filename = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                       ("TIFF files", "*.tiff"), ("SVG files", "*.svg"),
                       ("EPS files", "*.eps"), ("All Files", "*.*")],
            title="Save Plot Image"
        )
        if filename:
            self.multi_xls_fig.savefig(filename, dpi=300, bbox_inches='tight')

    def _save_multi_xls_heatmap_image(self):
        """Save the heatmap image of 'Multiple Files' to a file."""
        if self.multi_xls_heatmap_fig is None:
            return
        from tkinter.filedialog import asksaveasfilename
        filename = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                       ("TIFF files", "*.tiff"), ("SVG files", "*.svg"),
                       ("EPS files", "*.eps"), ("All Files", "*.*")],
            title="Save Heatmap Image"
        )
        if filename:
            self.multi_xls_heatmap_fig.savefig(filename, dpi=300, bbox_inches='tight')

    def _save_multi_xls_smoothed_data(self):
        """Save, into a single .xlsx or .csv file, all the common columns
        with exactly the same processing shown in the top plot
        (interpolated, normalized according to 'multi_xls_norm_mode_var'
        and - if 'Smoothing' is on - smoothed with Convex Envelope), for
        each loaded sheet. Each sheet's data is stacked one below the
        other in a single spreadsheet (or a single CSV), separated by 20
        blank rows, instead of a separate Excel sheet for each. Unlike
        the plot, which only shows the selected column, this exports all
        the common columns. Processing and saving run in a separate
        thread (see _run_with_progress_window) so the progress window
        doesn't become unresponsive with large files."""
        if not self.multi_xls_datasets or not self.multi_xls_common_columns:
            messagebox.showwarning("No Data", "Load data first.")
            return

        from tkinter.filedialog import asksaveasfilename

        filename = asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All Files", "*.*")],
            title="Save Smoothed Data"
        )
        if not filename:
            return

        # Snapshot everything the worker thread needs up front, since Tk
        # widgets/variables aren't safe to read from a background thread -
        # the worker only ever touches these plain Python/pandas values.
        datasets = list(self.multi_xls_datasets)
        common_columns = list(self.multi_xls_common_columns)
        norm_mode = self.multi_xls_norm_mode_var.get()
        smoothing_on = self.multi_xls_smoothing_var.get()
        try:
            smoothing_points = self.multi_xls_smoothing_points_var.get()
        except tk.TclError:
            smoothing_points = None

        def worker(report_progress, report_error):
            import pandas as pd

            gap_rows = 20
            n_steps = len(datasets) + 1  # +1 for the final write step

            # Precompute the same divisors _compute_multi_xls_series uses,
            # so the exported values match the plot exactly for every
            # normalization mode - not just the column currently on screen.
            column_global_min = {}
            if norm_mode == 'column_global':
                for col_name in common_columns:
                    pooled = []
                    for ds in datasets:
                        if col_name in ds['df'].columns:
                            v = ds['df'][col_name].to_numpy(dtype=float)
                            pooled.append(v[np.isfinite(v)])
                    if pooled:
                        pooled = np.concatenate(pooled)
                        if pooled.size:
                            column_global_min[col_name] = pooled.min()

            all_global_min = None
            if norm_mode == 'all_global':
                pooled = [ds['df'].to_numpy(dtype=float) for ds in datasets]
                finite = np.concatenate([m[np.isfinite(m)].ravel() for m in pooled]) if pooled else np.array([])
                if finite.size:
                    all_global_min = finite.min()

            blocks = []
            for i, ds in enumerate(datasets):
                report_progress(i, n_steps, f"Processing {ds['label']}  ({i + 1}/{len(datasets)})")

                sheet_min = None
                if norm_mode == 'sheet':
                    all_values = ds['df'].to_numpy(dtype=float)
                    finite = all_values[np.isfinite(all_values)]
                    sheet_min = finite.min() if finite.size else None

                out_cols = {}
                for col_name in common_columns:
                    if col_name not in ds['df'].columns:
                        continue
                    values = ds['df'][col_name].to_numpy(dtype=float)
                    values = pd.Series(values).interpolate(limit_direction='both').to_numpy()

                    if norm_mode == 'sheet':
                        divisor = sheet_min
                    elif norm_mode == 'column_global':
                        divisor = column_global_min.get(col_name)
                    elif norm_mode == 'all_global':
                        divisor = all_global_min
                    else:  # 'local'
                        finite = values[np.isfinite(values)]
                        divisor = finite.min() if finite.size else None

                    if divisor is not None and divisor != 0:
                        values = values / divisor

                    if smoothing_on:
                        from peak_functions import _convex_envelope_detrend_signal
                        values = _convex_envelope_detrend_signal(values, n_points=smoothing_points)
                    out_cols[col_name] = values
                blocks.append(pd.DataFrame(out_cols))

            report_progress(len(datasets), n_steps, "Saving file...")
            stacked = []
            for i, block in enumerate(blocks):
                stacked.append(block)
                if i < len(blocks) - 1:
                    stacked.append(pd.DataFrame(np.nan, index=range(gap_rows), columns=block.columns))
            combined = pd.concat(stacked, ignore_index=True)
            if filename.lower().endswith('.csv'):
                combined.to_csv(filename, index=False, header=False)
            else:
                combined.to_excel(filename, index=False, header=False, engine='xlsxwriter')
            return filename

        self._run_with_progress_window(
            title="Saving Data", message="Saving smoothed data...",
            maximum=len(datasets) + 1, worker_fn=worker,
            on_complete=lambda fn: messagebox.showinfo("Saved", f"Data saved to:\n{fn}"),
            on_error=lambda exc: messagebox.showerror("Error", f"Could not save the file:\n{exc}"),
        )

    def _multi_xls_concatenated_column(self, col_name):
        """Return the single concatenated vector for 'col_name' across
        every loaded sheet, in the same order/segments as the top plot
        (via _compute_multi_xls_series) - used for correlation and the
        Dendrogram tab, both of which need one full-length vector per
        selected column rather than per-sheet segments."""
        series = self._compute_multi_xls_series(col_name)
        if not series:
            return np.array([])
        return np.concatenate([values for _, values in series])

    def _add_to_multi_xls_selection(self):
        """Add the highlighted Data Column to the Selection list (single
        selection, so at most one new entry per click)."""
        sel = self.multi_xls_column_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx not in self.multi_xls_selection_indices:
            self.multi_xls_selection_indices.append(idx)
            self.multi_xls_selection_indices.sort()
            self.multi_xls_selection_listbox.delete(0, tk.END)
            for i in self.multi_xls_selection_indices:
                self.multi_xls_selection_listbox.insert(tk.END, self.multi_xls_column_listbox.get(i))
            self._update_multi_xls_correlation_display()

    def _remove_from_multi_xls_selection(self):
        """Remove the highlighted entry from the Selection list."""
        sel = self.multi_xls_selection_listbox.curselection()
        if not sel:
            return
        list_idx = sel[0]
        self.multi_xls_selection_listbox.delete(list_idx)
        self.multi_xls_selection_indices.pop(list_idx)
        self._update_multi_xls_correlation_display()

    def _update_multi_xls_correlation_display(self):
        """Refresh the correlation heatmap in the bottom-most pane, computed
        over the concatenated cross-sheet series of every column currently
        in Selection (see _multi_xls_concatenated_column)."""
        if self.multi_xls_correlation_frame is None:
            return

        for widget in list(self.multi_xls_correlation_frame.winfo_children()):
            widget.destroy()

        if self._multi_xls_corr_fig is not None:
            plt.close(self._multi_xls_corr_fig)
            self._multi_xls_corr_fig = None
        self._multi_xls_corr_df = None
        if self.multi_xls_menu_grafica is not None:
            self.multi_xls_menu_grafica.entryconfigure("Save Correlation Data...", state='disabled')

        if len(self.multi_xls_selection_indices) < 2:
            tk.Label(
                self.multi_xls_correlation_frame,
                text="Add 2+ columns to Selection to see correlation",
                font=('Arial', 12), bg=_C['panel'], fg=_C['sub']
            ).pack(fill=tk.BOTH, expand=True)
            return

        import pandas as pd
        method = self.multi_xls_corr_method_var.get()
        col_labels = [self.multi_xls_column_listbox.get(i) for i in self.multi_xls_selection_indices]
        cols = {label: self._multi_xls_concatenated_column(self.multi_xls_common_columns[i])
                for i, label in zip(self.multi_xls_selection_indices, col_labels)}
        min_len = min(len(v) for v in cols.values())
        if min_len == 0:
            tk.Label(
                self.multi_xls_correlation_frame,
                text="Selected columns have no overlapping data",
                font=('Arial', 12), bg=_C['panel'], fg=_C['sub']
            ).pack(fill=tk.BOTH, expand=True)
            return
        df = pd.DataFrame({label: v[:min_len] for label, v in cols.items()})
        corr = df.corr(method=method)
        self._multi_xls_corr_df = corr
        if self.multi_xls_menu_grafica is not None:
            self.multi_xls_menu_grafica.entryconfigure("Save Correlation Data...", state='normal')

        self._multi_xls_corr_fig, ax = plt.subplots()
        cax = ax.matshow(corr.values, cmap='jet', vmin=-1, vmax=1)
        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(col_labels)))
        if self.multi_xls_show_corr_labels_var.get():
            ax.set_xticklabels(col_labels, rotation=45, ha='left', fontsize=8)
            ax.set_yticklabels(col_labels, fontsize=8)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        self._multi_xls_corr_fig.colorbar(cax, ax=ax, ticks=[-1, 0, 1], shrink=0.8)
        ax.set_title(f'{method.capitalize()} Correlation (Selection)', pad=20)
        self._multi_xls_corr_fig.tight_layout()

        corr_canvas = FigureCanvasTkAgg(self._multi_xls_corr_fig, master=self.multi_xls_correlation_frame)
        corr_canvas.draw()
        corr_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _save_multi_xls_correlation_image(self):
        """Save the current correlation heatmap to a file."""
        if self._multi_xls_corr_fig is None:
            messagebox.showwarning("No Plot", "Add 2+ columns to Selection first.")
            return
        from tkinter.filedialog import asksaveasfilename
        filename = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                       ("TIFF files", "*.tiff"), ("SVG files", "*.svg"),
                       ("EPS files", "*.eps"), ("All Files", "*.*")],
            title="Save Correlation Image"
        )
        if filename:
            self._multi_xls_corr_fig.savefig(filename, dpi=300, bbox_inches='tight')

    def _save_multi_xls_correlation_data(self):
        """Save the current correlation matrix values to a CSV or Excel file."""
        if self._multi_xls_corr_df is None:
            return
        from tkinter.filedialog import asksaveasfilename
        filename = asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"),
                       ("All Files", "*.*")],
            title="Save Correlation Data"
        )
        if not filename:
            return
        if filename.lower().endswith(('.xlsx', '.xls')):
            self._multi_xls_corr_df.to_excel(filename, sheet_name='Correlation', engine='xlsxwriter')
        else:
            self._multi_xls_corr_df.to_csv(filename)
        messagebox.showinfo("Saved", f"Correlation matrix saved to:\n{filename}")

    def _save_multi_xls_peaks_csv(self):
        """Run the current Peak Finder method on every column in Selection,
        for every loaded sheet, and save the peak flags (one row per
        sample, 1 where a peak was found) to a CSV or Excel file - the
        Multiple Files equivalent of Data Visualization's per-column peak
        export, expanded across sheets."""
        method = self.multi_xls_peak_method_var.get()
        if method == 'None':
            messagebox.showwarning("No Peak Method", "Select a peak finder method first.")
            return
        if not self.multi_xls_selection_indices:
            messagebox.showwarning("No Selection", "Add columns to Selection first.")
            return
        params = self.multi_xls_peak_method_params.get(method)
        if params is None:
            messagebox.showwarning("No Parameters",
                                   "Choose the peak finder method on a column first to set parameters.")
            return

        from tkinter.filedialog import asksaveasfilename
        from peak_functions import compute_peaks
        import pandas as pd

        filename = asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All Files", "*.*")],
            title="Save Peaks CSV"
        )
        if not filename:
            return

        rows = []
        for idx in self.multi_xls_selection_indices:
            col_name = self.multi_xls_common_columns[idx]
            col_label = self.multi_xls_column_listbox.get(idx)
            for label, values in self._compute_multi_xls_series(col_name):
                peaks = compute_peaks(values.reshape(-1, 1), 0, method, params)
                flags = np.zeros(len(values), dtype=int)
                if peaks:
                    flags[np.asarray(peaks, dtype=int)] = 1
                for sample_idx, flag in enumerate(flags):
                    rows.append({'Column': col_label, 'Sheet': label,
                                 'Sample_Index': sample_idx, 'Peak_Flag': flag})

        df_peaks = pd.DataFrame(rows)
        if filename.lower().endswith(('.xlsx', '.xls')):
            df_peaks.to_excel(filename, index=False, engine='xlsxwriter')
        else:
            df_peaks.to_csv(filename, index=False)
        messagebox.showinfo("Saved", f"Peak data saved to:\n{filename}")

    def _save_multi_xls_classifications(self):
        """Save, for each data column and each loaded sheet, the chosen
        classification (one row per column, one column per sheet), to an
        .xlsx or .csv file. Combinations not classified yet still use the
        same default value (the sheet's name) shown in the interface."""
        if not self.multi_xls_datasets or not self.multi_xls_common_columns:
            messagebox.showwarning("No Data", "Load data first.")
            return

        from tkinter.filedialog import asksaveasfilename
        import pandas as pd

        filename = asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All Files", "*.*")],
            title="Save Classifications"
        )
        if not filename:
            return

        sheet_labels = [ds['label'] for ds in self.multi_xls_datasets]
        rows = []
        for i, col_name in enumerate(self.multi_xls_common_columns):
            saved = self.multi_xls_classifications.get(i, {})
            row = {'Column': col_name}
            for label in sheet_labels:
                row[label] = saved.get(label, label)
            rows.append(row)

        # Also persist the classification options themselves, in their
        # current order, as an extra column -- so loading this file back
        # later can restore that exact dropdown ordering instead of just
        # inferring it from whichever cell values happen to appear first.
        for _ in range(max(0, len(self.multi_xls_classes) - len(rows))):
            rows.append({'Column': ''})
        for i, row in enumerate(rows):
            row['_ClassOptions'] = self.multi_xls_classes[i] if i < len(self.multi_xls_classes) else ''

        df = pd.DataFrame(rows).set_index('Column')
        if filename.lower().endswith('.csv'):
            df.to_csv(filename)
        else:
            df.to_excel(filename, engine='xlsxwriter')

        messagebox.showinfo("Saved", f"Classifications saved to:\n{filename}")

    def _load_multi_xls_classifications(self):
        """Load previously saved classifications (same format as "Save
        Classifications": one row per data column, one column per sheet)
        and apply them to the columns/sheets that match by name with what
        is currently loaded. Any new classification value is added to the
        options available in the comboboxes."""
        if not self.multi_xls_datasets or not self.multi_xls_common_columns:
            messagebox.showwarning("No Data", "Load data first.")
            return

        from tkinter.filedialog import askopenfilename
        import pandas as pd

        filename = askopenfilename(
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All Files", "*.*")],
            title="Load Classifications"
        )
        if not filename:
            return

        if filename.lower().endswith('.csv'):
            df = pd.read_csv(filename, index_col=0)
        else:
            df = pd.read_excel(filename, index_col=0)

        # Restore the classification options in the exact order they were
        # saved in, if that column is present, rather than only inferring an
        # order from whichever cell values happen to be encountered first.
        restored_classes = []
        if '_ClassOptions' in df.columns:
            for val in df['_ClassOptions']:
                if pd.isna(val):
                    continue
                val = str(val)
                if val and val not in restored_classes:
                    restored_classes.append(val)
            df = df.drop(columns=['_ClassOptions'])

        column_to_index = {name: i for i, name in enumerate(self.multi_xls_common_columns)}
        sheet_labels = {ds['label'] for ds in self.multi_xls_datasets}
        matched_sheets = [label for label in df.columns if label in sheet_labels]

        applied = 0
        new_classes = []
        for col_name, row in df.iterrows():
            col_idx = column_to_index.get(str(col_name))
            if col_idx is None:
                continue
            for label in matched_sheets:
                value = row[label]
                if pd.isna(value):
                    continue
                value = str(value)
                self.multi_xls_classifications.setdefault(col_idx, {})[label] = value
                applied += 1
                if value not in restored_classes and value not in new_classes:
                    new_classes.append(value)

        if not matched_sheets:
            messagebox.showwarning(
                "No Matches",
                "No column or sheet in the file matches the loaded data."
            )
            return

        if restored_classes or new_classes:
            # Saved order first, then anything from the current session or
            # the loaded data that wasn't part of the saved options list.
            combined = list(restored_classes)
            for c in self.multi_xls_classes:
                if c not in combined:
                    combined.append(c)
            for c in new_classes:
                if c not in combined:
                    combined.append(c)
            self.multi_xls_classes = combined
            for combo in self.multi_xls_class_combos.values():
                combo.configure(values=self.multi_xls_classes)

        if self.multi_xls_current_index is not None:
            self._sync_multi_xls_class_row_values(self.multi_xls_current_index)

        messagebox.showinfo("Loaded", f"Applied {applied} classifications from:\n{filename}")

    # ==================== IMAGE PROCESSING METHODS ====================
    
    def _load_default_image(self):
        """Load a default image or show a placeholder."""
        # Create placeholder image
        pil_img = Image.new('RGB', (400, 300), color='#1e1e1e')
        self._display_pil_image(pil_img)

    def _display_pil_image(self, pil_img):
        """Show a PIL image in the main label."""
        # Get the frame's available size
        self.image_frame.update_idletasks()
        frame_width = self.image_frame.winfo_width() - 20
        frame_height = self.image_frame.winfo_height() - 20

        if frame_width < 100:
            frame_width = int(self.width * 0.7)
        if frame_height < 100:
            frame_height = int(self.height * 0.9)

        # Resize while preserving aspect ratio
        width_pil, height_pil = pil_img.size
        ratio = min(frame_width / width_pil, frame_height / height_pil)
        new_size = (int(width_pil * ratio), int(height_pil * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)

        # Convert and display
        image_tk = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk  # Keep a reference

    def _apply_brightness_contrast(self, image_array):
        """Apply brightness and contrast to the image."""
        brightness = self.brightness_slider.get()
        contrast = self.contrast_slider.get()

        # Convert to float for the operations
        img = image_array.astype(np.float32)

        # Apply brightness
        img = img + brightness

        # Apply contrast
        if contrast != 0:
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            img = factor * (img - 128) + 128

        # Clamp values
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def _update_image_display(self):
        """Update the displayed image according to the current state."""
        if self.img_array is None:
            return

        slice_idx = self.slice_slider.get()
        current_slice = self.img_array[slice_idx, :, :].copy()

        # Apply brightness and contrast
        current_slice = self._apply_brightness_contrast(current_slice)

        # Apply threshold if enabled
        if self.threshold_enabled.get():
            threshold_val = self.threshold_slider.get()
            if HAS_IMAGE_MODULES:
                pil_img = threshold_image_pil(current_slice, threshold=threshold_val)
            else:
                pil_img = self._apply_binarize(current_slice, threshold_val)
        else:
            pil_img = Image.fromarray(current_slice)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

        self._display_pil_image(pil_img)

        # Update frame info
        total_frames = self.img_array.shape[0]
        self.frame_info_label.config(text=f"Frame: {slice_idx + 1} / {total_frames}")

    def _reset_adjustments(self):
        """Reset the brightness and contrast adjustments."""
        self.brightness_slider.set(0)
        self.contrast_slider.set(0)
        self._update_image_display()

    # ==================== CALLBACKS ====================

    def _on_slice_changed(self, val):
        """Callback for when the layer slider changes."""
        self._update_image_display()

    def _on_threshold_changed(self, val):
        """Callback for when the threshold slider changes."""
        if self.threshold_enabled.get():
            self._update_image_display()

    def _on_adjustment_changed(self, val):
        """Callback for when the brightness/contrast adjustments change."""
        self._update_image_display()

    # ==================== MENU COMMANDS - FILE ====================

    def open_ometiff_file(self):
        """Open an OME-TIFF file."""
        if HAS_IMAGE_MODULES:
            img, metadata, xml_metadata = load_ometiff_image()
        else:
            filename = filedialog.askopenfilename(
                title="Open OME-TIFF",
                filetypes=[("OME-TIFF files", "*.ome.tiff *.ome.tif"), ("All files", "*.*")]
            )
            if not filename:
                return

            try:
                reader = OMETIFFReader(fpath=filename)
                img, metadata, xml_metadata = reader.read()
            except Exception as e:
                messagebox.showerror("Error", f"Could not load the image:\n{str(e)}")
                return

        if img is None:
            return

        self.img_original = img
        self.img_array = img.copy()

        # Update the layer slider
        num_slices = self.img_array.shape[0]
        self.slice_slider.configure(to=num_slices - 1)
        self.slice_slider.set(0)

        # Reset adjustments
        self._reset_adjustments()

        # Update information
        shape = self.img_array.shape
        info_text = f"Dimensions: {shape[2]}x{shape[1]}\nFrames: {shape[0]}\nType: {self.img_array.dtype}"
        self.info_text.config(text=info_text)
        
        # Enable image menu items now that an image is loaded
        self.menu_imagen.entryconfig("Auto Contrast", state=NORMAL)
        self.menu_imagen.entryconfig("Histogram", state=NORMAL)
        self.menu_imagen.entryconfig("Binarize", state=NORMAL)
        self.menu_imagen.entryconfig("Restore Original", state=NORMAL)
        self.menu_bar.entryconfig("Variability Analysis", state=NORMAL)

        # Switch to the image tab
        self.notebook.select(self.image_tab)

        self._update_image_display()

    # ==================== MENU COMMANDS - IMAGE ====================

    def restore_original(self):
        """Restore the original image."""
        if self.img_original is None:
            messagebox.showwarning("Warning", "No original image loaded")
            return
        self.img_array = self.img_original.copy()
        self._reset_adjustments()

    def apply_auto_contrast(self):
        """Apply auto contrast to the image."""
        if self.img_array is None:
            messagebox.showwarning("Warning", "Load an OME-TIFF image first")
            return

        if HAS_IMAGE_MODULES:
            # Use the processing module if available
            self.img_array = auto_contrast(self.img_array)
        else:
            # Fallback: apply auto contrast frame by frame
            for i in range(self.img_array.shape[0]):
                im_pil = Image.fromarray(self.img_array[i, :, :])
                if im_pil.mode != 'RGB':
                    im_pil = im_pil.convert('RGB')
                im2 = ImageOps.autocontrast(im_pil, cutoff=2, ignore=2).convert('L')
                self.img_array[i, :, :] = np.array(im2)

        self._update_image_display()

    def show_histogram(self):
        """Show the image's histogram."""
        if self.img_original is None:
            messagebox.showwarning("Warning", "Load an OME-TIFF image first")
            return

        var_im = np.var(self.img_original, axis=0)
        plt.figure(figsize=(8, 6))
        plt.hist(var_im.flatten(), bins=50)
        plt.title('Variance Histogram')
        plt.xlabel('Variance')
        plt.ylabel('Frequency')
        plt.show()

    def show_binarize(self):
        """Show the binarization window."""
        if self.img_original is None:
            messagebox.showwarning("Warning", "Load an OME-TIFF image first")
            return

        var_im = np.var(self.img_original, axis=0)
        pil_img = self._apply_binarize(var_im, 150)

        # Create popup window
        top = tk.Toplevel(self.root)
        top.title("Binarization")
        top.geometry("600x700")

        # Frame for the image
        img_frame = tk.Frame(top, relief=tk.RAISED, borderwidth=1)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        image_ = ImageTk.PhotoImage(pil_img)
        label = tk.Label(img_frame, image=image_)
        label.image = image_
        label.pack(fill=tk.BOTH, expand=True)

        # Frame for controls
        control_frame = tk.Frame(top)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(control_frame, text="Threshold:").pack(side=tk.LEFT, padx=5)

        inputtxt = tk.Text(control_frame, height=1, width=10)
        inputtxt.insert("1.0", "150")
        inputtxt.pack(side=tk.LEFT, padx=5)

        def update_binarization():
            try:
                threshold = int(inputtxt.get(1.0, "end-1c"))
                new_img = self._apply_binarize(var_im, threshold)
                new_image = ImageTk.PhotoImage(new_img)
                label.configure(image=new_image)
                label.image = new_image
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")

        tk.Button(
            control_frame,
            text="Apply Binarization",
            command=update_binarization
        ).pack(side=tk.LEFT, padx=5)

    def _apply_binarize(self, var_im, threshold):
        """Apply binarization to the image based on the threshold."""
        if len(var_im.shape) == 2:
            # It's a 2D (variance) image
            binary = (var_im > threshold).astype(np.uint8) * 255
        else:
            # It's a regular image
            binary = (var_im > threshold).astype(np.uint8) * 255

        pil_img = Image.fromarray(binary)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        return pil_img

    def show_variability_menu(self, method_index):
        """Show the full variability analysis."""
        if self.img_array is None or len(self.img_array) == 0:
            messagebox.showwarning("Warning", "Load an OME-TIFF image first")
            return
        show_variability_analysis(self.img_array, method_index, self.root)

    # ==================== HELP / AUTO-UPDATER ====================

    _REPO = "sergiocruzunamia/neclabunam"
    _UPDATABLE_FILES = [
        "interface3.py",
        "peak_functions.py",
        "visualization_helpers.py",
        "corr_dendo_functions.py",
        "variability_functions.py",
        "image_loader.py",
        "image_processing.py",
        "multi_xls_helpers.py",
    ]

    def _get_local_sha(self):
        version_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version.json")
        try:
            with open(version_path, "r") as f:
                return json.load(f).get("sha", "")
        except Exception:
            return ""

    def _save_local_sha(self, sha):
        version_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version.json")
        with open(version_path, "w") as f:
            json.dump({"sha": sha}, f)

    def _check_for_updates(self):
        """Check GitHub for a newer version in a background thread."""
        messagebox.showinfo("Checking for Updates", "Checking for updates, please wait…")
        threading.Thread(target=self._fetch_update_info, daemon=True).start()

    def _fetch_update_info(self):
        api_url = f"https://api.github.com/repos/{self._REPO}/commits/main"
        try:
            req = urllib.request.Request(
                api_url,
                headers={"Accept": "application/vnd.github+json",
                         "User-Agent": "NecLab-Updater/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            remote_sha = data["sha"]
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Update Check Failed",
                f"Could not reach GitHub:\n{e}"
            ))
            return

        local_sha = self._get_local_sha()
        if remote_sha == local_sha:
            self.root.after(0, lambda: messagebox.showinfo(
                "Up to Date", "NecLab is already up to date."
            ))
            return

        # Ask user before downloading
        self.root.after(0, lambda: self._offer_update(remote_sha))

    def _offer_update(self, remote_sha):
        if not messagebox.askyesno(
            "Update Available",
            "A new version of NecLab is available on GitHub.\n\n"
            "Download and restart now?"
        ):
            return
        threading.Thread(target=self._download_and_restart,
                         args=(remote_sha,), daemon=True).start()

    def _download_and_restart(self, remote_sha):
        base_url = f"https://raw.githubusercontent.com/{self._REPO}/main/"
        app_dir = os.path.dirname(os.path.abspath(__file__))
        errors = []
        for filename in self._UPDATABLE_FILES:
            url = base_url + filename
            dest = os.path.join(app_dir, filename)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "NecLab-Updater/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content = resp.read()
                with open(dest, "wb") as f:
                    f.write(content)
            except Exception as e:
                errors.append(f"{filename}: {e}")

        if errors:
            msg = "Some files could not be downloaded:\n" + "\n".join(errors)
            self.root.after(0, lambda: messagebox.showwarning("Partial Update", msg))
            return

        try:
            self._save_local_sha(remote_sha)
        except Exception:
            pass

        self.root.after(0, self._restart_app)

    def _restart_app(self):
        if messagebox.askyesno("Restart", "Update complete. Restart NecLab now?"):
            subprocess.Popen([sys.executable] + sys.argv)
            self.root.quit()
            self.root.destroy()
            sys.exit(0)

    def _show_about(self):
        messagebox.showinfo(
            "About NecLab",
            "NecLab — Microscopy Image Analysis and Data Visualization\n\n"
            f"Repository: github.com/{self._REPO}\n"
            "Contact: sergio.cruz@ciencias.unam.mx"
        )


def main():
    """Main function."""
    root = tk.Tk()
    app = NecLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
