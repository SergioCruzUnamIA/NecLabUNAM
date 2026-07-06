"""
NecLab - Herramienta de análisis de imágenes de microscopía y visualización de datos
Interfaz gráfica principal (versión unificada)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # Limita número de threads

import tkinter as tk
from tkinter import Menu, Grid, filedialog, FALSE, DISABLED, NORMAL, ttk, messagebox
import customtkinter as ctk
ctk.set_appearance_mode("light")
from PIL import Image, ImageTk, ImageOps
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import threading
import subprocess
import json
import urllib.request

# Shared horizontal axes bounds (figure-fraction) for the Datos Multiples top
# line plot and bottom heatmap, so a sample's x-position lines up vertically
# between the two regardless of the heatmap's colorbar taking extra space.
_MULTI_XLS_AX_LEFT = 0.09
_MULTI_XLS_AX_RIGHT = 0.86

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

# Módulos locales
from pyometiff import OMETIFFReader
from visualization_helpers import initialize_visualization
from variability_functions import show_variability_analysis, get_variability_methods
from corr_dendo_functions import load_correlation_matrix
from multi_xls_helpers import pick_files_and_sheets, load_selected_sheets, common_column_names

# Intentar importar módulos de procesamiento de imagen si existen
try:
    from image_loader import load_ometiff_image, process_image_slice
    from image_processing import auto_contrast, threshold_image_pil
    HAS_IMAGE_MODULES = True
except ImportError:
    HAS_IMAGE_MODULES = False



class NecLabApp:
    """Clase principal de la aplicación NecLab - Versión unificada."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("NecLab - Análisis de Imágenes y Datos")
        self.root.tk.call('tk', 'windowingsystem')
        self.root.option_add('*tearOff', FALSE)
        
        # Configurar tamaño de ventana (90% de la pantalla)
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.width = int(self.screen_width * 0.9)
        self.height = int(self.screen_height * 0.9)
        self.root.geometry(f"{self.width}x{self.height}")
        
        # Variables de estado - Imágenes
        self.img_original = None  # Imagen original sin modificar
        self.img_array = None     # Imagen de trabajo (puede tener modificaciones)
        self.img_display = None   # Imagen para visualización (con contraste, etc.)
        
        # Variables de estado - Datos de visualización
        self.loaded_data = None
        self.current_column = 0
        self.canvas = None
        self.corr = None
        self._data_fig = None
        self._corr_fig = None
        self.selection_column_indices = []
        self.plot_top_frame = None
        self.plot_bottom_frame = None
        self.corr_method_var = tk.StringVar(value='pearson')
        self.peak_method_var = tk.StringVar(value='None')
        self.show_corr_labels_var = tk.BooleanVar(value=True)
        self.smoothing_var = tk.BooleanVar(value=False)
        self.smoothing_points_var = tk.IntVar(value=2)
        self.show_smoothing_points_var = tk.BooleanVar(value=True)
        self._mouse_click = False       # flag to suppress double-redraw on mouse click
        self.btn_save_data = None
        self.btn_save_corr = None
        self.btn_save_peaks = None
        self.peak_method_combo = None
        self.smoothing_check = None
        self.smoothing_points_spinbox = None
        self.show_smoothing_points_check = None
        self.peak_method_params = {}  # saved params per method name
        self.plot_mid_frame = None
        self._mid_fig = None

        # Variables de estado - Tab Datos Multiples (múltiples archivos .xls)
        self.multi_xls_tab = None
        self.multi_xls_datasets = []
        self.multi_xls_common_columns = []
        self.multi_xls_column_listbox = None
        self.multi_xls_show_labels_var = tk.BooleanVar(value=False)
        self.multi_xls_smoothing_var = tk.BooleanVar(value=True)
        self.multi_xls_smoothing_points_var = tk.IntVar(value=2)
        self.multi_xls_smoothing_check = None
        self.multi_xls_smoothing_points_spinbox = None
        self.multi_xls_shared_scale_var = tk.BooleanVar(value=True)
        self.multi_xls_plot_frame = None
        self.multi_xls_fig = None
        self.multi_xls_current_column = None
        self.multi_xls_current_index = None
        self.multi_xls_heatmap_frame = None
        self.multi_xls_heatmap_fig = None
        self._multi_xls_plot_resize_job = None
        self._multi_xls_heatmap_resize_job = None
        self.btn_save_multi_xls_plot = None
        self.btn_save_multi_xls_heatmap = None
        self.multi_xls_class_row = None
        self.multi_xls_plot_placeholder = None
        self.multi_xls_heatmap_placeholder = None
        # El canvas/figura/ejes de cada gráfica se crean una sola vez y se
        # reutilizan en cada redibujado (en vez de destruir y crear uno
        # nuevo cada vez), y el tamaño se actualiza con forward=False: así
        # nunca se le ordena a Tk que redimensione el widget del canvas,
        # que es lo que generaba un <Configure> nuevo en cada redibujado y
        # provocaba un ciclo de redibujado infinito.
        self._multi_xls_plot_canvas = None
        self._multi_xls_plot_ax = None
        self._multi_xls_heatmap_canvas = None
        self._multi_xls_heatmap_ax = None
        self._multi_xls_heatmap_colorbar = None
        self.multi_xls_classes = []           # nombres disponibles para clasificar hojas (cosmético)
        self.multi_xls_sheet_class_var = {}    # sheet label -> tk.StringVar con la clasificación elegida
        self.multi_xls_class_combos = {}       # sheet label -> ttk.Combobox (posicionado sobre su segmento)
        self.multi_xls_classifications = {}    # col_index -> {sheet label -> clasificación guardada}
        self._multi_xls_syncing_class_row = False

        # Variables de estado - Tab Dendograma
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

        # Construir la interfaz
        self._create_menu()
        self._create_layout()
        
        # Manejar cierre de ventana para liberar todo el proceso
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self.root.lift()
        self.root.focus_force()
    
    def _on_close(self):
        """Cerrar la aplicación completamente, liberando todos los recursos."""
        if messagebox.askokcancel("Salir", "¿Desea salir de NecLab?"):
            plt.close('all')
            self.root.quit()
            self.root.destroy()
            sys.exit(0)
    
    # ==================== MENÚ ====================
    
    def _create_menu(self):
        """Crea la barra de menús unificada."""
        self.menu_bar = Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        # Menú Archivo
        self.menu_archivo = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_archivo, label="Archivo")
        self.menu_archivo.add_command(
            label="Abrir OME-TIFF",
            accelerator="Ctrl+O",
            command=self.open_ometiff_file
        )
        self.menu_archivo.add_separator()
        self.menu_archivo.add_command(
            label='Abrir Datos (.npy / .csv)',
            command=self.open_visualization_data,
            state=NORMAL
        )
        self.menu_archivo.add_command(
            label='Cargar Matriz de Correlacion',
            command=self.load_correlation_matrix_wrapper,
            state=NORMAL
        )
        self.menu_archivo.add_separator()
        self.menu_archivo.add_command(
            label='Abrir Multiples Archivos (.xls)',
            command=self.open_multiple_xls_files,
            state=NORMAL
        )
        self.menu_archivo.add_separator()
        self.menu_archivo.add_command(
            label="Salir",
            command=self._on_close
        )
        
        # Menú Imagen
        self.menu_imagen = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_imagen, label="Imagen")
        self.menu_imagen.add_command(
            label="Auto Contraste",
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
            label="Restaurar Original",
            command=self.restore_original,
            state=DISABLED
        )

        # Menú Análisis de Variabilidad (top-level, between Imagen and Visualizacion)
        self.menu_variabilidad = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_variabilidad, label="Análisis de Variabilidad", state=DISABLED)

        # Agregar los 7 métodos de variabilidad
        methods = get_variability_methods()
        for i, method_name in enumerate(methods):
            self.menu_variabilidad.add_command(
                label=method_name,
                command=lambda idx=i: self.show_variability_menu(idx)
            )

        # Menú Visualización
        self.menu_visual = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_visual, label="Visualizacion")

        self.menu_visual.add_command(label='Dendograma', command=None, state=DISABLED)
        self.menu_visual.add_separator()
        self.menu_visual.add_command(label='Series de tiempo', command=None, state=DISABLED)

        # Menú Ayuda
        self.menu_ayuda = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_ayuda, label="Help")
        self.menu_ayuda.add_command(label='Check for Updates', command=self._check_for_updates)
        self.menu_ayuda.add_separator()
        self.menu_ayuda.add_command(label='About NecLab', command=self._show_about)

        # Atajo de teclado
        self.root.bind('<Control-o>', lambda e: self.open_ometiff_file())
    
    # ==================== LAYOUT PRINCIPAL ====================
    
    def _create_layout(self):
        """Crea el layout principal con tabs para diferentes modos."""
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

        # Tab 1: Procesamiento de Imágenes
        self.image_tab = tk.Frame(self.notebook, bg=_C['bg'])
        self.notebook.add(self.image_tab, text="  Procesamiento de Imágenes  ")
        self._create_image_processing_layout()

        # Tab 2: Visualización de Datos
        self.data_tab = tk.Frame(self.notebook, bg=_C['bg'])
        self.notebook.add(self.data_tab, text="  Visualización de Datos  ")
        self._create_data_visualization_layout()

    
    def _create_image_processing_layout(self):
        """Crea el layout para procesamiento de imágenes."""
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
        """Crea el panel lateral derecho con controles de imagen."""
        self.controls_panel = tk.Frame(self.image_tab, bg=_C['panel'],
                                       highlightbackground=_C['border'],
                                       highlightthickness=1)
        self.controls_panel.grid(row=0, column=1, sticky='nsew', padx=(4, 8), pady=8)

        tk.Label(self.controls_panel, text="Controles de Imagen",
                 font=('Arial', 13, 'bold'), bg=_C['panel'], fg=_C['text']).pack(pady=12)

        tk.Frame(self.controls_panel, bg=_C['border'], height=1).pack(fill='x', padx=10)
        
        # ===== SECCIÓN: Navegación =====
        self._create_section_navigation()
        
        # ===== SECCIÓN: Ajustes de Imagen =====
        self._create_section_image_adjustments()
        
        # ===== SECCIÓN: Procesamiento =====
        self._create_section_processing()
        
        # ===== SECCIÓN: Información =====
        self._create_section_info()
    
    def _create_section_navigation(self):
        """Sección de navegación de frames."""
        def _sec_label(parent, text):
            tk.Label(parent, text=text, font=('Arial', 8, 'bold'),
                     bg=_C['panel'], fg=_C['sub']).pack(anchor='w', padx=12, pady=(12, 2))
            tk.Frame(parent, bg=_C['border'], height=1).pack(fill='x', padx=10)

        _sec_label(self.controls_panel, "NAVEGACIÓN")

        inner = tk.Frame(self.controls_panel, bg=_C['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        tk.Label(inner, text="Capa (Frame):", bg=_C['panel'],
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
        """Sección de ajustes de imagen."""
        def _sec_label(text):
            tk.Label(self.controls_panel, text=text, font=('Arial', 8, 'bold'),
                     bg=_C['panel'], fg=_C['sub']).pack(anchor='w', padx=12, pady=(12, 2))
            tk.Frame(self.controls_panel, bg=_C['border'], height=1).pack(fill='x', padx=10)

        _sec_label("AJUSTES DE IMAGEN")
        inner = tk.Frame(self.controls_panel, bg=_C['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        tk.Label(inner, text="Brillo:", bg=_C['panel'], fg=_C['text'], font=('Arial', 9)).pack(anchor='w')
        self.brightness_slider = tk.Scale(inner, from_=-100, to=100, orient="horizontal",
                                          command=self._on_adjustment_changed,
                                          bg=_C['panel'], troughcolor=_C['card'],
                                          highlightthickness=0, relief='flat', sliderlength=16)
        self.brightness_slider.set(0)
        self.brightness_slider.pack(fill='x')

        tk.Label(inner, text="Contraste:", bg=_C['panel'], fg=_C['text'],
                 font=('Arial', 9)).pack(anchor='w', pady=(5, 0))
        self.contrast_slider = tk.Scale(inner, from_=-100, to=100, orient="horizontal",
                                        command=self._on_adjustment_changed,
                                        bg=_C['panel'], troughcolor=_C['card'],
                                        highlightthickness=0, relief='flat', sliderlength=16)
        self.contrast_slider.set(0)
        self.contrast_slider.pack(fill='x')

        ctk.CTkButton(inner, text="Auto Contraste", height=30, corner_radius=6,
                      fg_color=_C['acc'], hover_color=_C['acc2'], text_color='white',
                      font=ctk.CTkFont(size=11),
                      command=self.apply_auto_contrast).pack(fill='x', pady=(6, 2))
        ctk.CTkButton(inner, text="Resetear Ajustes", height=30, corner_radius=6,
                      fg_color=_C['card'], hover_color=_C['border'],
                      text_color=_C['text'], border_width=1, border_color=_C['border'],
                      font=ctk.CTkFont(size=11),
                      command=self._reset_adjustments).pack(fill='x', pady=2)

    def _create_section_processing(self):
        """Sección de procesamiento."""
        def _sec_label(text):
            tk.Label(self.controls_panel, text=text, font=('Arial', 8, 'bold'),
                     bg=_C['panel'], fg=_C['sub']).pack(anchor='w', padx=12, pady=(12, 2))
            tk.Frame(self.controls_panel, bg=_C['border'], height=1).pack(fill='x', padx=10)

        _sec_label("PROCESAMIENTO")
        inner = tk.Frame(self.controls_panel, bg=_C['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        row = tk.Frame(inner, bg=_C['panel'])
        row.pack(fill='x')
        tk.Label(row, text="Threshold:", bg=_C['panel'], fg=_C['text'], font=('Arial', 9)).pack(side='left')
        self.threshold_enabled = tk.BooleanVar(value=False)
        self.threshold_check = tk.Checkbutton(row, text="Aplicar", variable=self.threshold_enabled,
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
        """Sección de información de la imagen."""
        def _sec_label(text):
            tk.Label(self.controls_panel, text=text, font=('Arial', 8, 'bold'),
                     bg=_C['panel'], fg=_C['sub']).pack(anchor='w', padx=12, pady=(12, 2))
            tk.Frame(self.controls_panel, bg=_C['border'], height=1).pack(fill='x', padx=10)

        _sec_label("INFORMACIÓN")
        inner = tk.Frame(self.controls_panel, bg=_C['panel'], padx=10, pady=6)
        inner.pack(fill='x')

        self.info_text = tk.Label(inner, text="No hay imagen cargada", bg=_C['panel'],
                                   fg=_C['text'], justify='left', anchor='w', font=('Arial', 9))
        self.info_text.pack(fill='x')
    
    def _create_data_visualization_layout(self):
        """Crea el layout para visualización de datos."""
        Grid.rowconfigure(self.data_tab, 0, weight=1)
        Grid.columnconfigure(self.data_tab, 0, weight=0)
        Grid.columnconfigure(self.data_tab, 1, weight=1)

        # ── Sidebar ──────────────────────────────────────────────────────────
        sidebar_frame = tk.Frame(self.data_tab, bg=_C['panel'], width=250,
                                 highlightbackground=_C['border'], highlightthickness=1)
        sidebar_frame.grid(row=0, column=0, sticky='nsew', padx=(8, 0), pady=8)
        sidebar_frame.grid_propagate(False)
        sidebar_frame.columnconfigure(0, weight=1)

        def _sec(parent, text, row):
            tk.Label(parent, text=text, font=('Arial', 8, 'bold'),
                     bg=_C['panel'], fg=_C['sub']).grid(row=row, column=0, sticky='w',
                                                         padx=12, pady=(12, 2))
            row += 1
            tk.Frame(parent, bg=_C['border'], height=1).grid(row=row, column=0,
                                                              sticky='ew', padx=10)
            return row + 1

        row = _sec(sidebar_frame, "COLUMNAS DE DATOS", 0)

        # Column listbox
        listbox_frame = tk.Frame(sidebar_frame, bg=_C['card'],
                                 highlightbackground=_C['border'], highlightthickness=1)
        listbox_frame.grid(row=row, column=0, sticky='nsew', padx=10, pady=(2, 4))
        sidebar_frame.rowconfigure(row, weight=2)
        listbox_frame.rowconfigure(0, weight=1)
        listbox_frame.columnconfigure(0, weight=1)
        row += 1

        scrollbar = tk.Scrollbar(listbox_frame, relief='flat', width=10)
        scrollbar.grid(row=0, column=1, sticky='ns')

        self.column_listbox = tk.Listbox(
            listbox_frame, yscrollcommand=scrollbar.set,
            selectmode=tk.EXTENDED, font=('Arial', 10),
            bg=_C['card'], fg=_C['text'],
            selectbackground=_C['acc'], selectforeground='white',
            relief='flat', bd=0, highlightthickness=0, activestyle='none'
        )
        self.column_listbox.grid(row=0, column=0, sticky='nsew')
        self.column_listbox.bind('<ButtonPress-1>', lambda e: setattr(self, '_mouse_click', True))
        self.column_listbox.bind('<<ListboxSelect>>', self.update_column_display)
        self.column_listbox.bind('<ButtonRelease-1>', self._on_column_click)
        scrollbar.config(command=self.column_listbox.yview)

        # ── Peak Finder ──
        row = _sec(sidebar_frame, "PEAK FINDER", row)

        self.peak_method_combo = ttk.Combobox(
            sidebar_frame, textvariable=self.peak_method_var,
            values=['None', 'Elliptic Envelope', 'Peak Caller', 'Local Outlier Factor',
                    'Peak Function 4', 'Isolation Forest', 'Linear Model', 'Peak Function 7'],
            state='readonly', width=22
        )
        self.peak_method_combo.grid(row=row, column=0, padx=10, pady=(2, 2), sticky='ew')
        self.peak_method_combo.bind('<<ComboboxSelected>>', lambda e: self._run_peak_on_column(show_dialog=True))
        self.peak_method_combo.config(state=DISABLED)
        row += 1

        # Smoothing row
        smooth_frame = tk.Frame(sidebar_frame, bg=_C['panel'])
        smooth_frame.grid(row=row, column=0, sticky='ew', padx=10, pady=(0, 2))
        smooth_frame.columnconfigure(0, weight=1)
        row += 1

        self.smoothing_check = tk.Checkbutton(
            smooth_frame, text="Smoothing", variable=self.smoothing_var,
            command=self._on_smoothing_toggle,
            bg=_C['panel'], fg=_C['text'], font=('Arial', 10),
            activebackground=_C['panel'], selectcolor=_C['panel'],
            state=DISABLED
        )
        self.smoothing_check.grid(row=0, column=0, sticky='w')

        self.show_smoothing_points_check = tk.Checkbutton(
            smooth_frame, text="Show points", variable=self.show_smoothing_points_var,
            command=self._on_smoothing_toggle,
            bg=_C['panel'], fg=_C['text'], font=('Arial', 10),
            activebackground=_C['panel'], selectcolor=_C['panel'],
            state=DISABLED
        )
        self.show_smoothing_points_check.grid(row=0, column=1, sticky='e')

        points_frame = tk.Frame(smooth_frame, bg=_C['panel'])
        points_frame.grid(row=1, column=0, sticky='w')
        tk.Label(points_frame, text="Points:", bg=_C['panel'],
                 fg=_C['sub'], font=('Arial', 8)).pack(side='left', padx=(0, 2))
        self.smoothing_points_spinbox = ttk.Spinbox(
            points_frame, from_=2, to=50, textvariable=self.smoothing_points_var,
            width=4
        )
        self.smoothing_points_spinbox.pack(side='left')
        self.smoothing_points_spinbox.config(state=DISABLED)
        self.smoothing_points_var.trace_add('write', lambda *args: self._on_smoothing_toggle())

        # ── Correlation ──
        row = _sec(sidebar_frame, "CORRELACIÓN", row)

        corr_method_combo = ttk.Combobox(
            sidebar_frame, textvariable=self.corr_method_var,
            values=['pearson', 'kendall', 'spearman'], state='readonly', width=15
        )
        corr_method_combo.grid(row=row, column=0, padx=10, pady=(2, 2), sticky='w')
        corr_method_combo.bind('<<ComboboxSelected>>', lambda e: self._update_correlation_display())
        row += 1

        tk.Checkbutton(sidebar_frame, text="Show Labels", variable=self.show_corr_labels_var,
                       command=self._update_correlation_display,
                       bg=_C['panel'], fg=_C['text'], selectcolor=_C['card'],
                       activebackground=_C['panel'], font=('Arial', 9)).grid(
            row=row, column=0, sticky='w', padx=10, pady=(0, 4))
        row += 1

        # ── Selection ──
        row = _sec(sidebar_frame, "SELECCIÓN", row)

        sel_listbox_frame = tk.Frame(sidebar_frame, bg=_C['card'],
                                     highlightbackground=_C['border'], highlightthickness=1)
        sel_listbox_frame.grid(row=row, column=0, sticky='nsew', padx=10, pady=(2, 4))
        sidebar_frame.rowconfigure(row, weight=1)
        sel_listbox_frame.rowconfigure(0, weight=1)
        sel_listbox_frame.columnconfigure(0, weight=1)
        row += 1

        sel_scrollbar = tk.Scrollbar(sel_listbox_frame, relief='flat', width=10)
        sel_scrollbar.grid(row=0, column=1, sticky='ns')

        self.selection_listbox = tk.Listbox(
            sel_listbox_frame, yscrollcommand=sel_scrollbar.set,
            selectmode=tk.SINGLE, font=('Arial', 10),
            bg=_C['card'], fg=_C['text'],
            selectbackground=_C['acc'], selectforeground='white',
            relief='flat', bd=0, highlightthickness=0, activestyle='none'
        )
        self.selection_listbox.grid(row=0, column=0, sticky='nsew')
        sel_scrollbar.config(command=self.selection_listbox.yview)

        self.btn_add_sel = ctk.CTkButton(
            sidebar_frame, text="Add to Selection", height=28, corner_radius=6,
            fg_color=_C['acc'], hover_color=_C['acc2'], text_color='white',
            font=ctk.CTkFont(size=11), state='disabled',
            command=self._add_to_selection
        )
        self.btn_add_sel.grid(row=row, column=0, sticky='ew', padx=10, pady=(2, 2))
        row += 1

        self.btn_remove_sel = ctk.CTkButton(
            sidebar_frame, text="Remove from Selection", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            state='disabled', command=self._remove_from_selection
        )
        self.btn_remove_sel.grid(row=row, column=0, sticky='ew', padx=10, pady=(0, 4))
        row += 1

        # ── Save buttons ──
        tk.Frame(sidebar_frame, bg=_C['border'], height=1).grid(
            row=row, column=0, sticky='ew', padx=10, pady=4
        )
        row += 1

        self.btn_save_data = ctk.CTkButton(
            sidebar_frame, text="Save Data Image", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            state='disabled', command=self._save_data_image
        )
        self.btn_save_data.grid(row=row, column=0, sticky='ew', padx=10, pady=(2, 2))
        row += 1

        self.btn_save_corr = ctk.CTkButton(
            sidebar_frame, text="Save Correlation", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            state='disabled', command=self._save_correlation_image
        )
        self.btn_save_corr.grid(row=row, column=0, sticky='ew', padx=10, pady=(0, 2))
        row += 1

        self.btn_save_peaks = ctk.CTkButton(
            sidebar_frame, text="Save Peaks CSV", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            state='disabled', command=self._save_peaks_csv
        )
        self.btn_save_peaks.grid(row=row, column=0, sticky='ew', padx=10, pady=(0, 10))

        # Right side - main plot area
        self.main_plot_frame = tk.Frame(self.data_tab, bg=_C['panel'],
                                         highlightbackground=_C['border'],
                                         highlightthickness=1)
        self.main_plot_frame.grid(row=0, column=1, sticky='nsew', padx=8, pady=8)

        tk.Label(self.main_plot_frame, text="Load a data file to start",
                 font=('Arial', 18), bg=_C['panel'], fg=_C['sub']).pack(
            fill=tk.BOTH, expand=True)

    
    # ==================== DATA VISUALIZATION METHODS ====================

    def _on_column_click(self, event):
        """Set current_column from the exact row under the mouse, then redraw."""
        self._mouse_click = False
        idx = self.column_listbox.nearest(event.y)
        if idx < 0 or self.loaded_data is None:
            return
        self.current_column = idx
        self._run_peak_on_column()

    def update_column_display(self, event=None):
        """Update the data graph when a column is selected. Does not refresh correlation."""
        if self._mouse_click:
            return
        if self.loaded_data is None:
            if hasattr(self.root, 'loaded_data'):
                self.loaded_data = self.root.loaded_data
        if self.loaded_data is None:
            return

        selection = self.column_listbox.curselection()
        if not selection:
            return

        # For keyboard navigation (arrow keys) use the last item in the selection.
        # Mouse clicks are handled by _on_column_click which sets current_column first.
        if self.current_column not in selection:
            self.current_column = selection[-1]

        # Delegate drawing to peak runner (handles None → raw data, or a peak method,
        # and ensures the split layout frames exist before drawing).
        self._run_peak_on_column()

    def _ensure_plot_frames(self):
        """Create the split top/mid/bottom plot layout on first use (or if destroyed)."""
        if self.plot_top_frame is not None and self.plot_top_frame.winfo_exists():
            return
        for widget in list(self.main_plot_frame.winfo_children()):
            widget.destroy()
        self.main_plot_frame.rowconfigure(0, weight=2)
        self.main_plot_frame.rowconfigure(1, weight=2)
        self.main_plot_frame.rowconfigure(2, weight=2)
        self.main_plot_frame.columnconfigure(0, weight=1)
        self.plot_top_frame = tk.Frame(self.main_plot_frame)
        self.plot_top_frame.grid(row=0, column=0, sticky='nsew')
        self.plot_mid_frame = tk.Frame(self.main_plot_frame,
                                       highlightbackground=_C['border'],
                                       highlightthickness=1)
        self.plot_mid_frame.grid(row=1, column=0, sticky='nsew')
        self.plot_bottom_frame = tk.Frame(self.main_plot_frame, relief=tk.GROOVE, borderwidth=1)
        self.plot_bottom_frame.grid(row=2, column=0, sticky='nsew')
        self._update_correlation_display()

    def _update_correlation_display(self):
        """Refresh the Pearson correlation heatmap in the bottom frame."""
        if self.plot_bottom_frame is None:
            return

        for widget in list(self.plot_bottom_frame.winfo_children()):
            widget.destroy()

        if self._corr_fig is not None:
            plt.close(self._corr_fig)
            self._corr_fig = None

        if len(self.selection_column_indices) < 2:
            tk.Label(
                self.plot_bottom_frame,
                text="Add 2+ columns to Selection to see correlation",
                font=('Arial', 12), bg=_C['panel'], fg=_C['sub']
            ).pack(fill=tk.BOTH, expand=True)
            return

        import pandas as pd
        method = self.corr_method_var.get()
        sel_indices = self.selection_column_indices
        data_sel = self.loaded_data[:, sel_indices]
        col_labels = [self.column_listbox.get(i) for i in sel_indices]
        df = pd.DataFrame(data_sel, columns=col_labels)
        corr = df.corr(method=method)

        self._corr_fig, ax = plt.subplots()
        cax = ax.matshow(corr.values, cmap='jet', vmin=-1, vmax=1)
        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(col_labels)))
        if self.show_corr_labels_var.get():
            ax.set_xticklabels(col_labels, rotation=45, ha='left', fontsize=8)
            ax.set_yticklabels(col_labels, fontsize=8)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        self._corr_fig.colorbar(cax, ax=ax, ticks=[-1, 0, 1], shrink=0.8)
        ax.set_title(f'{method.capitalize()} Correlation (Selection)', pad=20)
        self._corr_fig.tight_layout()

        corr_canvas = FigureCanvasTkAgg(self._corr_fig, master=self.plot_bottom_frame)
        corr_canvas.draw()
        corr_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_to_selection(self):
        """Add all highlighted Data Columns to the Selection list, keeping it sorted."""
        sel = self.column_listbox.curselection()
        if not sel:
            return
        changed = False
        for idx in sel:
            if idx not in self.selection_column_indices:
                self.selection_column_indices.append(idx)
                changed = True
        if changed:
            self.selection_column_indices.sort()
            self.selection_listbox.delete(0, tk.END)
            for idx in self.selection_column_indices:
                self.selection_listbox.insert(tk.END, self.column_listbox.get(idx))
            self._update_correlation_display()

    def _remove_from_selection(self):
        """Remove the highlighted entry from the Selection list."""
        sel = self.selection_listbox.curselection()
        if not sel:
            return
        list_idx = sel[0]
        self.selection_listbox.delete(list_idx)
        self.selection_column_indices.pop(list_idx)
        self._update_correlation_display()

    # Parameter specs for each peak method (used by the dialog and param cache)
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

    def _on_smoothing_toggle(self):
        """Enable/disable the points spinbox and redraw."""
        enabled = self.smoothing_var.get()
        if self.smoothing_points_spinbox:
            self.smoothing_points_spinbox.config(state='normal' if enabled else DISABLED)
        self._run_peak_on_column()

    def _smooth_signal(self, signal, col_idx):
        """Apply Convex Envelope smoothing when the checkbox is on."""
        if not self.smoothing_var.get():
            return signal
        try:
            n_points = self.smoothing_points_var.get()
        except tk.TclError:
            return signal
        from peak_functions import _convex_envelope_detrend_signal
        return _convex_envelope_detrend_signal(signal, n_points=n_points)

    def _get_data_for_peak(self, col_idx):
        """Return a data array with col_idx smoothed according to the current method."""
        if not self.smoothing_var.get():
            return self.loaded_data
        data = self.loaded_data.copy()
        data[:, col_idx] = self._smooth_signal(self.loaded_data[:, col_idx], col_idx)
        return data

    def _get_smoothing_points(self, col_idx):
        """Return the (x, y) lowest points used by Convex Envelope smoothing,
        in the original signal's scale, or None when smoothing is off or the
        "Show points" checkbox is unchecked."""
        if not self.smoothing_var.get() or not self.show_smoothing_points_var.get():
            return None
        try:
            n_points = self.smoothing_points_var.get()
        except tk.TclError:
            return None
        from peak_functions import convex_envelope_lowest_points
        return convex_envelope_lowest_points(self.loaded_data[:, col_idx], n_points=n_points)

    def _run_peak_on_column(self, show_dialog=False, event=None):
        """Redraw the current column: plot_top_frame always shows the original
        signal, plot_mid_frame always shows the processed signal (smoothed,
        and/or the peak-finder method's own output). Smoothing is applied to
        the column before the peak-finder method runs on it.
        show_dialog=True forces the parameter dialog (used when the method changes).
        show_dialog=False reuses cached params (used when the column changes)."""
        if self.loaded_data is None:
            return
        self._ensure_plot_frames()

        method = self.peak_method_var.get()
        col_idx = self.current_column

        if self._data_fig is not None:
            plt.close(self._data_fig)
            self._data_fig = None
        for w in list(self.plot_top_frame.winfo_children()):
            w.destroy()

        if self._mid_fig is not None:
            plt.close(self._mid_fig)
            self._mid_fig = None
        if self.plot_mid_frame and self.plot_mid_frame.winfo_exists():
            for w in list(self.plot_mid_frame.winfo_children()):
                w.destroy()

        saved_params = None
        if method != 'None':
            # Show dialog only when method changes or no params saved yet
            if show_dialog or method not in self.peak_method_params:
                from peak_functions import show_parameter_dialog
                spec = self._PEAK_PARAM_SPECS.get(method)
                if spec:
                    title, param_list = spec
                    new_params = show_parameter_dialog(self.root, title, param_list)
                    if new_params is None:
                        # User cancelled — revert to no peak-finder method
                        self.peak_method_var.set('None')
                        method = 'None'
                    else:
                        self.peak_method_params[method] = new_params
            if method != 'None':
                saved_params = self.peak_method_params.get(method)

        # Peaks are found on the already-smoothed column (if Smoothing is on),
        # computed once so both plots mark the exact same peaks.
        data_for_peak = None
        peaks = None
        if method != 'None' and saved_params is not None:
            from peak_functions import compute_peaks
            data_for_peak = self._get_data_for_peak(col_idx)
            peaks = compute_peaks(data_for_peak, col_idx, method, saved_params)

        self._draw_original_view(col_idx, peaks)
        self._draw_processed_view(col_idx, method, saved_params, data_for_peak, peaks)

    def _plot_smoothing_overlay(self, ax, col_idx, t):
        """Overlay the Convex Envelope baseline line and its lowest points onto ax."""
        points = self._get_smoothing_points(col_idx)
        if points is None:
            return
        px, py = points
        baseline = np.interp(t, px, py)
        ax.plot(t, baseline, color='darkorange', linewidth=1.3, linestyle='--', label='Baseline')
        ax.scatter(px, py, color='darkorange', s=25, zorder=5, label='Lowest points used')

    def _draw_original_view(self, col_idx, peaks):
        """Always plot the original signal in plot_top_frame, overlaying peak
        markers (if a peak-finder method is active) and the smoothing
        baseline/points (if Smoothing is enabled)."""
        col_label = self.column_listbox.get(col_idx)
        original = self.loaded_data[:, col_idx]
        t = np.arange(len(original))

        self._data_fig, ax = plt.subplots()
        ax.plot(t, original, color='steelblue', linewidth=0.8, label='Original')
        if peaks is not None and len(peaks) > 0:
            ax.scatter(peaks, original[peaks], color='crimson', s=25, zorder=5, label='Peaks')
        self._plot_smoothing_overlay(ax, col_idx, t)
        ax.set_title(col_label)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        if self.smoothing_var.get() or (peaks is not None and len(peaks) > 0):
            ax.legend(fontsize=8, loc='upper right')
        self._data_fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self._data_fig, master=self.plot_top_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _draw_processed_view(self, col_idx, method, params, data_for_peak, peaks):
        """Always plot the processed signal in plot_mid_frame: the smoothed
        signal alone, or the peak-finder method's own output run on the
        already-smoothed column when one is active."""
        if self.plot_mid_frame is None or not self.plot_mid_frame.winfo_exists():
            return

        if method == 'None':
            self._draw_smoothed_preview(col_idx)
            return

        from peak_functions import (elliptic_envelope_peak, actual_peak_caller,
                                    local_outlier_factor_peak, clf_peak,
                                    isolation_forest_peak, linear_model_peak, lasso_peak)
        method_map = {
            'Elliptic Envelope': elliptic_envelope_peak,
            'Peak Caller': actual_peak_caller,
            'Local Outlier Factor': local_outlier_factor_peak,
            'Peak Function 4': clf_peak,
            'Isolation Forest': isolation_forest_peak,
            'Linear Model': linear_model_peak,
            'Peak Function 7': lasso_peak,
        }
        func = method_map.get(method)
        if not func:
            return

        result = func(data_for_peak, col_idx,
                      main_window=None, canvas=None,
                      target_frame=self.plot_mid_frame,
                      params=params)
        if result is None:
            return
        mid_canvas, self._mid_fig = result
        col_label = self.column_listbox.get(col_idx)
        if self._mid_fig and self._mid_fig.axes:
            ax = self._mid_fig.axes[0]
            ax.set_title(col_label)

            reference = data_for_peak[:, col_idx]
            smoothing_on = self.smoothing_var.get()

            # "Elliptic Envelope" applies its own further internal detrend on
            # top of the data we hand it, so its own plot's line and outlier
            # markers are a vertically-shifted version of `reference`. Correct
            # them in place (rather than drawing a second, separate line)
            # so its plot matches every other method's plot consistently.
            if method == 'Elliptic Envelope':
                lines = ax.get_lines()
                if lines:
                    lines[0].set_ydata(reference)
                if len(lines) > 1:
                    marker_x = lines[1].get_xdata().astype(int)
                    lines[1].set_ydata(reference[marker_x])
                ax.relim()
                ax.autoscale_view()

            if peaks is not None and len(peaks) > 0:
                ax.scatter(peaks, reference[peaks], color='crimson', s=25, zorder=5, label='Peaks')
            if smoothing_on:
                points = self._get_smoothing_points(col_idx)
                if points is not None:
                    px, _ = points
                    px_int = px.astype(int)
                    ax.plot(px, reference[px_int], color='purple', linewidth=1.3, linestyle='--',
                            label='Smoothed lowest points')
                    ax.scatter(px, reference[px_int], color='purple', s=25, zorder=5)
            ax.legend(fontsize=8, loc='upper right')
            mid_canvas.draw()

    def _draw_smoothed_preview(self, col_idx):
        """Plot the smoothed (after) signal for col_idx in plot_mid_frame, so it's
        visible even when no peak-finder method is selected."""
        if self.plot_mid_frame is None or not self.plot_mid_frame.winfo_exists():
            return
        col_label = self.column_listbox.get(col_idx)
        smoothed = self._smooth_signal(self.loaded_data[:, col_idx], col_idx)
        t = np.arange(len(smoothed))

        self._mid_fig, ax = plt.subplots()
        ax.plot(t, smoothed, color='seagreen', linewidth=0.8, label='Smoothed')
        points = self._get_smoothing_points(col_idx)
        if points is not None:
            px, _ = points
            px_int = px.astype(int)
            ax.plot(px, smoothed[px_int], color='darkorange', linewidth=1.3, linestyle='--',
                    label='Baseline')
            ax.scatter(px, smoothed[px_int], color='darkorange', s=25, zorder=5,
                       label='Lowest points used (after smoothing)')
        ax.set_title(f'{col_label} — smoothed')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8, loc='upper right')
        self._mid_fig.tight_layout()

        c = FigureCanvasTkAgg(self._mid_fig, master=self.plot_mid_frame)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _save_data_image(self):
        """Save the current data / peak-finder plot to a file."""
        if self._data_fig is None:
            return
        from tkinter.filedialog import asksaveasfilename
        filename = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                       ("TIFF files", "*.tiff"), ("SVG files", "*.svg"),
                       ("EPS files", "*.eps"), ("All Files", "*.*")],
            title="Save Data Image"
        )
        if filename:
            self._data_fig.savefig(filename, dpi=300, bbox_inches='tight')

    def _save_correlation_image(self):
        """Save the current correlation heatmap to a file."""
        if self._corr_fig is None:
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
            self._corr_fig.savefig(filename, dpi=300, bbox_inches='tight')

    def open_visualization_data(self):
        """Abre datos para visualización de picos."""
        canvas = initialize_visualization(
            self.data_tab,
            self.menu_visual,
            self.canvas,
            self.column_listbox,
            self.update_column_display,
            self.notebook
        )
        if canvas is not None:
            self.canvas = canvas
            self.notebook.select(self.data_tab)
        if hasattr(self.root, 'loaded_data'):
            self.loaded_data = self.root.loaded_data
        if self.loaded_data is not None:
            self.btn_add_sel.configure(state='normal')
            self.btn_remove_sel.configure(state='normal')
            self.btn_save_data.configure(state='normal')
            self.btn_save_corr.configure(state='normal')
            self.btn_save_peaks.configure(state='normal')
            self.peak_method_combo.config(state='readonly')
            self.smoothing_check.configure(state='normal')
            self.show_smoothing_points_check.configure(state='normal')
            self.menu_visual.entryconfig(
                "Dendograma",
                command=self._run_dendogram_on_selection,
                state=NORMAL
            )
            # The initial column is drawn by the legacy single-plot helper in
            # visualization_helpers.py, which knows nothing about the
            # smoothing/peak-finder split layout. Re-render through the same
            # pipeline column clicks use, so the first column respects
            # whichever Smoothing/peak-finder options are already selected.
            self.current_column = 0
            self._run_peak_on_column()
    
    def _save_peaks_csv(self):
        """Run peak detection on every selection column and save peak indices to CSV."""
        method = self.peak_method_var.get()
        if method == 'None':
            messagebox.showwarning("No Peak Method", "Select a peak finder method first.")
            return
        if not self.selection_column_indices:
            messagebox.showwarning("No Selection", "Add columns to Selection first.")
            return
        params = self.peak_method_params.get(method)
        if params is None:
            messagebox.showwarning("No Parameters",
                                   "Run the peak finder on a column first to set parameters.")
            return

        from tkinter.filedialog import asksaveasfilename
        from peak_functions import compute_peaks
        import pandas as pd

        filename = asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All Files", "*.*")],
            title="Save Peaks CSV"
        )
        if not filename:
            return

        n_time = self.loaded_data.shape[0]
        data_dict = {'TIME': list(range(n_time))}
        for col_idx in self.selection_column_indices:
            col_name = self.column_listbox.get(col_idx)
            data_for_peak = self._get_data_for_peak(col_idx)
            peaks = compute_peaks(data_for_peak, col_idx, method, params)
            flags = np.zeros(n_time, dtype=int)
            flags[peaks] = 1
            data_dict[col_name] = flags

        pd.DataFrame(data_dict).to_csv(filename, index=False)
        messagebox.showinfo("Saved", f"Peak data saved to:\n{filename}")

    def _run_dendogram_on_selection(self):
        """Create the Dendogram tab on first use, then switch to it."""
        if self.dendo_tab is None or not self.dendo_tab.winfo_exists():
            self.dendo_tab = tk.Frame(self.notebook, bg=_C['bg'])
            self.notebook.add(self.dendo_tab, text="Dendograma")
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
            if self.loaded_data is not None:
                self.btn_dendo_add_sel.configure(state='normal')
                self.btn_dendo_remove_sel.configure(state='normal')
                self.btn_dendo_save_img.configure(state='normal')
                self.btn_dendo_save_csv.configure(state='normal')
        self.notebook.select(self.dendo_tab)

    # ==================== DENDOGRAM TAB ====================

    def _create_dendogram_layout(self):
        """Build the permanent Dendogram tab (sidebar + plot area)."""
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

        drow = _dsec("COLUMNAS DE DATOS", 0)

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

        drow = _dsec("SELECCIÓN", drow)

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

    def _dendo_populate_columns(self):
        """Fill the Dendogram tab column listbox with the same names as the main tab."""
        if self.dendo_column_listbox is None or self.loaded_data is None:
            return
        self.dendo_column_listbox.delete(0, tk.END)
        for i in range(self.column_listbox.size()):
            self.dendo_column_listbox.insert(tk.END, self.column_listbox.get(i))

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
        if idx < 0 or self.loaded_data is None:
            return
        self.dendo_current_column = idx
        self._dendo_draw_signal(idx)

    def _dendo_show_signal(self, event=None):
        """Draw the clicked column's signal into the top frame (keyboard nav only)."""
        if self._dendo_mouse_click:
            return
        if self.loaded_data is None or self.dendo_top_frame is None:
            return
        sel = self.dendo_column_listbox.curselection()
        if not sel:
            return
        col_idx = sel[-1]
        self.dendo_current_column = col_idx
        self._dendo_draw_signal(col_idx)

    def _dendo_draw_signal(self, col_idx):
        """Render the signal for col_idx into dendo_top_frame."""
        if self.loaded_data is None or self.dendo_top_frame is None:
            return

        if self.dendo_signal_fig is not None:
            plt.close(self.dendo_signal_fig)
            self.dendo_signal_fig = None
        for w in list(self.dendo_top_frame.winfo_children()):
            w.destroy()

        col_label = self.dendo_column_listbox.get(col_idx)
        self.dendo_signal_fig, ax = plt.subplots()
        ax.plot(np.array(range(len(self.loaded_data[:, col_idx]))).reshape(-1, 1),
                self.loaded_data[:, col_idx])
        ax.set_title(col_label)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        self.dendo_signal_fig.tight_layout()
        c = FigureCanvasTkAgg(self.dendo_signal_fig, master=self.dendo_top_frame)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _dendo_update_plot(self):
        """Render the dendrogram in the bottom frame once selection has 2+ columns."""
        if self.loaded_data is None or self.dendo_bottom_frame is None:
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

        plot_data = self.loaded_data[:, self.dendo_selection_indices]
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
        """Save dendrogram clustering data (labels + linkage matrix) to CSV."""
        if self.loaded_data is None:
            return
        from corr_dendo_functions import AgglomerativeClustering
        from tkinter.filedialog import asksaveasfilename
        import pandas as pd

        if len(self.dendo_selection_indices) >= 2:
            plot_data = self.loaded_data[:, self.dendo_selection_indices]
        else:
            plot_data = self.loaded_data

        filename = asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All Files", "*.*")],
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

        with open(filename, 'w', newline='') as f:
            f.write("# Cluster Labels\n")
            df_labels.to_csv(f, index=False)
            f.write("\n# Linkage Matrix\n")
            df_linkage.to_csv(f, index=False)

        messagebox.showinfo("Saved", f"Dendrogram data saved to:\n{filename}")

    def load_correlation_matrix_wrapper(self):
        """Wrapper para cargar matriz de correlación."""
        load_correlation_matrix(self.data_tab, self.canvas)

    # ==================== DATOS MULTIPLES (XLS) TAB ====================

    def open_multiple_xls_files(self):
        """Abre múltiples archivos .xls/.xlsx, deja elegir qué hojas cargar,
        y muestra sus columnas de datos en la pestaña 'Datos Multiples'."""
        selection = pick_files_and_sheets(self.root)
        if not selection:
            return

        datasets = self._load_xls_with_progress(selection)
        if not datasets:
            messagebox.showwarning("Sin datos", "No se pudo cargar ninguna hoja seleccionada.")
            return

        self.multi_xls_datasets = datasets

        if self.multi_xls_tab is None or not self.multi_xls_tab.winfo_exists():
            self.multi_xls_tab = tk.Frame(self.notebook, bg=_C['bg'])
            self.notebook.add(self.multi_xls_tab, text="  Datos Multiples  ")
            self._create_multi_xls_layout()

        self._populate_multi_xls_columns()
        self.notebook.select(self.multi_xls_tab)

    def _load_xls_with_progress(self, selection):
        """Carga las hojas seleccionadas mostrando una barra de progreso
        determinada por el número de hojas ya cargadas."""
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Cargando Archivos")
        progress_win.geometry("420x120")
        progress_win.transient(self.root)
        progress_win.grab_set()
        progress_win.resizable(False, False)
        progress_win.protocol("WM_DELETE_WINDOW", lambda: None)

        tk.Label(progress_win, text="Cargando hojas de Excel...",
                 font=('Arial', 11, 'bold')).pack(pady=(15, 4), padx=15, anchor='w')

        status_label = tk.Label(progress_win, text="Preparando...", font=('Arial', 9),
                                 fg=_C['sub'], anchor='w')
        status_label.pack(fill='x', padx=15, anchor='w')

        progress_bar = ttk.Progressbar(progress_win, orient='horizontal',
                                        length=390, mode='determinate',
                                        maximum=max(len(selection), 1))
        progress_bar.pack(pady=(8, 15), padx=15)

        progress_win.update_idletasks()

        def _on_progress(done, total, filepath, sheet):
            progress_bar['value'] = done
            status_label.config(
                text=f"{os.path.basename(filepath)} — {sheet}  ({done}/{total})"
            )
            progress_win.update_idletasks()

        try:
            datasets = load_selected_sheets(selection, progress_callback=_on_progress)
        finally:
            progress_win.destroy()

        return datasets

    def _create_multi_xls_layout(self):
        """Construye la pestaña 'Datos Multiples': lista de columnas (celdas)
        a la izquierda y la gráfica combinada, con scroll horizontal, a la derecha."""
        Grid.rowconfigure(self.multi_xls_tab, 0, weight=1)
        Grid.columnconfigure(self.multi_xls_tab, 0, weight=0)
        Grid.columnconfigure(self.multi_xls_tab, 1, weight=1)

        sidebar = tk.Frame(self.multi_xls_tab, bg=_C['panel'], width=250,
                           highlightbackground=_C['border'], highlightthickness=1)
        sidebar.grid(row=0, column=0, sticky='nsew', padx=(8, 0), pady=8)
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)

        tk.Label(sidebar, text="DATOS (CELDAS)", font=('Arial', 8, 'bold'),
                 bg=_C['panel'], fg=_C['sub']).grid(row=0, column=0, sticky='w',
                                                     padx=12, pady=(12, 2))
        tk.Frame(sidebar, bg=_C['border'], height=1).grid(row=1, column=0, sticky='ew', padx=10)

        lb_frame = tk.Frame(sidebar, bg=_C['card'],
                            highlightbackground=_C['border'], highlightthickness=1)
        lb_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=(6, 4))
        sidebar.rowconfigure(2, weight=1)
        lb_frame.rowconfigure(0, weight=1)
        lb_frame.columnconfigure(0, weight=1)

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

        tk.Checkbutton(sidebar, text="Mostrar Nombres de Datos",
                       variable=self.multi_xls_show_labels_var,
                       command=self._on_multi_xls_show_labels_toggle,
                       bg=_C['panel'], fg=_C['text'], selectcolor=_C['card'],
                       activebackground=_C['panel'], font=('Arial', 9)).grid(
            row=3, column=0, sticky='w', padx=10, pady=(0, 6))

        multi_smooth_frame = tk.Frame(sidebar, bg=_C['panel'])
        multi_smooth_frame.grid(row=4, column=0, sticky='ew', padx=10, pady=(0, 6))

        self.multi_xls_smoothing_check = tk.Checkbutton(
            multi_smooth_frame, text="Smoothing", variable=self.multi_xls_smoothing_var,
            command=self._on_multi_xls_smoothing_toggle,
            bg=_C['panel'], fg=_C['text'], selectcolor=_C['card'],
            activebackground=_C['panel'], font=('Arial', 9)
        )
        self.multi_xls_smoothing_check.grid(row=0, column=0, sticky='w')

        points_frame = tk.Frame(multi_smooth_frame, bg=_C['panel'])
        points_frame.grid(row=1, column=0, sticky='w')
        tk.Label(points_frame, text="Points:", bg=_C['panel'],
                 fg=_C['sub'], font=('Arial', 8)).pack(side='left', padx=(0, 2))
        self.multi_xls_smoothing_points_spinbox = ttk.Spinbox(
            points_frame, from_=2, to=50, textvariable=self.multi_xls_smoothing_points_var,
            width=4
        )
        self.multi_xls_smoothing_points_spinbox.pack(side='left')
        self.multi_xls_smoothing_points_spinbox.config(
            state='normal' if self.multi_xls_smoothing_var.get() else DISABLED)
        self.multi_xls_smoothing_points_var.trace_add(
            'write', lambda *args: self._on_multi_xls_smoothing_toggle())

        tk.Checkbutton(sidebar, text="Escala de Color Compartida (mapa de calor)",
                       variable=self.multi_xls_shared_scale_var,
                       command=self._on_multi_xls_shared_scale_toggle,
                       bg=_C['panel'], fg=_C['text'], selectcolor=_C['card'],
                       activebackground=_C['panel'], font=('Arial', 9),
                       wraplength=210, justify='left').grid(
            row=5, column=0, sticky='w', padx=10, pady=(0, 6))

        self.btn_save_multi_xls_plot = ctk.CTkButton(
            sidebar, text="Save Plot Image", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            state='disabled', command=self._save_multi_xls_plot_image
        )
        self.btn_save_multi_xls_plot.grid(row=6, column=0, sticky='ew', padx=10, pady=(0, 2))

        self.btn_save_multi_xls_heatmap = ctk.CTkButton(
            sidebar, text="Save Heatmap Image", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            state='disabled', command=self._save_multi_xls_heatmap_image
        )
        self.btn_save_multi_xls_heatmap.grid(row=7, column=0, sticky='ew', padx=10, pady=(0, 2))

        ctk.CTkButton(
            sidebar, text="Editar Clasificaciones", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            command=self._open_multi_xls_class_editor
        ).grid(row=8, column=0, sticky='ew', padx=10, pady=(0, 2))

        self.btn_save_multi_xls_classifications = ctk.CTkButton(
            sidebar, text="Save Classifications", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            state='disabled', command=self._save_multi_xls_classifications
        )
        self.btn_save_multi_xls_classifications.grid(row=9, column=0, sticky='ew', padx=10, pady=(0, 2))

        self.btn_load_multi_xls_classifications = ctk.CTkButton(
            sidebar, text="Load Classifications", height=28, corner_radius=6,
            fg_color=_C['card'], hover_color=_C['border'], text_color=_C['text'],
            border_width=1, border_color=_C['border'], font=ctk.CTkFont(size=11),
            state='disabled', command=self._load_multi_xls_classifications
        )
        self.btn_load_multi_xls_classifications.grid(row=10, column=0, sticky='ew', padx=10, pady=(0, 10))

        # Right side - stacked plot areas (line plot on top, heatmap below).
        # Both are redrawn to exactly fill their frame on every resize, so
        # the whole graph and every sheet image are always visible without
        # needing to scroll.
        right_container = tk.Frame(self.multi_xls_tab, bg=_C['bg'])
        right_container.grid(row=0, column=1, sticky='nsew', padx=8, pady=8)
        right_container.rowconfigure(0, weight=3)
        right_container.rowconfigure(1, weight=2)
        right_container.columnconfigure(0, weight=1)

        self.multi_xls_plot_frame = tk.Frame(right_container, bg=_C['panel'],
                                              highlightbackground=_C['border'],
                                              highlightthickness=1)
        self.multi_xls_plot_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 4))
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
            self.multi_xls_plot_frame, text="Selecciona una columna para graficar",
            font=('Arial', 14), bg=_C['panel'], fg=_C['sub'])
        self.multi_xls_plot_placeholder.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Bottom: heatmap images (one per sheet, columns=datos, rows=muestras)
        self.multi_xls_heatmap_frame = tk.Frame(right_container, bg=_C['panel'],
                                                 highlightbackground=_C['border'],
                                                 highlightthickness=1)
        self.multi_xls_heatmap_frame.grid(row=1, column=0, sticky='nsew', pady=(4, 0))
        self.multi_xls_heatmap_frame.bind('<Configure>', self._on_multi_xls_heatmap_frame_resize)

        self.multi_xls_heatmap_placeholder = tk.Label(
            self.multi_xls_heatmap_frame, text="Las imágenes de las hojas aparecerán aquí",
            font=('Arial', 14), bg=_C['panel'], fg=_C['sub'])
        self.multi_xls_heatmap_placeholder.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def _populate_multi_xls_columns(self):
        """Llena la lista lateral con nombres genéricos 'Column N' (uno por
        cada columna común a todas las hojas cargadas) y grafica la primera
        por defecto. Estos nombres no cambian con el checkbox 'Mostrar
        Nombres de Datos', que solo afecta a las etiquetas de la gráfica."""
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
                "Sin columnas comunes",
                "Las hojas seleccionadas no comparten ninguna columna de datos con el mismo nombre."
            )

        self._draw_multi_xls_heatmap()

        if self.multi_xls_datasets:
            self.btn_save_multi_xls_plot.configure(state='normal')
            self.btn_save_multi_xls_heatmap.configure(state='normal')
            self.btn_multi_xls_next_column.configure(state='normal')
            self.btn_save_multi_xls_classifications.configure(state='normal')
            self.btn_load_multi_xls_classifications.configure(state='normal')

    def _rebuild_multi_xls_class_row(self):
        """Recrea el combobox de clasificación de cada hoja cargada (uno por
        hoja). Las opciones por defecto son los nombres de todas las hojas
        cargadas; cada hoja empieza clasificada con su propio nombre. Se
        posicionan en el siguiente redibujado de la gráfica de líneas. La
        clasificación elegida se guarda por separado para cada columna de
        datos (self.multi_xls_classifications), así que cambiar de columna
        en la lista actualiza los combobox con lo que se eligió para esa
        columna en particular."""
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
        """Guarda la clasificación elegida para 'sheet_label' bajo la
        columna actualmente seleccionada."""
        if self._multi_xls_syncing_class_row or self.multi_xls_current_index is None:
            return
        value = self.multi_xls_sheet_class_var[sheet_label].get()
        self.multi_xls_classifications.setdefault(self.multi_xls_current_index, {})[sheet_label] = value

    def _sync_multi_xls_class_row_values(self, index):
        """Actualiza los combobox para mostrar la clasificación guardada de
        cada hoja para la columna 'index' (o el nombre de la hoja como
        valor por defecto si esa combinación columna+hoja aún no se ha
        clasificado)."""
        saved = self.multi_xls_classifications.get(index, {})
        self._multi_xls_syncing_class_row = True
        try:
            for label, var in self.multi_xls_sheet_class_var.items():
                var.set(saved.get(label, label))
        finally:
            self._multi_xls_syncing_class_row = False

    def _position_multi_xls_class_row(self, ax, bounds):
        """Coloca cada combobox de clasificación alineado en pixeles con el
        segmento (offset_inicio, offset_fin) de su hoja en la gráfica 'ax'
        recién dibujada."""
        if self.multi_xls_class_row is None:
            return
        placed = set()
        for ds, (x0_data, x1_data) in zip(self.multi_xls_datasets, bounds):
            label = ds['label']
            combo = self.multi_xls_class_combos.get(label)
            if combo is None:
                continue
            x0_px = ax.transData.transform((x0_data, 0))[0]
            x1_px = ax.transData.transform((x1_data, 0))[0]
            width_px = max(x1_px - x0_px, 30)
            combo.place(x=x0_px, y=2, width=width_px - 2, height=26)
            placed.add(label)

        for label, combo in self.multi_xls_class_combos.items():
            if label not in placed:
                combo.place_forget()

    def _open_multi_xls_class_editor(self):
        """Diálogo para renombrar, eliminar o agregar opciones de
        clasificación. Al aplicar, actualiza todos los combobox y conserva
        (o reasigna) la clasificación ya elegida en cada hoja."""
        if not self.multi_xls_classes:
            messagebox.showinfo("Sin datos", "Carga archivos .xls primero.")
            return

        import uuid as _uuid

        dialog = tk.Toplevel(self.root)
        dialog.title("Editar Clasificaciones")
        dialog.geometry("360x440")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Opciones de clasificación:", font=('Arial', 11, 'bold')).pack(
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
                messagebox.showwarning("Duplicado", "Esa opción ya existe.", parent=dialog)
                return
            entries.append({'id': str(_uuid.uuid4()), 'name': name})
            listbox.insert(tk.END, name)

        def rename_option():
            sel = listbox.curselection()
            if not sel:
                messagebox.showinfo("Sin selección", "Selecciona una opción para renombrar.", parent=dialog)
                return
            name = edit_var.get().strip()
            if not name:
                return
            idx = sel[0]
            if any(i != idx and e['name'] == name for i, e in enumerate(entries)):
                messagebox.showwarning("Duplicado", "Esa opción ya existe.", parent=dialog)
                return
            entries[idx]['name'] = name
            listbox.delete(idx)
            listbox.insert(idx, name)
            listbox.selection_set(idx)

        def delete_option():
            sel = listbox.curselection()
            if not sel:
                messagebox.showinfo("Sin selección", "Selecciona una opción para eliminar.", parent=dialog)
                return
            if len(entries) <= 1:
                messagebox.showwarning("No permitido", "Debe quedar al menos una opción.", parent=dialog)
                return
            idx = sel[0]
            del entries[idx]
            listbox.delete(idx)
            edit_var.set('')

        btn_row = tk.Frame(dialog)
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 6))
        tk.Button(btn_row, text="Agregar", command=add_option).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Renombrar", command=rename_option).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(btn_row, text="Eliminar", command=delete_option).pack(side=tk.LEFT, padx=(6, 0))

        def on_apply():
            self._apply_multi_xls_class_changes(original_entries, entries)
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ok_row = tk.Frame(dialog)
        ok_row.pack(pady=10)
        tk.Button(ok_row, text="Aplicar", command=on_apply, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(ok_row, text="Cancelar", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)

    def _apply_multi_xls_class_changes(self, original_entries, final_entries):
        """Aplica los cambios del editor de clasificaciones: propaga
        renombrados a las hojas que tenían esa opción elegida, reasigna las
        hojas que tenían una opción eliminada, y refresca las opciones
        disponibles en todos los combobox."""
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

    def _on_multi_xls_show_labels_toggle(self):
        """Redibuja ambas gráficas para mostrar u ocultar las etiquetas de
        cada hoja bajo el eje X, sin tocar los nombres de la lista de datos."""
        if self.multi_xls_current_index is not None:
            self._draw_multi_xls_plot(self.multi_xls_current_index)
        self._draw_multi_xls_heatmap()

    def _on_multi_xls_smoothing_toggle(self):
        """Enable/disable the points spinbox and redraw the top plot with
        Convex Envelope smoothing applied (or removed)."""
        enabled = self.multi_xls_smoothing_var.get()
        if self.multi_xls_smoothing_points_spinbox:
            self.multi_xls_smoothing_points_spinbox.config(state='normal' if enabled else DISABLED)
        if self.multi_xls_current_index is not None:
            self._draw_multi_xls_plot(self.multi_xls_current_index)

    def _on_multi_xls_shared_scale_toggle(self):
        """Redibuja el mapa de calor con una escala de color compartida entre
        todas las hojas, o una escala independiente para cada hoja."""
        self._draw_multi_xls_heatmap()

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

    def _on_multi_xls_column_select(self, event=None):
        """Callback cuando el usuario elige una columna en la lista lateral."""
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
        """Redibuja la gráfica de líneas (con un pequeño retraso, para no
        redibujar en cada pixel mientras se arrastra la ventana) para que
        siga llenando exactamente el panel tras cambiar de tamaño."""
        if self._multi_xls_plot_resize_job is not None:
            self.root.after_cancel(self._multi_xls_plot_resize_job)
        self._multi_xls_plot_resize_job = self.root.after(200, self._redraw_multi_xls_plot_for_resize)

    def _redraw_multi_xls_plot_for_resize(self):
        self._multi_xls_plot_resize_job = None
        if self.multi_xls_current_index is not None:
            self._draw_multi_xls_plot(self.multi_xls_current_index)

    def _on_multi_xls_heatmap_frame_resize(self, event=None):
        """Redibuja el panel de imágenes (con un pequeño retraso) para que
        siga llenando exactamente el panel tras cambiar de tamaño."""
        if self._multi_xls_heatmap_resize_job is not None:
            self.root.after_cancel(self._multi_xls_heatmap_resize_job)
        self._multi_xls_heatmap_resize_job = self.root.after(200, self._draw_multi_xls_heatmap)

    def _draw_multi_xls_plot(self, index):
        """Grafica la columna en la posición 'index' de cada hoja cargada,
        una junto a la otra en la misma figura, separadas por líneas
        verticales punteadas. El título de la gráfica usa la misma etiqueta
        que está seleccionada en la lista de datos, y las etiquetas de cada
        hoja se muestran u ocultan según el checkbox 'Mostrar Nombres de
        Datos'. El canvas se crea una sola vez y se reutiliza en cada
        redibujado (ver comentario en __init__), ajustándose exactamente al
        tamaño del panel para que se vea completo sin hacer scroll."""
        if not self.multi_xls_datasets or self.multi_xls_plot_frame is None:
            return
        if index >= len(self.multi_xls_common_columns):
            return

        frame = self.multi_xls_plot_frame
        frame.update_idletasks()
        w_px, h_px = frame.winfo_width(), frame.winfo_height()
        if w_px <= 1 or h_px <= 1:
            # El panel todavía no está visible (p.ej. la pestaña se acaba de
            # crear). Recordamos qué columna dibujar; el <Configure> que se
            # dispare cuando el panel obtenga su tamaño real hará el dibujo.
            self.multi_xls_current_index = index
            self.multi_xls_current_column = self.multi_xls_common_columns[index]
            return

        import pandas as pd

        col_name = self.multi_xls_common_columns[index]
        display_label = self.multi_xls_column_listbox.get(index)
        show_labels = self.multi_xls_show_labels_var.get()

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
            # forward=False: solo actualiza el tamaño interno de la figura,
            # sin ordenarle a Tk que redimensione el widget del canvas.
            self.multi_xls_fig.set_size_inches(fig_width, fig_height, forward=False)

        ax = self._multi_xls_plot_ax
        offset = 0
        tick_positions = []
        tick_labels = []
        bounds = []
        for ds in self.multi_xls_datasets:
            if col_name not in ds['df'].columns:
                continue
            values = ds['df'][col_name].to_numpy(dtype=float)
            values = pd.Series(values).interpolate(limit_direction='both').to_numpy()
            n = len(values)
            if n == 0:
                continue
            values = self._smooth_multi_xls_signal(values)
            x = np.arange(offset, offset + n)
            ax.plot(x, values, linewidth=0.8)
            if offset > 0:
                ax.axvline(offset, color=_C['sub'], linestyle='--', linewidth=1, alpha=0.6)
            tick_positions.append(offset + n / 2)
            tick_labels.append(ds['label'])
            bounds.append((offset, offset + n))
            offset += n

        ax.set_xticks(tick_positions)
        if show_labels:
            ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.set_title(display_label)
        ax.set_ylabel('Value')
        self.multi_xls_fig.tight_layout()

        # Force the same horizontal bounds used by the heatmap below, so a
        # sample's x-position lines up vertically between the two plots
        # regardless of the heatmap's colorbar taking extra space on its side.
        pos = ax.get_position()
        ax.set_position([_MULTI_XLS_AX_LEFT, pos.y0,
                          _MULTI_XLS_AX_RIGHT - _MULTI_XLS_AX_LEFT, pos.height])

        self._multi_xls_plot_canvas.draw()

        self._position_multi_xls_class_row(ax, bounds)
        self.btn_multi_xls_next_column.lift()

    def _draw_multi_xls_heatmap(self):
        """Dibuja, para cada hoja cargada, una imagen donde el eje X son las
        muestras (el mismo eje que la gráfica de líneas de arriba) y el eje Y
        son las columnas (las series de datos); el color de cada pixel
        representa su valor dividido entre el mínimo propio de esa hoja (cada
        hoja se normaliza contra su propio mínimo, no uno global). Todas las
        hojas comparten la misma escala de color. Las imágenes de cada hoja
        se colocan una junto a otra. El canvas se crea una sola vez y se
        reutiliza en cada redibujado (ver comentario en __init__), ajustándose
        exactamente al tamaño del panel."""
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
                self.multi_xls_heatmap_frame, text="Las imágenes de las hojas aparecerán aquí",
                font=('Arial', 14), bg=_C['panel'], fg=_C['sub'])
            self.multi_xls_heatmap_placeholder.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            return

        frame = self.multi_xls_heatmap_frame
        frame.update_idletasks()
        w_px, h_px = frame.winfo_width(), frame.winfo_height()
        if w_px <= 1 or h_px <= 1:
            return

        raw_matrices = [ds['df'].to_numpy(dtype=float) for ds in self.multi_xls_datasets]
        finite_vals = np.concatenate([m[np.isfinite(m)].ravel() for m in raw_matrices])
        if finite_vals.size == 0:
            return

        # Normalize each sheet's values against its own minimum (not one
        # global minimum shared across all sheets), then transpose so rows =
        # data columns and columns = samples.
        matrices = []
        for m in raw_matrices:
            sheet_min = float(m[np.isfinite(m)].min())
            matrices.append((m / sheet_min).T)

        shared_scale = self.multi_xls_shared_scale_var.get()
        if shared_scale:
            norm_finite = np.concatenate([m[np.isfinite(m)].ravel() for m in matrices])
            vmin, vmax = float(norm_finite.min()), float(norm_finite.max())

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
            # forward=False: solo actualiza el tamaño interno de la figura,
            # sin ordenarle a Tk que redimensione el widget del canvas (eso
            # generaba un <Configure> nuevo en cada redibujado y provocaba
            # un ciclo de redibujado infinito).
            self.multi_xls_heatmap_fig.set_size_inches(fig_width, fig_height, forward=False)

        ax = self._multi_xls_heatmap_ax
        offset = 0
        tick_positions = []
        tick_labels = []
        im = None
        for ds, matrix in zip(self.multi_xls_datasets, matrices):
            n_cols, n_samples = matrix.shape
            if shared_scale:
                im_vmin, im_vmax = vmin, vmax
            else:
                finite = matrix[np.isfinite(matrix)]
                im_vmin, im_vmax = float(finite.min()), float(finite.max())
            im = ax.imshow(matrix, aspect='auto', cmap='jet', vmin=im_vmin, vmax=im_vmax,
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
        if shared_scale:
            ax.set_title('Todas las hojas (valor / mínimo de cada hoja)')
        else:
            ax.set_title('Todas las hojas (valor / mínimo de cada hoja, escala individual por hoja)')
        self.multi_xls_heatmap_fig.tight_layout()

        # Force the same horizontal bounds used by the top line plot, so a
        # sample's x-position lines up vertically between the two. The
        # colorbar (when shown) goes in its own explicit axes to the right
        # of this fixed span, instead of letting fig.colorbar() shrink ax
        # to make room for it.
        pos = ax.get_position()
        ax.set_position([_MULTI_XLS_AX_LEFT, pos.y0,
                          _MULTI_XLS_AX_RIGHT - _MULTI_XLS_AX_LEFT, pos.height])
        if shared_scale and im is not None:
            cax = self.multi_xls_heatmap_fig.add_axes(
                [_MULTI_XLS_AX_RIGHT + 0.02, pos.y0, 0.02, pos.height])
            self._multi_xls_heatmap_colorbar = self.multi_xls_heatmap_fig.colorbar(im, cax=cax)

        self._multi_xls_heatmap_canvas.draw()

    def _save_multi_xls_plot_image(self):
        """Guarda la gráfica de líneas combinada de 'Datos Multiples' en un archivo."""
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
        """Guarda la imagen del heatmap de 'Datos Multiples' en un archivo."""
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

    def _save_multi_xls_classifications(self):
        """Guarda, para cada columna de datos y cada hoja cargada, la
        clasificación elegida (una fila por columna, una columna por hoja),
        en un archivo .xlsx o .csv. Las combinaciones no clasificadas
        todavía usan el mismo valor por defecto (el nombre de la hoja) que
        se muestra en la interfaz."""
        if not self.multi_xls_datasets or not self.multi_xls_common_columns:
            messagebox.showwarning("Sin Datos", "Carga datos primero.")
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
            df.to_excel(filename)

        messagebox.showinfo("Saved", f"Classifications saved to:\n{filename}")

    def _load_multi_xls_classifications(self):
        """Carga clasificaciones previamente guardadas (mismo formato que
        "Save Classifications": una fila por columna de datos, una columna
        por hoja) y las aplica a las columnas/hojas que coincidan por
        nombre con lo actualmente cargado. Cualquier valor de clasificación
        nuevo se agrega a las opciones disponibles en los combobox."""
        if not self.multi_xls_datasets or not self.multi_xls_common_columns:
            messagebox.showwarning("Sin Datos", "Carga datos primero.")
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
                "Sin Coincidencias",
                "Ninguna columna u hoja del archivo coincide con los datos cargados."
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
        """Carga una imagen por defecto o muestra placeholder."""
        # Crear imagen placeholder
        pil_img = Image.new('RGB', (400, 300), color='#1e1e1e')
        self._display_pil_image(pil_img)
    
    def _display_pil_image(self, pil_img):
        """Muestra una imagen PIL en el label principal."""
        # Obtener tamaño disponible del frame
        self.image_frame.update_idletasks()
        frame_width = self.image_frame.winfo_width() - 20
        frame_height = self.image_frame.winfo_height() - 20
        
        if frame_width < 100:
            frame_width = int(self.width * 0.7)
        if frame_height < 100:
            frame_height = int(self.height * 0.9)
        
        # Redimensionar manteniendo proporción
        width_pil, height_pil = pil_img.size
        ratio = min(frame_width / width_pil, frame_height / height_pil)
        new_size = (int(width_pil * ratio), int(height_pil * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
        
        # Convertir y mostrar
        image_tk = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk  # Mantener referencia
    
    def _apply_brightness_contrast(self, image_array):
        """Aplica brillo y contraste a la imagen."""
        brightness = self.brightness_slider.get()
        contrast = self.contrast_slider.get()
        
        # Convertir a float para operaciones
        img = image_array.astype(np.float32)
        
        # Aplicar brillo
        img = img + brightness
        
        # Aplicar contraste
        if contrast != 0:
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            img = factor * (img - 128) + 128
        
        # Clamp valores
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    def _update_image_display(self):
        """Actualiza la imagen mostrada según el estado actual."""
        if self.img_array is None:
            return
        
        slice_idx = self.slice_slider.get()
        current_slice = self.img_array[slice_idx, :, :].copy()
        
        # Aplicar brillo y contraste
        current_slice = self._apply_brightness_contrast(current_slice)
        
        # Aplicar threshold si está habilitado
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
        
        # Actualizar info de frame
        total_frames = self.img_array.shape[0]
        self.frame_info_label.config(text=f"Frame: {slice_idx + 1} / {total_frames}")
    
    def _reset_adjustments(self):
        """Resetea los ajustes de brillo y contraste."""
        self.brightness_slider.set(0)
        self.contrast_slider.set(0)
        self._update_image_display()
    
    # ==================== CALLBACKS ====================
    
    def _on_slice_changed(self, val):
        """Callback cuando cambia el slider de capa."""
        self._update_image_display()
    
    def _on_threshold_changed(self, val):
        """Callback cuando cambia el slider de threshold."""
        if self.threshold_enabled.get():
            self._update_image_display()
    
    def _on_adjustment_changed(self, val):
        """Callback cuando cambian los ajustes de brillo/contraste."""
        self._update_image_display()
    
    # ==================== COMANDOS DE MENÚ - ARCHIVO ====================
    
    def open_ometiff_file(self):
        """Abre un archivo OME-TIFF."""
        if HAS_IMAGE_MODULES:
            img, metadata, xml_metadata = load_ometiff_image()
        else:
            filename = filedialog.askopenfilename(
                title="Abrir OME-TIFF",
                filetypes=[("OME-TIFF files", "*.ome.tiff *.ome.tif"), ("All files", "*.*")]
            )
            if not filename:
                return
            
            try:
                reader = OMETIFFReader(fpath=filename)
                img, metadata, xml_metadata = reader.read()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{str(e)}")
                return
        
        if img is None:
            return
        
        self.img_original = img
        self.img_array = img.copy()
        
        # Actualizar slider de capas
        num_slices = self.img_array.shape[0]
        self.slice_slider.configure(to=num_slices - 1)
        self.slice_slider.set(0)
        
        # Resetear ajustes
        self._reset_adjustments()
        
        # Actualizar información
        shape = self.img_array.shape
        info_text = f"Dimensiones: {shape[2]}x{shape[1]}\nFrames: {shape[0]}\nTipo: {self.img_array.dtype}"
        self.info_text.config(text=info_text)
        
        # Enable image menu items now that an image is loaded
        self.menu_imagen.entryconfig("Auto Contraste", state=NORMAL)
        self.menu_imagen.entryconfig("Histogram", state=NORMAL)
        self.menu_imagen.entryconfig("Binarize", state=NORMAL)
        self.menu_imagen.entryconfig("Restaurar Original", state=NORMAL)
        self.menu_bar.entryconfig("Análisis de Variabilidad", state=NORMAL)

        # Cambiar a la pestaña de imagen
        self.notebook.select(self.image_tab)

        self._update_image_display()
    
    # ==================== COMANDOS DE MENÚ - IMAGEN ====================
    
    def restore_original(self):
        """Restaura la imagen original."""
        if self.img_original is None:
            messagebox.showwarning("Advertencia", "No hay imagen original cargada")
            return
        self.img_array = self.img_original.copy()
        self._reset_adjustments()
    
    def apply_auto_contrast(self):
        """Aplica auto contraste a la imagen."""
        if self.img_array is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen OME-TIFF")
            return
        
        if HAS_IMAGE_MODULES:
            # Usar módulo de procesamiento si está disponible
            self.img_array = auto_contrast(self.img_array)
        else:
            # Fallback: aplicar auto contraste frame por frame
            for i in range(self.img_array.shape[0]):
                im_pil = Image.fromarray(self.img_array[i, :, :])
                if im_pil.mode != 'RGB':
                    im_pil = im_pil.convert('RGB')
                im2 = ImageOps.autocontrast(im_pil, cutoff=2, ignore=2).convert('L')
                self.img_array[i, :, :] = np.array(im2)
        
        self._update_image_display()
    
    def show_histogram(self):
        """Muestra el histograma de la imagen."""
        if self.img_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen OME-TIFF")
            return
        
        var_im = np.var(self.img_original, axis=0)
        plt.figure(figsize=(8, 6))
        plt.hist(var_im.flatten(), bins=50)
        plt.title('Histograma de Varianza')
        plt.xlabel('Varianza')
        plt.ylabel('Frecuencia')
        plt.show()
    
    def show_binarize(self):
        """Muestra ventana de binarización."""
        if self.img_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen OME-TIFF")
            return
        
        var_im = np.var(self.img_original, axis=0)
        pil_img = self._apply_binarize(var_im, 150)
        
        # Crear ventana emergente
        top = tk.Toplevel(self.root)
        top.title("Binarización")
        top.geometry("600x700")
        
        # Frame para la imagen
        img_frame = tk.Frame(top, relief=tk.RAISED, borderwidth=1)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        image_ = ImageTk.PhotoImage(pil_img)
        label = tk.Label(img_frame, image=image_)
        label.image = image_
        label.pack(fill=tk.BOTH, expand=True)
        
        # Frame para controles
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
                messagebox.showerror("Error", "Por favor ingresa un número válido")
        
        tk.Button(
            control_frame,
            text="Aplicar Binarización",
            command=update_binarization
        ).pack(side=tk.LEFT, padx=5)
    
    def _apply_binarize(self, var_im, threshold):
        """Aplica binarización a la imagen basado en el threshold."""
        if len(var_im.shape) == 2:
            # Es una imagen 2D (varianza)
            binary = (var_im > threshold).astype(np.uint8) * 255
        else:
            # Es una imagen normal
            binary = (var_im > threshold).astype(np.uint8) * 255
        
        pil_img = Image.fromarray(binary)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        return pil_img
    
    def show_variability_menu(self, method_index):
        """Mostrar análisis de variabilidad completo."""
        if self.img_array is None or len(self.img_array) == 0:
            messagebox.showwarning("Advertencia", "Primero carga una imagen OME-TIFF")
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
            "NecLab — Análisis de Imágenes de Microscopía y Visualización de Datos\n\n"
            f"Repository: github.com/{self._REPO}\n"
            "Contact: sergio.cruz@ciencias.unam.mx"
        )


def main():
    """Función principal."""
    root = tk.Tk()
    app = NecLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
