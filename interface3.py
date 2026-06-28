"""
NecLab - Herramienta de análisis de imágenes de microscopía y visualización de datos
Interfaz gráfica principal (versión unificada)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # Limita número de threads

import tkinter as tk
from tkinter import Menu, Grid, filedialog, FALSE, DISABLED, NORMAL, ttk, messagebox
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

# Módulos locales
from pyometiff import OMETIFFReader
from visualization_helpers import initialize_visualization
from variability_functions import show_variability_analysis, get_variability_methods
from corr_dendo_functions import load_correlation_matrix

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
        self.btn_save_data = None
        self.btn_save_corr = None
        self.btn_save_peaks = None
        self.peak_method_combo = None
        self.peak_method_params = {}  # saved params per method name

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
        # Configurar grid principal
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Crear notebook (tabs) para cambiar entre modos
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Tab 1: Procesamiento de Imágenes
        self.image_tab = tk.Frame(self.notebook, bg='#f0f0f0')
        self.notebook.add(self.image_tab, text="Procesamiento de Imágenes")
        self._create_image_processing_layout()
        
        # Tab 2: Visualización de Datos
        self.data_tab = tk.Frame(self.notebook, bg='#f0f0f0')
        self.notebook.add(self.data_tab, text="Visualización de Datos")
        self._create_data_visualization_layout()

    
    def _create_image_processing_layout(self):
        """Crea el layout para procesamiento de imágenes."""
        # Configurar grid
        self.image_tab.columnconfigure(0, weight=3)
        self.image_tab.columnconfigure(1, weight=1)
        self.image_tab.rowconfigure(0, weight=1)
        
        # Frame izquierdo para la imagen
        self.image_frame = tk.Frame(self.image_tab, bg='#1e1e1e', relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Label para mostrar la imagen
        self.image_label = tk.Label(self.image_frame, bg='#1e1e1e')
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Panel de controles
        self._create_image_controls_panel()
        
        # Cargar imagen por defecto o placeholder
        self._load_default_image()
    
    def _create_image_controls_panel(self):
        """Crea el panel lateral derecho con controles de imagen."""
        # Frame principal del panel
        self.controls_panel = tk.Frame(self.image_tab, bg='#f0f0f0', relief=tk.RAISED, borderwidth=2)
        self.controls_panel.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        
        # Título del panel
        title_label = tk.Label(
            self.controls_panel,
            text="Controles de Imagen",
            font=('Arial', 14, 'bold'),
            bg='#f0f0f0'
        )
        title_label.pack(pady=10)
        
        # Separador
        ttk.Separator(self.controls_panel, orient='horizontal').pack(fill='x', padx=10)
        
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
        section = tk.LabelFrame(
            self.controls_panel,
            text="Navegación",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0',
            padx=10,
            pady=5
        )
        section.pack(fill='x', padx=10, pady=10)
        
        # Slider de capa
        tk.Label(section, text="Capa (Frame):", bg='#f0f0f0').pack(anchor='w')
        
        slider_frame = tk.Frame(section, bg='#f0f0f0')
        slider_frame.pack(fill='x')
        
        self.slice_slider = tk.Scale(
            slider_frame,
            from_=0,
            to=0,
            orient="horizontal",
            command=self._on_slice_changed,
            bg='#f0f0f0',
            highlightthickness=0
        )
        self.slice_slider.pack(fill='x', expand=True)
        
        # Label para mostrar frame actual / total
        self.frame_info_label = tk.Label(
            section,
            text="Frame: 0 / 0",
            bg='#f0f0f0',
            font=('Arial', 9)
        )
        self.frame_info_label.pack(anchor='w')
    
    def _create_section_image_adjustments(self):
        """Sección de ajustes de imagen."""
        section = tk.LabelFrame(
            self.controls_panel,
            text="Ajustes de Imagen",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0',
            padx=10,
            pady=5
        )
        section.pack(fill='x', padx=10, pady=10)
        
        # Brillo
        tk.Label(section, text="Brillo:", bg='#f0f0f0').pack(anchor='w')
        self.brightness_slider = tk.Scale(
            section,
            from_=-100,
            to=100,
            orient="horizontal",
            command=self._on_adjustment_changed,
            bg='#f0f0f0',
            highlightthickness=0
        )
        self.brightness_slider.set(0)
        self.brightness_slider.pack(fill='x')
        
        # Contraste
        tk.Label(section, text="Contraste:", bg='#f0f0f0').pack(anchor='w', pady=(5,0))
        self.contrast_slider = tk.Scale(
            section,
            from_=-100,
            to=100,
            orient="horizontal",
            command=self._on_adjustment_changed,
            bg='#f0f0f0',
            highlightthickness=0
        )
        self.contrast_slider.set(0)
        self.contrast_slider.pack(fill='x')
        
        # Botón Auto Contraste
        tk.Button(
            section,
            text="Auto Contraste",
            command=self.apply_auto_contrast
        ).pack(fill='x', pady=5)
        
        # Botón Reset
        tk.Button(
            section,
            text="Resetear Ajustes",
            command=self._reset_adjustments
        ).pack(fill='x')
    
    def _create_section_processing(self):
        """Sección de procesamiento."""
        section = tk.LabelFrame(
            self.controls_panel,
            text="Procesamiento",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0',
            padx=10,
            pady=5
        )
        section.pack(fill='x', padx=10, pady=10)
        
        # Threshold
        threshold_frame = tk.Frame(section, bg='#f0f0f0')
        threshold_frame.pack(fill='x')
        
        tk.Label(threshold_frame, text="Threshold:", bg='#f0f0f0').pack(side='left')
        
        self.threshold_enabled = tk.BooleanVar(value=False)
        self.threshold_check = tk.Checkbutton(
            threshold_frame,
            text="Aplicar",
            variable=self.threshold_enabled,
            command=self._update_image_display,
            bg='#f0f0f0'
        )
        self.threshold_check.pack(side='right')
        
        self.threshold_slider = tk.Scale(
            section,
            from_=0,
            to=255,
            orient="horizontal",
            command=self._on_threshold_changed,
            bg='#f0f0f0',
            highlightthickness=0
        )
        self.threshold_slider.set(128)
        self.threshold_slider.pack(fill='x')
    
    def _create_section_info(self):
        """Sección de información de la imagen."""
        section = tk.LabelFrame(
            self.controls_panel,
            text="Información",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0',
            padx=10,
            pady=5
        )
        section.pack(fill='x', padx=10, pady=10)
        
        self.info_text = tk.Label(
            section,
            text="No hay imagen cargada",
            bg='#f0f0f0',
            justify='left',
            anchor='w',
            font=('Arial', 9)
        )
        self.info_text.pack(fill='x')
    
    def _create_data_visualization_layout(self):
        """Crea el layout para visualización de datos."""
        # Configurar grid
        Grid.rowconfigure(self.data_tab, 0, weight=1)
        Grid.columnconfigure(self.data_tab, 0, weight=0)
        Grid.columnconfigure(self.data_tab, 1, weight=1)
        
        # Left sidebar for column selection
        sidebar_frame = tk.Frame(self.data_tab, relief=tk.RAISED, borderwidth=1, width=250)
        sidebar_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        sidebar_frame.grid_propagate(False)

        # Configure sidebar grid: columnconfigure; row weights set after building rows
        sidebar_frame.columnconfigure(0, weight=1)

        row = 0
        # Column list title
        tk.Label(sidebar_frame, text="Data Columns", font=("Arial", 12, "bold")).grid(
            row=row, column=0, pady=(10, 5), sticky='ew'
        )
        row += 1

        # Column listbox with scrollbar
        listbox_frame = tk.Frame(sidebar_frame)
        listbox_frame.grid(row=row, column=0, sticky='nsew', padx=5, pady=5)
        sidebar_frame.rowconfigure(row, weight=2)
        listbox_frame.rowconfigure(0, weight=1)
        listbox_frame.columnconfigure(0, weight=1)
        row += 1

        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.grid(row=0, column=1, sticky='ns')

        self.column_listbox = tk.Listbox(
            listbox_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.EXTENDED,
            font=("Arial", 10)
        )
        self.column_listbox.grid(row=0, column=0, sticky='nsew')
        self.column_listbox.bind('<<ListboxSelect>>', self.update_column_display)
        scrollbar.config(command=self.column_listbox.yview)

        # ── Peak Finder selector ──
        ttk.Separator(sidebar_frame, orient='horizontal').grid(
            row=row, column=0, sticky='ew', padx=5, pady=5
        )
        row += 1

        tk.Label(sidebar_frame, text="Peak Finder", font=("Arial", 10, "bold")).grid(
            row=row, column=0, pady=(5, 2), sticky='ew'
        )
        row += 1

        self.peak_method_combo = ttk.Combobox(
            sidebar_frame, textvariable=self.peak_method_var,
            values=['None', 'Elliptic Envelope', 'Peak Caller', 'Local Outlier Factor',
                    'Peak Function 4', 'Isolation Forest', 'Linear Model', 'Peak Function 7'],
            state='readonly', width=20
        )
        self.peak_method_combo.grid(row=row, column=0, padx=5, pady=(0, 5))
        self.peak_method_combo.bind('<<ComboboxSelected>>', lambda e: self._run_peak_on_column(show_dialog=True))
        self.peak_method_combo.config(state=DISABLED)
        row += 1

        # ── Correlation method selector ──
        ttk.Separator(sidebar_frame, orient='horizontal').grid(
            row=row, column=0, sticky='ew', padx=5, pady=5
        )
        row += 1

        tk.Label(sidebar_frame, text="Correlation Method", font=("Arial", 10, "bold")).grid(
            row=row, column=0, pady=(5, 2), sticky='ew'
        )
        row += 1

        corr_method_combo = ttk.Combobox(
            sidebar_frame, textvariable=self.corr_method_var,
            values=['pearson', 'kendall', 'spearman'],
            state='readonly', width=15
        )
        corr_method_combo.grid(row=row, column=0, padx=5, pady=(0, 5))
        corr_method_combo.bind('<<ComboboxSelected>>', lambda e: self._update_correlation_display())
        row += 1

        ttk.Checkbutton(
            sidebar_frame, text="Show Labels",
            variable=self.show_corr_labels_var,
            command=self._update_correlation_display
        ).grid(row=row, column=0, sticky='w', padx=5, pady=(0, 5))
        row += 1

        # ── Selection section ──
        ttk.Separator(sidebar_frame, orient='horizontal').grid(
            row=row, column=0, sticky='ew', padx=5, pady=5
        )
        row += 1

        tk.Label(sidebar_frame, text="Selection", font=("Arial", 12, "bold")).grid(
            row=row, column=0, pady=(0, 5), sticky='ew'
        )
        row += 1  # row 9 is the listbox (set weight=1 above)

        sel_listbox_frame = tk.Frame(sidebar_frame)
        sel_listbox_frame.grid(row=row, column=0, sticky='nsew', padx=5, pady=5)
        sidebar_frame.rowconfigure(row, weight=1)
        sel_listbox_frame.rowconfigure(0, weight=1)
        sel_listbox_frame.columnconfigure(0, weight=1)
        row += 1

        sel_scrollbar = tk.Scrollbar(sel_listbox_frame)
        sel_scrollbar.grid(row=0, column=1, sticky='ns')

        self.selection_listbox = tk.Listbox(
            sel_listbox_frame,
            yscrollcommand=sel_scrollbar.set,
            selectmode=tk.SINGLE,
            font=("Arial", 10)
        )
        self.selection_listbox.grid(row=0, column=0, sticky='nsew')
        sel_scrollbar.config(command=self.selection_listbox.yview)

        self.btn_add_sel = tk.Button(
            sidebar_frame, text="Add to Selection",
            command=self._add_to_selection, state=DISABLED
        )
        self.btn_add_sel.grid(row=row, column=0, sticky='ew', padx=5, pady=(2, 2))
        row += 1

        self.btn_remove_sel = tk.Button(
            sidebar_frame, text="Remove from Selection",
            command=self._remove_from_selection, state=DISABLED
        )
        self.btn_remove_sel.grid(row=row, column=0, sticky='ew', padx=5, pady=(0, 5))
        row += 1

        # ── Save buttons ──
        ttk.Separator(sidebar_frame, orient='horizontal').grid(
            row=row, column=0, sticky='ew', padx=5, pady=5
        )
        row += 1

        self.btn_save_data = tk.Button(
            sidebar_frame, text="Save Data Image",
            command=self._save_data_image, state=DISABLED
        )
        self.btn_save_data.grid(row=row, column=0, sticky='ew', padx=5, pady=(2, 2))
        row += 1

        self.btn_save_corr = tk.Button(
            sidebar_frame, text="Save Correlation",
            command=self._save_correlation_image, state=DISABLED
        )
        self.btn_save_corr.grid(row=row, column=0, sticky='ew', padx=5, pady=(0, 5))
        row += 1

        self.btn_save_peaks = tk.Button(
            sidebar_frame, text="Save Peaks CSV",
            command=self._save_peaks_csv, state=DISABLED
        )
        self.btn_save_peaks.grid(row=row, column=0, sticky='ew', padx=5, pady=(0, 10))

        # Right side - main plot area
        self.main_plot_frame = tk.Frame(self.data_tab, relief=tk.RAISED, borderwidth=1)
        self.main_plot_frame.grid(row=0, column=1, sticky='nsew')
        
        # Create placeholder label
        placeholder_label = tk.Label(
            self.main_plot_frame,
            text="Load a data file to start",
            font=("Arial", 20),
            bg="#f0f0f0",
            fg="#666666"
        )
        placeholder_label.pack(fill=tk.BOTH, expand=True)

    
    # ==================== DATA VISUALIZATION METHODS ====================
    
    def update_column_display(self, event=None):
        """Update the data graph when a column is selected. Does not refresh correlation."""
        if self.loaded_data is None:
            if hasattr(self.root, 'loaded_data'):
                self.loaded_data = self.root.loaded_data
        if self.loaded_data is None:
            return

        selection = self.column_listbox.curselection()
        if not selection:
            return

        # In EXTENDED mode use the last-clicked (ACTIVE) item for the display column
        try:
            self.current_column = self.column_listbox.index(tk.ACTIVE)
        except Exception:
            self.current_column = selection[0]

        # Initialize the split layout on first call (or if frames were destroyed)
        if self.plot_top_frame is None or not self.plot_top_frame.winfo_exists():
            for widget in list(self.main_plot_frame.winfo_children()):
                widget.destroy()
            self.main_plot_frame.rowconfigure(0, weight=3)
            self.main_plot_frame.rowconfigure(1, weight=2)
            self.main_plot_frame.columnconfigure(0, weight=1)
            self.plot_top_frame = tk.Frame(self.main_plot_frame)
            self.plot_top_frame.grid(row=0, column=0, sticky='nsew')
            self.plot_bottom_frame = tk.Frame(self.main_plot_frame, relief=tk.GROOVE, borderwidth=1)
            self.plot_bottom_frame.grid(row=1, column=0, sticky='nsew')
            self._update_correlation_display()

        # Delegate drawing to peak runner (handles None → raw data, or a peak method)
        self._run_peak_on_column()

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
                font=("Arial", 12),
                fg="#666666"
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

    def _run_peak_on_column(self, show_dialog=False, event=None):
        """Draw raw data or run the selected peak finder on the current column.
        show_dialog=True forces the parameter dialog (used when the method changes).
        show_dialog=False reuses cached params (used when the column changes)."""
        if self.loaded_data is None or self.plot_top_frame is None:
            return
        if not self.plot_top_frame.winfo_exists():
            return

        method = self.peak_method_var.get()
        col_idx = self.current_column

        if self._data_fig is not None:
            plt.close(self._data_fig)
            self._data_fig = None
        for w in list(self.plot_top_frame.winfo_children()):
            w.destroy()

        if method == 'None':
            self._draw_raw_data(col_idx)
            return

        # Show dialog only when method changes or no params saved yet
        if show_dialog or method not in self.peak_method_params:
            from peak_functions import show_parameter_dialog
            spec = self._PEAK_PARAM_SPECS.get(method)
            if spec:
                title, param_list = spec
                new_params = show_parameter_dialog(self.root, title, param_list)
                if new_params is None:
                    # User cancelled — revert to raw data view
                    self.peak_method_var.set('None')
                    self._draw_raw_data(col_idx)
                    return
                self.peak_method_params[method] = new_params

        saved_params = self.peak_method_params.get(method)

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
        if func:
            result = func(self.loaded_data, col_idx,
                          main_window=None, canvas=None,
                          target_frame=self.plot_top_frame,
                          params=saved_params)
            if result is not None:
                self.canvas, self._data_fig = result

    def _draw_raw_data(self, col_idx):
        """Plot the raw normalized data for col_idx into plot_top_frame."""
        col_label = self.column_listbox.get(col_idx)
        self._data_fig, ax = plt.subplots()
        ax.plot(
            np.array(range(len(self.loaded_data[:, col_idx]))).reshape(-1, 1),
            self.loaded_data[:, col_idx]
        )
        ax.set_title(col_label)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        self._data_fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self._data_fig, master=self.plot_top_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _save_data_image(self):
        """Save the current data / peak-finder plot to a file."""
        if self._data_fig is None:
            return
        from tkinter.filedialog import asksaveasfilename
        filename = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                       ("TIFF files", "*.tiff"), ("All Files", "*.*")],
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
                       ("TIFF files", "*.tiff"), ("All Files", "*.*")],
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
            self.btn_add_sel.config(state=NORMAL)
            self.btn_remove_sel.config(state=NORMAL)
            self.btn_save_data.config(state=NORMAL)
            self.btn_save_corr.config(state=NORMAL)
            self.btn_save_peaks.config(state=NORMAL)
            self.peak_method_combo.config(state='readonly')
            self.menu_visual.entryconfig(
                "Dendograma",
                command=self._run_dendogram_on_selection,
                state=NORMAL
            )
    
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
            peaks = compute_peaks(self.loaded_data, col_idx, method, params)
            flags = np.zeros(n_time, dtype=int)
            flags[peaks] = 1
            data_dict[col_name] = flags

        pd.DataFrame(data_dict).to_csv(filename, index=False)
        messagebox.showinfo("Saved", f"Peak data saved to:\n{filename}")

    def _run_dendogram_on_selection(self):
        """Create the Dendogram tab on first use, then switch to it."""
        if self.dendo_tab is None or not self.dendo_tab.winfo_exists():
            self.dendo_tab = tk.Frame(self.notebook, bg='#f0f0f0')
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
                self.btn_dendo_add_sel.config(state=NORMAL)
                self.btn_dendo_remove_sel.config(state=NORMAL)
                self.btn_dendo_save_img.config(state=NORMAL)
                self.btn_dendo_save_csv.config(state=NORMAL)
        self.notebook.select(self.dendo_tab)

    # ==================== DENDOGRAM TAB ====================

    def _create_dendogram_layout(self):
        """Build the permanent Dendogram tab (sidebar + plot area)."""
        Grid.rowconfigure(self.dendo_tab, 0, weight=1)
        Grid.columnconfigure(self.dendo_tab, 0, weight=0)
        Grid.columnconfigure(self.dendo_tab, 1, weight=1)

        # ── Sidebar ──
        sidebar = tk.Frame(self.dendo_tab, relief=tk.RAISED, borderwidth=1, width=250)
        sidebar.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)

        drow = 0
        tk.Label(sidebar, text="Data Columns", font=("Arial", 12, "bold")).grid(
            row=drow, column=0, pady=(10, 5), sticky='ew'
        )
        drow += 1

        lb_frame = tk.Frame(sidebar)
        lb_frame.grid(row=drow, column=0, sticky='nsew', padx=5, pady=5)
        sidebar.rowconfigure(drow, weight=2)
        lb_frame.rowconfigure(0, weight=1)
        lb_frame.columnconfigure(0, weight=1)
        drow += 1

        lb_sb = tk.Scrollbar(lb_frame)
        lb_sb.grid(row=0, column=1, sticky='ns')
        self.dendo_column_listbox = tk.Listbox(
            lb_frame, yscrollcommand=lb_sb.set,
            selectmode=tk.EXTENDED, font=("Arial", 10)
        )
        self.dendo_column_listbox.grid(row=0, column=0, sticky='nsew')
        lb_sb.config(command=self.dendo_column_listbox.yview)

        # ── Selection ──
        ttk.Separator(sidebar, orient='horizontal').grid(
            row=drow, column=0, sticky='ew', padx=5, pady=5
        )
        drow += 1

        tk.Label(sidebar, text="Selection", font=("Arial", 12, "bold")).grid(
            row=drow, column=0, pady=(0, 5), sticky='ew'
        )
        drow += 1

        sel_frame = tk.Frame(sidebar)
        sel_frame.grid(row=drow, column=0, sticky='nsew', padx=5, pady=5)
        sidebar.rowconfigure(drow, weight=1)
        sel_frame.rowconfigure(0, weight=1)
        sel_frame.columnconfigure(0, weight=1)
        drow += 1

        sel_sb = tk.Scrollbar(sel_frame)
        sel_sb.grid(row=0, column=1, sticky='ns')
        self.dendo_selection_listbox = tk.Listbox(
            sel_frame, yscrollcommand=sel_sb.set,
            selectmode=tk.SINGLE, font=("Arial", 10)
        )
        self.dendo_selection_listbox.grid(row=0, column=0, sticky='nsew')
        sel_sb.config(command=self.dendo_selection_listbox.yview)

        self.btn_dendo_add_sel = tk.Button(
            sidebar, text="Add to Selection",
            command=self._dendo_add_to_selection, state=DISABLED
        )
        self.btn_dendo_add_sel.grid(row=drow, column=0, sticky='ew', padx=5, pady=(2, 2))
        drow += 1

        self.btn_dendo_remove_sel = tk.Button(
            sidebar, text="Remove from Selection",
            command=self._dendo_remove_from_selection, state=DISABLED
        )
        self.btn_dendo_remove_sel.grid(row=drow, column=0, sticky='ew', padx=5, pady=(0, 5))
        drow += 1

        # ── Save buttons ──
        ttk.Separator(sidebar, orient='horizontal').grid(
            row=drow, column=0, sticky='ew', padx=5, pady=5
        )
        drow += 1

        self.btn_dendo_save_img = tk.Button(
            sidebar, text="Save Dendrogram Image",
            command=self._dendo_save_image, state=DISABLED
        )
        self.btn_dendo_save_img.grid(row=drow, column=0, sticky='ew', padx=5, pady=(2, 2))
        drow += 1

        self.btn_dendo_save_csv = tk.Button(
            sidebar, text="Save Dendrogram CSV",
            command=self._dendo_save_csv, state=DISABLED
        )
        self.btn_dendo_save_csv.grid(row=drow, column=0, sticky='ew', padx=5, pady=(0, 10))

        # ── Plot area (top: signal preview, bottom: dendrogram) ──
        self.dendo_plot_frame = tk.Frame(self.dendo_tab, relief=tk.RAISED, borderwidth=1)
        self.dendo_plot_frame.grid(row=0, column=1, sticky='nsew')
        self.dendo_plot_frame.rowconfigure(0, weight=1)
        self.dendo_plot_frame.rowconfigure(1, weight=1)
        self.dendo_plot_frame.columnconfigure(0, weight=1)

        self.dendo_top_frame = tk.Frame(self.dendo_plot_frame)
        self.dendo_top_frame.grid(row=0, column=0, sticky='nsew')
        tk.Label(
            self.dendo_top_frame,
            text="Click a column to view its signal",
            font=("Arial", 14), fg="#666666"
        ).pack(fill=tk.BOTH, expand=True)

        self.dendo_bottom_frame = tk.Frame(
            self.dendo_plot_frame, relief=tk.GROOVE, borderwidth=1
        )
        self.dendo_bottom_frame.grid(row=1, column=0, sticky='nsew')
        tk.Label(
            self.dendo_bottom_frame,
            text="Add 2+ columns to Selection to see the dendrogram",
            font=("Arial", 14), fg="#666666"
        ).pack(fill=tk.BOTH, expand=True)

        # Bind column click to signal preview
        self.dendo_column_listbox.bind('<<ListboxSelect>>', self._dendo_show_signal)

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

    def _dendo_show_signal(self, event=None):
        """Draw the clicked column's signal into the top frame."""
        if self.loaded_data is None or self.dendo_top_frame is None:
            return
        sel = self.dendo_column_listbox.curselection()
        if not sel:
            return
        try:
            col_idx = self.dendo_column_listbox.index(tk.ACTIVE)
        except Exception:
            col_idx = sel[0]
        self.dendo_current_column = col_idx

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
                font=("Arial", 14), fg="#666666"
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
                       ("All Files", "*.*")],
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
