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
            label="Salir",
            command=self._on_close
        )
        
        # Menú Imagen
        self.menu_imagen = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_imagen, label="Imagen")
        self.menu_imagen.add_command(
            label="Auto Contraste",
            command=self.apply_auto_contrast
        )
        self.menu_imagen.add_command(
            label="Histogram",
            command=self.show_histogram
        )
        self.menu_imagen.add_command(
            label="Binarize",
            command=self.show_binarize
        )
        self.menu_imagen.add_separator()
        self.menu_imagen.add_command(
            label="Restaurar Original",
            command=self.restore_original
        )
        
        # Submenú de Análisis de Variabilidad
        self.menu_imagen.add_separator()
        self.menu_variabilidad = Menu(self.menu_imagen, tearoff=False)
        self.menu_imagen.add_cascade(menu=self.menu_variabilidad, label="Análisis de Variabilidad")
        
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
        self.menu_visual.add_command(
            label='Abrir Datos (.npy)',
            command=self.open_visualization_data,
            state=NORMAL
        )
        self.menu_visual.add_separator()
        
        # Opciones de suavizado
        self.menu_visual.add_command(
            label='Finite difference diffusion smoothing',
            command=None,
            state=DISABLED
        )
        self.menu_visual.add_command(
            label='Exponential moving average smoothing',
            command=None,
            state=DISABLED
        )
        self.menu_visual.add_command(
            label='Convex envelope smoothing',
            command=None,
            state=DISABLED
        )
        self.menu_visual.add_separator()
        
        # Opciones de detección de picos
        peak_methods = [
            'Elliptic Envelope',
            'Peak Caller',
            'Local Outlier Factor',
            'Peak Function 4',
            'Isolation Forest',
            'Linear Model',
            'Peak Function 7'
        ]
        for method in peak_methods:
            self.menu_visual.add_command(label=method, command=None, state=DISABLED)
        
        self.menu_visual.add_separator()
        
        # Opciones de correlación
        corr_methods = [
            'Correlacion Pearson',
            'Correlacion Kendall',
            'Correlacion Spearman'
        ]
        for method in corr_methods:
            self.menu_visual.add_command(label=method, command=None, state=DISABLED)
        
        self.menu_visual.add_separator()
        self.menu_visual.add_command(label='Dendograma', command=None, state=DISABLED)
        self.menu_visual.add_separator()
        self.menu_visual.add_command(label='Series de tiempo', command=None, state=DISABLED)
        
        # Menú Correlación
        self.menu_correlacion = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_correlacion, label="Correlacion")
        self.menu_correlacion.add_command(
            label='Cargar matriz de correlacion',
            command=self.load_correlation_matrix_wrapper,
            state=NORMAL
        )
        
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
        
        # Column list title
        list_title = tk.Label(sidebar_frame, text="Data Columns", font=("Arial", 12, "bold"))
        list_title.pack(pady=(10, 5))
        
        # Column listbox with scrollbar
        listbox_frame = tk.Frame(sidebar_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.column_listbox = tk.Listbox(
            listbox_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            font=("Arial", 10)
        )
        self.column_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.column_listbox.bind('<<ListboxSelect>>', self.update_column_display)
        
        scrollbar.config(command=self.column_listbox.yview)
        
        # Info label
        info_label = tk.Label(
            sidebar_frame,
            text="Load a file to\nsee columns",
            justify=tk.CENTER,
            wraplength=220,
            font=("Arial", 10)
        )
        info_label.pack(pady=10)
        
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
        """Update the plot when a different column is selected from the listbox."""
        if self.loaded_data is None:
            return
        
        selection = self.column_listbox.curselection()
        if not selection:
            return
        
        self.current_column = selection[0]
        
        # Remove any peak button frames when switching columns
        for widget in list(self.root.winfo_children()):
            if isinstance(widget, tk.Frame) and hasattr(widget, 'peak_button_frame'):
                widget.destroy()
        
        # Clear ALL widgets from the plot frame
        for widget in list(self.main_plot_frame.winfo_children()):
            widget.destroy()
        
        # Close all matplotlib figures to free memory
        plt.close('all')
        
        # Create new plot
        fig, ax = plt.subplots()
        plt.plot(
            np.array(range(len(self.loaded_data[:, self.current_column]))).reshape(-1, 1),
            self.loaded_data[:, self.current_column]
        )
        plt.title(f'Column {self.current_column + 1}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        
        # Display in main frame
        self.canvas = FigureCanvasTkAgg(fig, master=self.main_plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def open_visualization_data(self):
        """Abre datos para visualización de picos."""
        # Pass data_tab so _plot_data can find main_plot_frame (at column=1)
        # instead of self.root whose children include the menu bar and notebook.
        canvas = initialize_visualization(
            self.data_tab,
            self.menu_visual,
            self.canvas,
            self.column_listbox,
            self.update_column_display
        )
        if canvas is not None:
            self.canvas = canvas
            # Switch to the data visualization tab
            self.notebook.select(self.data_tab)
        # Update loaded_data reference after initialization
        # (stored on root by _plot_data_with_menu via master traversal)
        if hasattr(self.root, 'loaded_data'):
            self.loaded_data = self.root.loaded_data
    
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



def main():
    """Función principal."""
    root = tk.Tk()
    app = NecLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

