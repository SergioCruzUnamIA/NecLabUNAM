"""
NecLab - Herramienta de análisis de imágenes de microscopía
Interfaz gráfica principal
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  # Limita número de threads

import tkinter as tk
from tkinter import Menu, Grid, filedialog, FALSE, DISABLED, NORMAL, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

# Módulos locales
from pyometiff import OMETIFFReader
from image_loader import load_ometiff_image, process_image_slice
from image_processing import auto_contrast, threshold_image_pil
from visualization_helpers import initialize_visualization
from variability_functions import show_variability_analysis, get_variability_methods


class NecLabApp:
    """Clase principal de la aplicación NecLab."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("NecLab")
        self.root.tk.call('tk', 'windowingsystem')
        self.root.option_add('*tearOff', FALSE)
        
        # Configurar tamaño de ventana (90% de la pantalla)
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.width = int(self.screen_width * 0.9)
        self.height = int(self.screen_height * 0.9)
        self.root.geometry(f"{self.width}x{self.height}")
        
        # Variables de estado
        self.img_original = None  # Imagen original sin modificar
        self.img_array = None     # Imagen de trabajo (puede tener modificaciones)
        self.img_display = None   # Imagen para visualización (con contraste, etc.)
        self.canvas = None
        
        # Construir la interfaz
        self._create_menu()
        self._create_layout()
        self._load_default_image()
        
        # Manejar cierre de ventana para liberar todo el proceso
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self.root.focus()
    
    def _on_close(self):
        """Cerrar la aplicación completamente, liberando todos los recursos."""
        plt.close('all')
        self.root.quit()
        self.root.destroy()
    
    # ==================== MENÚ ====================
    
    def _create_menu(self):
        """Crea la barra de menús."""
        self.menu_bar = Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        # Menú Archivo
        self.menu_archivo = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_archivo, label="Archivo")
        self.menu_archivo.add_command(
            label="Abrir OME-TIFF",
            accelerator="Ctrl+O",
            command=self.open_file
        )
        self.menu_archivo.add_separator()
        self.menu_archivo.add_command(
            label="Salir",
            command=self.root.quit
        )
        
        # Menú Imagen
        self.menu_imagen = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(menu=self.menu_imagen, label="Imagen")
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
        self.menu_bar.add_cascade(menu=self.menu_visual, label="Visualización")
        self.menu_visual.add_command(
            label="Abrir Datos (.npy)",
            command=self.open_visualization_data
        )
        self.menu_visual.add_separator()
        
        # Opciones de detección de picos (inicialmente deshabilitadas)
        self.peak_methods = [
            "Elliptic Envelope",
            "Peak Caller", 
            "Local Outlier Factor",
            "Isolation Forest",
            "Linear Model"
        ]
        for method in self.peak_methods:
            self.menu_visual.add_command(label=method, command=None, state=DISABLED)
        
        self.menu_visual.add_separator()
        
        # Opciones de correlación (inicialmente deshabilitadas)
        self.corr_methods = [
            "Correlación Pearson",
            "Correlación Kendall", 
            "Correlación Spearman"
        ]
        for method in self.corr_methods:
            self.menu_visual.add_command(label=method, command=None, state=DISABLED)
        
        self.menu_visual.add_separator()
        self.menu_visual.add_command(label="Dendrograma", command=None, state=DISABLED)
        
        # Atajo de teclado
        self.root.bind('<Control-o>', lambda e: self.open_file())
    
    # ==================== LAYOUT PRINCIPAL ====================
    
    def _create_layout(self):
        """Crea el layout principal con imagen a la izquierda y controles a la derecha."""
        # Configurar grid principal
        self.root.columnconfigure(0, weight=3)  # Columna de imagen (más grande)
        self.root.columnconfigure(1, weight=1)  # Columna de controles
        self.root.rowconfigure(0, weight=1)
        
        # Frame izquierdo para la imagen
        self._create_image_frame()
        
        # Frame derecho para los controles
        self._create_controls_panel()
    
    def _create_image_frame(self):
        """Crea el frame para mostrar la imagen."""
        self.image_frame = tk.Frame(self.root, bg='#1e1e1e', relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Label para mostrar la imagen
        self.image_label = tk.Label(self.image_frame, bg='#1e1e1e')
        self.image_label.pack(fill=tk.BOTH, expand=True)
    
    def _create_controls_panel(self):
        """Crea el panel lateral derecho con todos los controles."""
        # Frame principal del panel
        self.controls_panel = tk.Frame(self.root, bg='#f0f0f0', relief=tk.RAISED, borderwidth=2)
        self.controls_panel.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        
        # Título del panel
        title_label = tk.Label(
            self.controls_panel, 
            text="Controles", 
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
            command=self._update_display,
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
    
    # ==================== FUNCIONES DE IMAGEN ====================
    
    def _load_default_image(self):
        """Carga una imagen por defecto o muestra placeholder."""
        try:
            pil_img = Image.open('input_image_7.png')
        except FileNotFoundError:
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
        factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
        img = factor * (img - 128) + 128
        
        # Clamp valores
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    def _update_display(self):
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
            pil_img = threshold_image_pil(current_slice, threshold=threshold_val)
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
        self._update_display()
    
    # ==================== CALLBACKS ====================
    
    def _on_slice_changed(self, val):
        """Callback cuando cambia el slider de capa."""
        self._update_display()
    
    def _on_threshold_changed(self, val):
        """Callback cuando cambia el slider de threshold."""
        if self.threshold_enabled.get():
            self._update_display()
    
    def _on_adjustment_changed(self, val):
        """Callback cuando cambian los ajustes de brillo/contraste."""
        self._update_display()
    
    # ==================== COMANDOS DE MENÚ ====================
    
    def open_file(self):
        """Abre un archivo OME-TIFF."""
        img, metadata, xml_metadata = load_ometiff_image()
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
        
        self._update_display()
    
    def restore_original(self):
        """Restaura la imagen original."""
        if self.img_original is None:
            return
        self.img_array = self.img_original.copy()
        self._reset_adjustments()
    
    def apply_auto_contrast(self):
        """Aplica auto contraste solo para visualización (no modifica datos originales)."""
        if self.img_array is None:
            return
        
        # auto_contrast ya trabaja sobre una copia
        self.img_display = auto_contrast(self.img_array)
        self._update_display()
    
    def open_visualization_data(self):
        """Abre datos para visualización de picos."""
        initialize_visualization(self.root, self.menu_visual, self.canvas)
    
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