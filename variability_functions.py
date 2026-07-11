import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from skimage.filters import unsharp_mask
from tkinter.filedialog import asksaveasfilename
from sklearn.cluster import KMeans
from progress_utils import run_save_with_progress

def calculate_variability(img_array, method=1):
    """
    Calcula la variabilidad de una imagen usando diferentes métodos.
    
    """
    methods_info = {
        0: {"name": "Rango", "default_th": 100},
        1: {"name": "Varianza Poblacional", "default_th": 120},
        2: {"name": "Varianza Muestral", "default_th": 200},
        3: {"name": "Desviación Estándar Poblacional", "default_th": 12},
        4: {"name": "Desviación Estándar Muestral", "default_th": 5},
        5: {"name": "Coeficiente de Variación", "default_th": 5},
        6: {"name": "Rango Intercuartílico (IQR)", "default_th": 20}
    }
    
    select = method
    
    if select == 0:
        # 1. Rango
        var_im = np.max(img_array, axis=0) - np.min(img_array, axis=0)
        th = 100
    elif select == 1:
        # 2. Varianza
        var_im = np.var(img_array, axis=0)  # Poblacional
        th = 120
    elif select == 2:
        # 2. Varianza
        var_im = np.var(img_array, axis=0, ddof=1)  # Muestral
        th = 200
    elif select == 3:
        # 3. Desviación estándar
        th = 12
        var_im = np.std(img_array, axis=0)  # Poblacional
    elif select == 4:
        # 3. Desviación estándar
        th = 5
        var_im = np.std(img_array, axis=0, ddof=1)  # Muestral
    elif select == 5:
        # 4. Coeficiente de variación
        th = 5
        var_im = np.std(img_array, axis=0, ddof=1)  # Muestral
        media = np.mean(img_array, axis=0)
        var_im = (var_im / media) * 100
    elif select == 6:
        th = 20
        # 5. Rango intercuartílico (IQR)
        q1 = np.percentile(img_array, 25, axis=0)
        q3 = np.percentile(img_array, 75, axis=0)
        var_im = q3 - q1
    else:
        raise ValueError("Método debe estar entre 0 y 6")
    
    return var_im, th, methods_info[method]["name"]

def apply_image_processing(var_im):
    """
    Aplica el procesamiento de imagen: unsharp mask y filtro.
    """
    result_1 = unsharp_mask(var_im, radius=20, amount=1)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv.filter2D(var_im, -1, kernel)
    return image

def apply_binarization(image, th):
    """
    Aplica binarización
    """
    deconvolved_RL2 = np.reshape(image, (image.shape[0] * image.shape[1]))
    res_labels = [int(deconvolved_RL2[i] > th) for i in range(len(deconvolved_RL2))]
    res_labels = np.reshape(res_labels, (image.shape[0], image.shape[1]))
    return res_labels

def candidate_neighbors(node):
    """
    Función original para encontrar vecinos.
    """
    return [(node[0] + 1, node[1]), (node[0], node[1] + 1), (node[0] + 1, node[1] + 1),
           (node[0] - 1, node[1]), (node[0], node[1] - 1), (node[0] - 1, node[1] - 1),
           (node[0] + 1, node[1] - 1), (node[0] - 1, node[1] + 1)]

def neighboring_groups(nodes):
    """
    Función original para agrupar píxeles conectados.
    """
    remain = set(nodes)
    while len(remain) > 0:
        visit = [remain.pop()]
        group = []
        while len(visit) > 0:
            node = visit.pop()
            group.append(node)
            for nb in candidate_neighbors(node):
                if nb in remain:
                    remain.remove(nb)
                    visit.append(nb)
        yield group

def extract_pixels_from_binary(res_labels):
    """
    Extrae píxeles de la imagen binarizada.
    """
    pixels = []
    for i in range(res_labels.shape[0]):
        for j in range(res_labels.shape[1]):
            if res_labels[i][j] == True:
                pixels.append((i, j))
    return pixels

def find_peaks(Z, dz_dx, dz_dy, selected_points):
    """
    Detecta picos usando cambio de signo en las derivadas.
    Código original de Jose.
    """
    peaks = []
    for i, j in selected_points:
        # Verificar que no estemos en los bordes
        if 1 <= i < Z.shape[0] - 1 and 1 <= j < Z.shape[1] - 1:
            # Buscar si es un pico (máximo local)
            if dz_dx[i, j - 1] > 0 and dz_dx[i, j] <= 0 and dz_dy[i - 1, j] > 0 and dz_dy[i, j] <= 0:
                peaks.append((i, j))
    return peaks

def assign_points_to_peaks(Z, peaks, selected_points):
    """
    Asigna puntos seleccionados a los picos más cercanos.
    Código original de Jose.
    """
    import math
    
    # Inicializar una lista para los conjuntos de puntos por pico
    peak_sets = {i: [] for i in range(len(peaks))}
    
    # Crear una matriz de etiquetas para asignar puntos a conjuntos
    peak_map = np.zeros_like(Z, dtype=int) - 1
    gx = np.gradient(Z, axis=1)  # Derivada parcial respecto a x
    gy = np.gradient(Z, axis=0)  # Derivada parcial respecto a y
    
    # Etiquetar los picos con IDs únicos
    for label_id, (pi, pj) in enumerate(peaks):
        if 0 <= pi < Z.shape[0] and 0 <= pj < Z.shape[1]:
            peak_map[pi, pj] = label_id
    
    # Asignar los puntos seleccionados a sus picos más cercanos
    for i, j in selected_points:
        if not (0 <= i < Z.shape[0] and 0 <= j < Z.shape[1]):
            continue
            
        x, y = i, j
        seen = []
        
        # Gradient descent hasta encontrar un pico
        while (x, y) not in seen and (x, y) not in peaks:
            if len(seen) > 100:  # Evitar loops infinitos
                break
                
            if not (0 <= x < Z.shape[0] and 0 <= y < Z.shape[1]):
                break
                
            # Obtener dirección del gradiente
            if abs(gx[x, y]) > 1e-10 or abs(gy[x, y]) > 1e-10:
                dx, dy = int(np.sign(gx[x, y])), int(np.sign(gy[x, y]))
            else:
                break
            
            # Moverse en la dirección del gradiente
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < Z.shape[0]) and (0 <= new_y < Z.shape[1]):
                seen.append((x, y))
                x, y = new_x, new_y
            else:
                break
        
        # Encontrar el pico más cercano
        if peaks:
            min_dist = float('inf')
            sel_peak = None
            for p in peaks:
                try:
                    dst = math.dist(p, (x, y))
                    if dst < min_dist:
                        min_dist = dst
                        sel_peak = p
                except:
                    continue
            
            # Asignar el punto al conjunto del pico
            if sel_peak is not None:
                peak_x, peak_y = sel_peak
                if 0 <= peak_x < peak_map.shape[0] and 0 <= peak_y < peak_map.shape[1]:
                    peak_id = peak_map[peak_x, peak_y]
                    if peak_id >= 0 and peak_id in peak_sets:
                        peak_sets[peak_id].append((i, j))
    
    return peak_sets

def process_clusters_advanced(var_im, res_clusters, min_size=20, max_size=200):
    """
    Procesa clusters con detección de picos avanzada.
    Código original de Jose adaptado con validaciones robustas.
    """
    try:
        from sklearn import linear_model
        
        clusters_min_size = []
        final_cl = []
        
        # Calcular derivadas parciales
        dz_dx = np.gradient(var_im, axis=1)
        dz_dy = np.gradient(var_im, axis=0)
        
        for cl in res_clusters:
            try:
                if max_size >= len(cl) >= min_size:
                    final_cl.append(cl)
                elif len(cl) >= min_size:
                    clusters_min_size.append(cl)
            except:
                continue
        
        # Procesar clusters con detección de picos
        for cl in clusters_min_size:
            try:
                if len(cl) == 0:
                    continue
                    
                # Extraer coordenadas y valores
                z = []
                x = []
                y = []
                
                for coord in cl:
                    if len(coord) >= 2:
                        i, j = coord[0], coord[1]
                        if 0 <= i < var_im.shape[0] and 0 <= j < var_im.shape[1]:
                            z.append(var_im[i, j])
                            x.append(i)
                            y.append(j)
                
                if len(z) < 3:  # Necesitamos al menos 3 puntos
                    final_cl.append(cl)
                    continue
                
                z = np.array(z)
                y = np.array(y)
                x = np.array(x)
                
                # Clasificación con SGD One-Class SVM
                try:
                    clf = linear_model.SGDOneClassSVM(random_state=42, nu=0.131)
                    clf.fit(z.reshape(-1, 1))
                    y_pred = clf.predict(z.reshape(-1, 1))
                    y_res = [i for i, pred in enumerate(list(y_pred)) if pred == -1]
                except:
                    # Si falla la clasificación, usar cluster original
                    final_cl.append(cl)
                    continue
                
                # Crear grid para superficie
                if len(x) > 0 and len(y) > 0:
                    min_x, max_x = int(min(x)), int(max(x))
                    min_y, max_y = int(min(y)), int(max(y))
                    
                    if max_x > min_x and max_y > min_y:
                        Z = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
                        
                        # Llenar grid con valores
                        for i in range(len(z)):
                            grid_x = int(x[i] - min_x)
                            grid_y = int(y[i] - min_y)
                            if 0 <= grid_x < Z.shape[0] and 0 <= grid_y < Z.shape[1]:
                                Z[grid_x, grid_y] = z[i]
                        
                        # Detectar picos
                        peaks = find_peaks(Z, dz_dx[min_x:max_x+1, min_y:max_y+1], 
                                         dz_dy[min_x:max_x+1, min_y:max_y+1], 
                                         [(i-min_x, j-min_y) for i, j in cl if min_x <= i <= max_x and min_y <= j <= max_y])
                        
                        # Convertir picos de vuelta a coordenadas originales
                        original_peaks = [(p[0] + min_x, p[1] + min_y) for p in peaks]
                        
                        if len(original_peaks) > 0:
                            # Asignar puntos a picos
                            relative_cl = [(i-min_x, j-min_y) for i, j in cl if min_x <= i <= max_x and min_y <= j <= max_y]
                            peak_sets = assign_points_to_peaks(Z, peaks, relative_cl)
                            
                            # Convertir de vuelta a coordenadas originales y agregar
                            for peak_id, points in peak_sets.items():
                                if points:
                                    original_points = [(p[0] + min_x, p[1] + min_y) for p in points]
                                    final_cl.append(original_points)
                        else:
                            final_cl.append(cl)
                    else:
                        final_cl.append(cl)
                else:
                    final_cl.append(cl)
                    
            except Exception as e:
                # Si algo falla en el procesamiento avanzado, usar cluster original
                pass  # Error silencioso
                final_cl.append(cl)
                continue
        
        return final_cl, clusters_min_size
        
    except Exception as e:
        pass  # Error silencioso
        # Fallback al método básico
        final_cl = []
        for cl in res_clusters:
            if min_size <= len(cl) <= max_size:
                final_cl.append(cl)
        return final_cl, []

def extract_time_series(img_array, cluster_points):
    """
    Extrae serie temporal de un cluster específico.
    """
    final_array = np.array(cluster_points)
    sel_data_final = img_array[:, final_array[:,0], final_array[:,1]]
    sel_data_final_mean = np.mean(sel_data_final, axis=1)
    return sel_data_final_mean

def decompose_large_clusters(res_clusters, var_im, min_size=20, max_size=200):
    """
    Descompone clusters grandes usando KMeans con features espaciales + variabilidad.
    Los clusters menores a min_size se descartan, los que están entre min_size y max_size
    se mantienen, y los mayores a max_size se subdividen con KMeans.
    Basado en el notebook cluster-Copy4.ipynb.
    """
    decomposed = []
    max_ax = np.max(var_im.shape)
    max_var = var_im.max()
    if max_var == 0:
        max_var = 1  # Evitar división por cero
    norm_var_im = (var_im / max_var) * max_ax
    
    for cl in res_clusters:
        cl_size = len(cl)
        if cl_size < min_size:
            continue  # Descartar clusters muy pequeños
        elif cl_size <= max_size:
            decomposed.append(cl)  # Mantener clusters de tamaño adecuado
        else:
            # Descomponer clusters grandes con KMeans
            num_sub = (cl_size // max_size) + 1
            try:
                # Crear features 3D: coordenadas escaladas + variabilidad normalizada
                new_cl = [
                    (cl[i][0] * 10, cl[i][1] * 10, norm_var_im[cl[i][0], cl[i][1]])
                    for i in range(cl_size)
                ]
                kmeans = KMeans(n_clusters=num_sub, random_state=0, n_init="auto").fit(new_cl)
                labels = kmeans.labels_
                for label_val in np.unique(labels):
                    sub_cluster = np.array(cl)[labels == label_val]
                    # Convertir de vuelta a lista de tuplas
                    sub_cluster_list = [tuple(p) for p in sub_cluster.tolist()]
                    if len(sub_cluster_list) >= min_size:
                        decomposed.append(sub_cluster_list)
            except Exception as e:
                pass  # Error silencioso
                decomposed.append(cl)  # Mantener original si falla
    
    return decomposed

class VariabilityAnalysisWindow:
    """
    Ventana para análisis de variabilidad con selección de clusters.
    """
    def __init__(self, img_array, method, main_window):
        self.img_array = img_array
        self.method = method
        self.main_window = main_window
        
        # Calcular variabilidad inicial
        self.var_im, self.default_th, self.method_name = calculate_variability(img_array, method)
        self.processed_image = apply_image_processing(self.var_im)
        
        # Variables para clustering
        self.res_labels = None
        self.res_clusters = None
        self.final_clusters = None
        self.selected_clusters = []
        self.scatter_objects = []
        self.rect_selector = None
        self.selection_mode = 'add'  # 'add' o 'remove'
        self.cluster_colors = []  # Colores aleatorios por cluster
        
        # Variables para análisis
        self.time_series_data = []
        self.cluster_labels = []
        
        self.create_window()
        self.update_display()
    
    def _generate_cluster_colors(self, n_clusters):
        """Genera colores aleatorios para cada cluster, excluyendo rojo (reservado para selección)."""
        import colorsys
        colors = []
        rng = np.random.RandomState(42)
        for _ in range(n_clusters):
            # Generar colores con hue lejos del rojo (0.0)
            # Rojo está en hue ~0.0 y ~1.0, evitamos rango [0.0, 0.05] y [0.92, 1.0]
            hue = rng.uniform(0.08, 0.88)
            sat = rng.uniform(0.5, 1.0)
            val = rng.uniform(0.6, 1.0)
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append(rgb)
        return colors
    
    def create_window(self):
        """Crear la ventana principal"""
        self.window = tk.Toplevel(self.main_window)
        self.window.title(f"Análisis Completo - {self.method_name}")
        self.window.geometry("1400x800")

        # Manejar cierre de ventana
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Barra de menú
        self.window.config(menu=self._create_menu_bar())

        # Frame principal
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Barra de parámetros (threshold, tamaños) e info
        self.create_controls(main_frame)

        # Frame horizontal para visualización y selección
        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Frame para visualización (lado izquierdo)
        self.viz_frame = tk.Frame(content_frame)
        self.viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame para selección (lado derecho)
        self.selection_frame = tk.Frame(content_frame, width=200, relief=tk.RAISED, borderwidth=1)
        self.selection_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.selection_frame.pack_propagate(False)

        self.create_selection_panel()
    
    def _create_menu_bar(self):
        """Crear la barra de menús de la ventana de análisis."""
        menu_bar = tk.Menu(self.window, tearoff=False)

        # Menú Clustering
        menu_clustering = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Clustering", menu=menu_clustering)
        menu_clustering.add_command(label="Procesar Cluster (Básico)", command=self.process_clusters_basic)
        menu_clustering.add_command(label="Procesar Cluster (Avanzado)", command=self.process_clusters_advanced)
        menu_clustering.add_separator()
        menu_clustering.add_command(label="Descomponer Clusters Grandes", command=self.decompose_clusters)

        # Menú Selección
        menu_seleccion = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Selección", menu=menu_seleccion)
        menu_seleccion.add_command(label="Seleccionar Todos", command=self.select_all_clusters)
        menu_seleccion.add_command(label="Limpiar Selección", command=self.clear_selection)

        # Menú Visualización
        menu_visual = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Visualización", menu=menu_visual)
        menu_visual.add_command(label="Vista 3D", command=self.show_3d_surface)

        # Menú Exportar
        menu_exportar = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Exportar", menu=menu_exportar)
        menu_exportar.add_command(label="Guardar Imagen", command=self.save_image)
        menu_exportar.add_command(label="Guardar .npy", command=self.save_selected_npy)
        menu_exportar.add_separator()
        menu_exportar.add_command(label="Usar Seleccionados (Correlaciones)", command=self.use_selected_clusters)

        return menu_bar

    def create_controls(self, parent):
        """Crear barra de parámetros (spinboxes) e info."""
        control_frame = tk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Threshold
        tk.Label(control_frame, text="Threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.threshold_var = tk.IntVar(value=self.default_th)
        ttk.Spinbox(control_frame, from_=1, to=1000, textvariable=self.threshold_var, width=10).pack(side=tk.LEFT, padx=(0, 15))

        # Tamaño mínimo
        tk.Label(control_frame, text="Min Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.min_size_var = tk.IntVar(value=20)
        ttk.Spinbox(control_frame, from_=1, to=500, textvariable=self.min_size_var, width=8).pack(side=tk.LEFT, padx=(0, 15))

        # Tamaño máximo
        tk.Label(control_frame, text="Max Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.max_size_var = tk.IntVar(value=200)
        ttk.Spinbox(control_frame, from_=1, to=1000, textvariable=self.max_size_var, width=8).pack(side=tk.LEFT, padx=(0, 15))

        self.info_label = tk.Label(control_frame, text="Usa el menú para procesar clusters")
        self.info_label.pack(side=tk.LEFT, padx=10)
    
    def create_selection_panel(self):
        """Crear panel de selección lateral"""
        # Título
        title_label = tk.Label(self.selection_frame, text="Clusters Seleccionados", 
                              font=("Arial", 12, "bold"))
        title_label.pack(pady=10)
        
        # Lista de seleccionados
        self.selected_listbox = tk.Listbox(self.selection_frame, height=15)
        self.selected_listbox.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # Frame para botones del panel
        button_frame = tk.Frame(self.selection_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Botón para remover seleccionado
        remove_btn = tk.Button(button_frame, text="Remover", 
                              command=self.remove_selected_from_list)
        remove_btn.pack(fill=tk.X, pady=2)
        
        # Separador
        ttk.Separator(self.selection_frame, orient='horizontal').pack(fill='x', padx=10, pady=5)
        
        # Controles de selección por región
        region_label = tk.Label(self.selection_frame, text="Selección por Región",
                               font=("Arial", 10, "bold"))
        region_label.pack(pady=(5, 2))
        
        region_info = tk.Label(self.selection_frame, 
                              text="Click derecho + arrastrar\nen el gráfico de clusters",
                              font=("Arial", 8), fg='gray',
                              justify=tk.CENTER)
        region_info.pack(pady=(0, 5))
        
        mode_frame = tk.Frame(self.selection_frame)
        mode_frame.pack(fill=tk.X, padx=10)
        
        self.selection_mode_var = tk.StringVar(value='add')
        tk.Radiobutton(mode_frame, text="Añadir", variable=self.selection_mode_var,
                       value='add', command=self._update_selection_mode).pack(side=tk.LEFT, expand=True)
        tk.Radiobutton(mode_frame, text="Quitar", variable=self.selection_mode_var,
                       value='remove', command=self._update_selection_mode).pack(side=tk.LEFT, expand=True)
        
        # Información
        self.selection_info = tk.Label(self.selection_frame, 
                                     text="Click en clusters\npara seleccionar",
                                     justify=tk.CENTER)
        self.selection_info.pack(pady=10)
    
    def update_display(self):
        """Actualizar la visualización"""
        # Limpiar frame anterior
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        # Crear figura
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Imagen de variabilidad original
        im1 = ax1.imshow(self.var_im, cmap='viridis')
        ax1.set_title(f'{self.method_name}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Imagen procesada
        im2 = ax2.imshow(self.processed_image, cmap='viridis')
        ax2.set_title('Imagen Procesada (Unsharp + Filter)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Binarización (si existe)
        if self.res_labels is not None:
            ax3.imshow(self.res_labels, cmap='gray')
            ax3.set_title(f'Binarización (th={self.threshold_var.get()})')
        else:
            ax3.text(0.5, 0.5, 'Aplica binarización', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Binarización')
        ax3.axis('off')
        
        # 4. Clusters (si existen) - CON INTERACTIVIDAD CORREGIDA
        if self.final_clusters is not None:
            # Limpiar scatter objects anteriores
            self.scatter_objects = []
            
            # Generar colores aleatorios si no los tenemos o cambió la cantidad
            if len(self.cluster_colors) != len(self.final_clusters):
                self.cluster_colors = self._generate_cluster_colors(len(self.final_clusters))
            
            for i, cl in enumerate(self.final_clusters):
                if len(cl) > 0:
                    cl_array = np.array(cl)
                    
                    # Rojo si seleccionado, color aleatorio si no
                    if i in self.selected_clusters:
                        color = 'red'
                        alpha = 1.0
                        zorder = 3
                    else:
                        color = self.cluster_colors[i]
                        alpha = 0.7
                        zorder = 2
                    
                    scatter = ax4.scatter(cl_array[:, 1], np.array(self.var_im).shape[0] - cl_array[:, 0], 
                                        marker='o', s=2, alpha=alpha, color=color,
                                        picker=True, pickradius=5, zorder=zorder)
                    
                    # Almacenar referencia para clicks
                    self.scatter_objects.append((scatter, i))
            
            ax4.set_title(f'Clusters Finales ({len(self.final_clusters)} encontrados) - Click para seleccionar')
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            
        else:
            ax4.text(0.5, 0.5, 'Procesa clusters', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Clusters - Click para seleccionar')
            
        plt.tight_layout()
        
        # Agregar a la ventana
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # CONECTAR EVENTO DE CLICK - MUY IMPORTANTE
        if self.final_clusters is not None:
            canvas.mpl_connect('pick_event', self.on_cluster_click)
            
            # Conectar rectangle selector para selección por región (botón derecho)
            self.rect_selector = RectangleSelector(
                ax4, self._on_region_selected,
                useblit=True,
                button=[3],  # Solo botón derecho del mouse
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=False,
                props=dict(facecolor='yellow', edgecolor='black', alpha=0.3, linewidth=1.5)
            )
        
        # Guardar referencia para save
        self.current_fig = fig
        self.current_canvas = canvas
    
    def _apply_binarization(self):
        """Binarizar la imagen procesada con el threshold actual."""
        th = self.threshold_var.get()
        self.res_labels = apply_binarization(self.processed_image, th)

    def _find_clusters(self):
        """Encontrar componentes conectados en la imagen binarizada."""
        pixels = extract_pixels_from_binary(self.res_labels)
        self.res_clusters = list(neighboring_groups(pixels))

    def process_clusters_basic(self):
        """Binarizar, encontrar componentes conectados y filtrar por tamaño."""
        min_size = self.min_size_var.get()
        max_size = self.max_size_var.get()
        th = self.threshold_var.get()

        self._apply_binarization()
        self._find_clusters()

        self.selected_clusters = []
        self.cluster_colors = []

        self.final_clusters = [cl for cl in self.res_clusters if min_size <= len(cl) <= max_size]

        self.info_label.config(
            text=f"Básico: {len(self.final_clusters)} clusters (th={th}, tamaño {min_size}–{max_size})"
        )
        self.update_display()
        self.update_selection_list()

    def process_clusters_advanced(self):
        """Binarizar, encontrar componentes conectados y procesar con detección de picos 3D."""
        min_size = self.min_size_var.get()
        max_size = self.max_size_var.get()
        th = self.threshold_var.get()

        self._apply_binarization()
        self._find_clusters()

        try:
            self.final_clusters, _ = process_clusters_advanced(
                self.var_im, self.res_clusters, min_size, max_size
            )
            self.info_label.config(
                text=f"Avanzado: {len(self.final_clusters)} clusters (th={th}, picos 3D)"
            )
            self.update_display()
        except Exception as e:
            messagebox.showwarning("Advertencia", f"Error en procesamiento avanzado: {e}\nUsando método básico como respaldo")
    
    def decompose_clusters(self):
        """Descomponer clusters grandes usando KMeans"""
        if self.res_clusters is None:
            messagebox.showwarning("Advertencia", "Primero procesa los clusters con 'Procesar Cluster (Básico)' o '(Avanzado)'")
            return
        
        min_size = self.min_size_var.get()
        max_size = self.max_size_var.get()
        
        # Contar clusters grandes antes de descomponer
        large_count = sum(1 for cl in self.res_clusters if len(cl) > max_size)
        
        # Descomponer
        self.res_clusters = decompose_large_clusters(
            self.res_clusters, self.var_im, min_size, max_size
        )
        
        # Limpiar selecciones y clusters finales
        self.selected_clusters = []
        self.final_clusters = None
        self.cluster_colors = []  # Regenerar colores
        
        self.info_label.config(
            text=f"Descompuestos {large_count} clusters grandes → {len(self.res_clusters)} clusters totales"
        )
        self.update_display()
        self.update_selection_list()
    
    def select_all_clusters(self):
        """Seleccionar todos los clusters"""
        if self.final_clusters is None or len(self.final_clusters) == 0:
            messagebox.showwarning("Advertencia", "Primero procesa los clusters")
            return
        
        self.selected_clusters = list(range(len(self.final_clusters)))
        self.update_cluster_colors()
        self.update_selection_list()
    
    def _update_selection_mode(self):
        """Actualizar modo de selección por región"""
        self.selection_mode = self.selection_mode_var.get()
    
    def _on_region_selected(self, eclick, erelease):
        """Callback para selección por región rectangular"""
        if self.final_clusters is None:
            return
        
        # Obtener coordenadas del rectángulo en el espacio del gráfico
        x_min = min(eclick.xdata, erelease.xdata)
        x_max = max(eclick.xdata, erelease.xdata)
        y_min = min(eclick.ydata, erelease.ydata)
        y_max = max(eclick.ydata, erelease.ydata)
        
        img_height = np.array(self.var_im).shape[0]
        
        # Encontrar clusters cuyo centroide cae dentro del rectángulo
        changed = False
        for i, cl in enumerate(self.final_clusters):
            if len(cl) == 0:
                continue
            cl_array = np.array(cl)
            # Calcular centroide en coordenadas del gráfico (igual que en scatter)
            centroid_x = np.mean(cl_array[:, 1])
            centroid_y = img_height - np.mean(cl_array[:, 0])
            
            if x_min <= centroid_x <= x_max and y_min <= centroid_y <= y_max:
                if self.selection_mode == 'add' and i not in self.selected_clusters:
                    self.selected_clusters.append(i)
                    changed = True
                elif self.selection_mode == 'remove' and i in self.selected_clusters:
                    self.selected_clusters.remove(i)
                    changed = True
        
        if changed:
            self.update_cluster_colors()
            self.update_selection_list()
    
    def on_cluster_click(self, event):
        """FUNCIÓN CLAVE - Manejar click en cluster (CON PROTECCIÓN ANTI-AUTO-CLICK)"""
        
        # PROTECCIÓN: Solo procesar clicks reales del usuario
        # Si no hay botón presionado, es un evento automático - ignorar
        if not hasattr(event, 'mouseevent') or event.mouseevent is None:
            return
            
        # PROTECCIÓN: Solo clicks del botón izquierdo del mouse
        if hasattr(event.mouseevent, 'button') and event.mouseevent.button != 1:
            return
        
        
        # Encontrar qué cluster fue clickeado
        clicked_cluster = None
        for scatter, cluster_idx in self.scatter_objects:
            if event.artist == scatter:
                clicked_cluster = cluster_idx
                break
        
        if clicked_cluster is not None:
            if clicked_cluster in self.selected_clusters:
                # Deseleccionar
                self.selected_clusters.remove(clicked_cluster)
            else:
                # Seleccionar
                self.selected_clusters.append(clicked_cluster)
            
            # Actualizar visualización y lista
            self.update_cluster_colors()
            self.update_selection_list()
    
    def update_cluster_colors(self):
        """Actualizar colores de clusters según selección: rojo=seleccionado, color aleatorio=no seleccionado.
        También muestra el ID del cluster sobre los clusters seleccionados."""
        # Limpiar anotaciones anteriores
        if hasattr(self, '_cluster_annotations'):
            for ann in self._cluster_annotations:
                try:
                    ann.remove()
                except:
                    pass
        self._cluster_annotations = []
        
        img_height = np.array(self.var_im).shape[0]
        
        for scatter, cluster_idx in self.scatter_objects:
            if cluster_idx in self.selected_clusters:
                scatter.set_color('red')
                scatter.set_alpha(1.0)
                scatter.set_zorder(3)
                
                # Mostrar ID del cluster en el centroide
                if self.final_clusters is not None and cluster_idx < len(self.final_clusters):
                    cl = self.final_clusters[cluster_idx]
                    if len(cl) > 0:
                        cl_array = np.array(cl)
                        cx = np.mean(cl_array[:, 1])
                        cy = img_height - np.mean(cl_array[:, 0])
                        ax = scatter.axes
                        ann = ax.annotate(
                            str(cluster_idx), (cx, cy),
                            fontsize=7, fontweight='bold', color='white',
                            ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.8),
                            zorder=5
                        )
                        self._cluster_annotations.append(ann)
            else:
                if cluster_idx < len(self.cluster_colors):
                    scatter.set_color(self.cluster_colors[cluster_idx])
                else:
                    scatter.set_color('blue')
                scatter.set_alpha(0.7)
                scatter.set_zorder(2)
        
        # Refrescar canvas
        if hasattr(self, 'current_canvas'):
            self.current_canvas.draw()
    
    def update_selection_list(self):
        """Actualizar lista de clusters seleccionados"""
        self.selected_listbox.delete(0, tk.END)
        
        for cluster_idx in sorted(self.selected_clusters):
            cluster_size = len(self.final_clusters[cluster_idx]) if self.final_clusters else 0
            self.selected_listbox.insert(tk.END, f"Cluster {cluster_idx} ({cluster_size} pts)")
        
        # Actualizar información
        count = len(self.selected_clusters)
        self.selection_info.config(text=f"{count} clusters\nseleccionados")
    
    def remove_selected_from_list(self):
        """Remover cluster seleccionado de la lista"""
        selection = self.selected_listbox.curselection()
        if selection:
            # Obtener índice del cluster desde el texto
            item_text = self.selected_listbox.get(selection[0])
            cluster_idx = int(item_text.split()[1])  # "Cluster X (...)" -> X
            
            if cluster_idx in self.selected_clusters:
                self.selected_clusters.remove(cluster_idx)
                self.update_cluster_colors()
                self.update_selection_list()
    
    def clear_selection(self):
        """Limpiar toda la selección"""
        self.selected_clusters = []
        self.update_cluster_colors()
        self.update_selection_list()
    
    def save_selected_npy(self):
        """Guardar clusters seleccionados como archivo .npy"""
        if not self.selected_clusters:
            messagebox.showwarning("Advertencia", "No hay clusters seleccionados")
            return
        
        if self.final_clusters is None:
            messagebox.showwarning("Advertencia", "Primero procesa los clusters")
            return
        
        filename = asksaveasfilename(
            initialfile='clusters_seleccionados.npy',
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All Files", "*.*")]
        )
        
        if filename:
            # Snapshot plain data needed by the worker before dispatching to
            # the background thread (self.selected_clusters/self.final_clusters
            # are plain lists, safe to read once here on the main thread).
            cluster_indices = sorted(self.selected_clusters)
            img_array = self.img_array
            final_clusters = self.final_clusters

            def worker():
                # Build 2D array (n_frames, 1 + n_clusters):
                #   column 0 = frame indices (skipped by _load_data with [:,1:])
                #   columns 1+ = mean time series for each selected cluster
                time_series_list = []
                for cluster_idx in cluster_indices:
                    ts = extract_time_series(img_array, final_clusters[cluster_idx])
                    time_series_list.append(ts)

                n_frames = len(time_series_list[0])
                frame_col = np.arange(n_frames, dtype=float).reshape(-1, 1)
                ts_matrix = np.column_stack(time_series_list).astype(float)
                data = np.hstack([frame_col, ts_matrix])

                np.save(filename, data)
                return filename

            run_save_with_progress(
                self.window,
                title="Guardando clusters",
                message="Guardando clusters seleccionados...",
                worker_fn=worker,
                success_message=lambda fn: f"Clusters guardados en {fn}")
    
    def _on_window_close(self):
        """Cerrar la ventana de análisis y liberar recursos"""
        plt.close('all')
        self.window.destroy()
    
    def use_selected_clusters(self):
        """Usar clusters seleccionados para análisis"""
        if not self.selected_clusters:
            messagebox.showwarning("Advertencia", "No hay clusters seleccionados")
            return
        
        # Crear ventana de análisis avanzado
        self.show_correlation_analysis()
    
    def show_correlation_analysis(self):
        """Mostrar análisis de correlaciones de clusters seleccionados"""
        # Crear ventana de análisis
        corr_window = tk.Toplevel(self.window)
        corr_window.title("Análisis de Correlaciones - Clusters Seleccionados")
        corr_window.geometry("1200x800")
        
        # Frame principal
        main_frame = tk.Frame(corr_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de controles superiores
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Información
        info_text = f"Analizando correlaciones entre {len(self.selected_clusters)} clusters seleccionados"
        info_label = tk.Label(control_frame, text=info_text, font=("Arial", 12, "bold"))
        info_label.pack(pady=5)
        
        # Botones de correlación
        corr_button_frame = tk.Frame(control_frame)
        corr_button_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(corr_button_frame, text="Tipo de Correlación:").pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(corr_button_frame, text="Pearson", 
                 command=lambda: self.calculate_correlation('pearson')).pack(side=tk.LEFT, padx=5)
        tk.Button(corr_button_frame, text="Kendall", 
                 command=lambda: self.calculate_correlation('kendall')).pack(side=tk.LEFT, padx=5)
        tk.Button(corr_button_frame, text="Spearman", 
                 command=lambda: self.calculate_correlation('spearman')).pack(side=tk.LEFT, padx=5)
        
        # Separador
        tk.Frame(corr_button_frame, height=2, bg="gray").pack(side=tk.LEFT, fill=tk.X, padx=20)
        
        # Botones de exportación
        tk.Button(corr_button_frame, text="Exportar Series Temporales", 
                 command=self.export_time_series).pack(side=tk.LEFT, padx=5)
        tk.Button(corr_button_frame, text="Exportar Coordenadas", 
                 command=self.export_coordinates).pack(side=tk.LEFT, padx=5)
        tk.Button(corr_button_frame, text="Generar Reporte", 
                 command=self.generate_report).pack(side=tk.LEFT, padx=5)
        
        # Frame para visualización
        self.corr_viz_frame = tk.Frame(main_frame)
        self.corr_viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Mostrar series temporales iniciales
        self.show_initial_time_series()
    
    def show_initial_time_series(self):
        """Mostrar series temporales de clusters seleccionados"""
        # Limpiar frame
        for widget in self.corr_viz_frame.winfo_children():
            widget.destroy()
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Extraer series temporales de clusters seleccionados
        self.time_series_data = []
        self.cluster_labels = []
        
        for cluster_idx in self.selected_clusters:
            if len(self.final_clusters[cluster_idx]) > 0:
                time_series = extract_time_series(self.img_array, self.final_clusters[cluster_idx])
                self.time_series_data.append(time_series)
                self.cluster_labels.append(f'Cluster {cluster_idx}')
        
        # Graficar series temporales individuales
        for i, (ts, label) in enumerate(zip(self.time_series_data, self.cluster_labels)):
            ax1.plot(ts, label=label, alpha=0.8)
        
        ax1.set_title('Series Temporales de Clusters Seleccionados')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Intensidad Promedio')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Mapa de calor de correlación preliminar
        if len(self.time_series_data) > 1:
            import pandas as pd
            df = pd.DataFrame(self.time_series_data).T
            df.columns = self.cluster_labels
            corr_matrix = df.corr()
            
            im = ax2.imshow(corr_matrix.values, cmap='RdBu', vmin=-1, vmax=1)
            ax2.set_xticks(range(len(self.cluster_labels)))
            ax2.set_yticks(range(len(self.cluster_labels)))
            ax2.set_xticklabels(self.cluster_labels, rotation=45)
            ax2.set_yticklabels(self.cluster_labels)
            ax2.set_title('Matriz de Correlación Preliminar (Pearson)')
            
            # Agregar valores en el mapa
            for i in range(len(self.cluster_labels)):
                for j in range(len(self.cluster_labels)):
                    text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontweight="bold")
            
            plt.colorbar(im, ax=ax2)
        else:
            ax2.text(0.5, 0.5, 'Selecciona al menos 2 clusters\npara análisis de correlación', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Matriz de Correlación')
        
        plt.tight_layout()
        
        # Agregar a la ventana
        canvas = FigureCanvasTkAgg(fig, master=self.corr_viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Guardar referencia
        self.current_corr_fig = fig
    
    def calculate_correlation(self, method):
        """Calcular correlación usando el método especificado"""
        if len(self.time_series_data) < 2:
            messagebox.showwarning("Advertencia", "Necesitas al menos 2 clusters seleccionados")
            return
        
        import pandas as pd
        
        # Crear DataFrame con las series temporales
        df = pd.DataFrame(self.time_series_data).T
        df.columns = self.cluster_labels
        
        # Calcular correlación según el método
        if method == 'pearson':
            corr_matrix = df.corr(method='pearson')
        elif method == 'kendall':
            corr_matrix = df.corr(method='kendall')
        elif method == 'spearman':
            corr_matrix = df.corr(method='spearman')
        
        # Actualizar visualización
        self.update_correlation_display(corr_matrix, method.capitalize())
    
    def update_correlation_display(self, corr_matrix, method_name):
        """Actualizar la visualización de correlación"""
        # Limpiar frame
        for widget in self.corr_viz_frame.winfo_children():
            widget.destroy()
        
        # Crear nueva figura
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Series temporales
        for i, (ts, label) in enumerate(zip(self.time_series_data, self.cluster_labels)):
            ax1.plot(ts, label=label, alpha=0.8)
        ax1.set_title('Series Temporales de Clusters Seleccionados')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Intensidad')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Matriz de correlación
        im2 = ax2.imshow(corr_matrix.values, cmap='RdBu', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(self.cluster_labels)))
        ax2.set_yticks(range(len(self.cluster_labels)))
        ax2.set_xticklabels(self.cluster_labels, rotation=45)
        ax2.set_yticklabels(self.cluster_labels)
        ax2.set_title(f'Matriz de Correlación ({method_name})')
        
        # Agregar valores en la matriz
        for i in range(len(self.cluster_labels)):
            for j in range(len(self.cluster_labels)):
                text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight="bold")
        
        plt.colorbar(im2, ax=ax2)
        
        # 3. Histograma de correlaciones
        correlations = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        ax3.hist(correlations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Distribución de Correlaciones')
        ax3.set_xlabel('Valor de Correlación')
        ax3.set_ylabel('Frecuencia')
        ax3.grid(True, alpha=0.3)
        
        # 4. Mapa de correlación reordenado
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
            
            # Crear matriz de distancias
            distance_matrix = 1 - np.abs(corr_matrix.values)
            condensed_distances = squareform(distance_matrix)
            
            # Clustering jerárquico
            linkage_matrix = linkage(condensed_distances, method='ward')
            dendro = dendrogram(linkage_matrix, labels=self.cluster_labels, ax=ax4)
            ax4.set_title('Dendrograma de Clusters')
            ax4.set_xlabel('Clusters')
            ax4.set_ylabel('Distancia')
        except Exception as e:
            ax4.text(0.5, 0.5, 'Error al generar dendrograma', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Dendrograma')
        
        plt.tight_layout()
        
        # Agregar a la ventana
        canvas = FigureCanvasTkAgg(fig, master=self.corr_viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Guardar para exportación
        self.current_corr_fig = fig
        self.current_corr_matrix = corr_matrix
        self.current_method = method_name
    
    def export_time_series(self):
        """Exportar series temporales a CSV"""
        if not hasattr(self, 'time_series_data') or not self.time_series_data:
            messagebox.showwarning("Advertencia", "No hay datos para exportar")
            return
        
        filename = asksaveasfilename(
            initialfile='series_temporales_clusters.csv',
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All Files", "*.*")]
        )
        
        if filename:
            time_series_data = self.time_series_data
            cluster_labels = self.cluster_labels

            def worker():
                import pandas as pd
                df = pd.DataFrame(time_series_data).T
                df.columns = cluster_labels
                df.index.name = 'Frame'
                df.to_csv(filename)
                return filename

            run_save_with_progress(
                self.window,
                title="Exportando series temporales",
                message="Exportando series temporales...",
                worker_fn=worker,
                success_message=lambda fn: f"Series temporales exportadas a {fn}")
    
    def export_coordinates(self):
        """Exportar coordenadas de clusters seleccionados"""
        filename = asksaveasfilename(
            initialfile='coordenadas_clusters.txt',
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filename:
            # Tk variables must be read on the main thread; snapshot before
            # dispatching to the background worker.
            threshold = self.threshold_var.get()
            method_name = self.method_name
            selected_clusters = self.selected_clusters
            final_clusters = self.final_clusters

            def worker():
                with open(filename, 'w') as f:
                    f.write("# Coordenadas de Clusters Seleccionados\n")
                    f.write(f"# Método de variabilidad: {method_name}\n")
                    f.write(f"# Threshold usado: {threshold}\n")
                    f.write(f"# Total de clusters: {len(selected_clusters)}\n\n")

                    for cluster_idx in selected_clusters:
                        cluster_points = final_clusters[cluster_idx]
                        f.write(f"Cluster {cluster_idx} ({len(cluster_points)} puntos):\n")
                        for point in cluster_points:
                            f.write(f"{point[0]},{point[1]}\n")
                        f.write("\n")
                return filename

            run_save_with_progress(
                self.window,
                title="Exportando coordenadas",
                message="Exportando coordenadas...",
                worker_fn=worker,
                success_message=lambda fn: f"Coordenadas exportadas a {fn}")
    
    def generate_report(self):
        """Generar reporte completo del análisis"""
        if not hasattr(self, 'current_corr_matrix'):
            messagebox.showwarning("Advertencia", "Primero calcula una correlación")
            return
        
        filename = asksaveasfilename(
            initialfile='reporte_analisis_clusters.txt',
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filename:
            # Tk variables must be read on the main thread; snapshot before
            # dispatching to the background worker. Remaining values are
            # plain attributes, safe to read once here as well.
            threshold = self.threshold_var.get()
            min_size = self.min_size_var.get()
            max_size = self.max_size_var.get()
            method_name = self.method_name
            selected_clusters = self.selected_clusters
            final_clusters = self.final_clusters
            current_method = self.current_method
            current_corr_matrix = self.current_corr_matrix

            def worker():
                with open(filename, 'w') as f:
                    f.write("REPORTE DE ANÁLISIS DE CLUSTERS\n")
                    f.write("=" * 50 + "\n\n")

                    f.write(f"Método de variabilidad: {method_name}\n")
                    f.write(f"Threshold usado: {threshold}\n")
                    f.write(f"Rango de tamaños: {min_size} - {max_size}\n")
                    f.write(f"Clusters seleccionados: {len(selected_clusters)}\n")
                    f.write(f"Método de correlación: {current_method}\n\n")

                    f.write("DETALLES DE CLUSTERS:\n")
                    f.write("-" * 25 + "\n")
                    for cluster_idx in selected_clusters:
                        cluster_size = len(final_clusters[cluster_idx])
                        f.write(f"Cluster {cluster_idx}: {cluster_size} píxeles\n")

                    f.write(f"\nMATRIZ DE CORRELACIÓN ({current_method}):\n")
                    f.write("-" * 35 + "\n")
                    f.write(current_corr_matrix.to_string())
                    f.write("\n\n")

                    f.write("ESTADÍSTICAS DE CORRELACIÓN:\n")
                    f.write("-" * 30 + "\n")
                    correlations = current_corr_matrix.values[np.triu_indices_from(current_corr_matrix.values, k=1)]
                    f.write(f"Correlación promedio: {np.mean(correlations):.3f}\n")
                    f.write(f"Correlación máxima: {np.max(correlations):.3f}\n")
                    f.write(f"Correlación mínima: {np.min(correlations):.3f}\n")
                    f.write(f"Desviación estándar: {np.std(correlations):.3f}\n")
                return filename

            run_save_with_progress(
                self.window,
                title="Generando reporte",
                message="Generando reporte...",
                worker_fn=worker,
                success_message=lambda fn: f"Reporte generado en {fn}")
    
    def show_3d_surface(self):
        """Mostrar visualización 3D de la superficie de variabilidad"""
        from mpl_toolkits.mplot3d import Axes3D
        
        # Crear ventana para visualización 3D
        window_3d = tk.Toplevel(self.window)
        window_3d.title(f"Superficie 3D - {self.method_name}")
        window_3d.geometry("800x600")
        
        # Submuestrear la imagen para que la visualización sea más rápida
        step = max(1, min(self.var_im.shape) // 100)  # Máximo 100 puntos por dimensión
        var_im_sub = self.var_im[::step, ::step]
        
        # Crear meshgrid
        x = np.arange(0, var_im_sub.shape[1])
        y = np.arange(0, var_im_sub.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = var_im_sub
        
        # Crear figura 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Graficar superficie
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Configurar ejes
        ax.set_xlabel('X (píxeles)')
        ax.set_ylabel('Y (píxeles)')
        ax.set_zlabel(f'{self.method_name}')
        ax.set_title(f'Superficie de Variabilidad - {self.method_name}')
        
        # Agregar barra de color
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=self.method_name)
        
        # Agregar a la ventana
        canvas = FigureCanvasTkAgg(fig, master=window_3d)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame para controles
        control_frame = tk.Frame(window_3d)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Botón para guardar
        def save_3d():
            filename = asksaveasfilename(
                initialfile=f'{self.method_name.replace(" ", "_")}_3D.png',
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                           ("SVG files", "*.svg"), ("EPS files", "*.eps"),
                           ("All Files", "*.*")]
            )
            if filename:
                def worker():
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    return filename

                run_save_with_progress(
                    self.window,
                    title="Guardando imagen 3D",
                    message="Guardando imagen 3D...",
                    worker_fn=worker,
                    success_message=lambda fn: f"Imagen 3D guardada en {fn}")
        
        tk.Button(control_frame, text="Guardar Imagen 3D", command=save_3d).pack(side=tk.LEFT, padx=10)
        tk.Label(control_frame, text="Usa el mouse para rotar la vista").pack(side=tk.LEFT, padx=10)
    
    def save_image(self):
        """Guardar la imagen actual"""
        if hasattr(self, 'current_fig'):
            filename = asksaveasfilename(
                initialfile=f'{self.method_name.replace(" ", "_")}_analysis.png',
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                           ("SVG files", "*.svg"), ("EPS files", "*.eps"),
                           ("All Files", "*.*")]
            )
            if filename:
                current_fig = self.current_fig

                def worker():
                    current_fig.savefig(filename, dpi=300, bbox_inches='tight')
                    return filename

                run_save_with_progress(
                    self.window,
                    title="Guardando imagen",
                    message="Guardando imagen...",
                    worker_fn=worker,
                    success_message=lambda fn: f"Imagen guardada en {fn}")

def show_variability_analysis(img_array, method, main_window):
    """
    Función principal para mostrar el análisis de variabilidad completo.
    """
    if img_array is None or len(img_array) == 0:
        messagebox.showwarning("Advertencia", "No hay imagen cargada")
        return
    
    # Crear la ventana de análisis
    analysis_window = VariabilityAnalysisWindow(img_array, method, main_window)

def get_variability_methods():
    """
    Retorna lista de métodos disponibles para el menú.
    """
    return [
        "Rango",
        "Varianza Poblacional", 
        "Varianza Muestral",
        "Desviación Estándar Poblacional",
        "Desviación Estándar Muestral", 
        "Coeficiente de Variación",
        "Rango Intercuartílico (IQR)"
    ]
