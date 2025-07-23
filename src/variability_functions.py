import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from skimage.filters import unsharp_mask
from tkinter.filedialog import asksaveasfilename

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
                print(f"Error procesando cluster: {e}")
                final_cl.append(cl)
                continue
        
        return final_cl, clusters_min_size
        
    except Exception as e:
        print(f"Error en process_clusters_advanced: {e}")
        # Fallback al método básico
        final_cl = []
        for cl in res_clusters:
            if min_size <= len(cl) <= max_size:
                final_cl.append(cl)
def extract_time_series(img_array, cluster_points):
    """
    Extrae serie temporal de un cluster específico.
    """
    final_array = np.array(cluster_points)
    sel_data_final = img_array[:, final_array[:,0], final_array[:,1]]
    sel_data_final_mean = np.mean(sel_data_final, axis=1)
    return sel_data_final_mean

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
        
        # Variables para análisis
        self.time_series_data = []
        self.cluster_labels = []
        
        self.create_window()
        self.update_display()
    
    def create_window(self):
        """Crear la ventana principal"""
        self.window = tk.Toplevel(self.main_window)
        self.window.title(f"Análisis Completo - {self.method_name}")
        self.window.geometry("1400x800")
        
        # Frame principal
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de controles
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
    
    def create_controls(self, parent):
        """Crear controles de la interfaz"""
        control_frame = tk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 1: Threshold
        row1 = tk.Frame(control_frame)
        row1.pack(fill=tk.X, pady=2)
        
        tk.Label(row1, text="Threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.threshold_var = tk.IntVar(value=self.default_th)
        threshold_spinbox = ttk.Spinbox(row1, from_=1, to=1000, textvariable=self.threshold_var, width=10)
        threshold_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(row1, text="Aplicar Binarización", command=self.apply_binarization).pack(side=tk.LEFT, padx=5)
        tk.Button(row1, text="Encontrar Clusters", command=self.find_clusters).pack(side=tk.LEFT, padx=5)
        
        # Row 2: Cluster controls
        row2 = tk.Frame(control_frame)
        row2.pack(fill=tk.X, pady=2)
        
        tk.Label(row2, text="Min Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.min_size_var = tk.IntVar(value=20)
        ttk.Spinbox(row2, from_=1, to=500, textvariable=self.min_size_var, width=8).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(row2, text="Max Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.max_size_var = tk.IntVar(value=200)
        ttk.Spinbox(row2, from_=1, to=1000, textvariable=self.max_size_var, width=8).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(row2, text="Procesar Clusters (Básico)", command=self.process_clusters_basic).pack(side=tk.LEFT, padx=5)
        tk.Button(row2, text="Procesar Clusters (Avanzado)", command=self.process_clusters_advanced).pack(side=tk.LEFT, padx=5)
        tk.Button(row2, text="Mostrar Serie Temporal", command=self.show_time_series).pack(side=tk.LEFT, padx=5)
        
        # Row 3: Save and selection
        row3 = tk.Frame(control_frame)
        row3.pack(fill=tk.X, pady=2)
        
        tk.Button(row3, text="Guardar Imagen", command=self.save_image).pack(side=tk.LEFT, padx=5)
        tk.Button(row3, text="Limpiar Selección", command=self.clear_selection).pack(side=tk.LEFT, padx=5)
        tk.Button(row3, text="Usar Seleccionados", command=self.use_selected_clusters).pack(side=tk.LEFT, padx=5)
        
        self.info_label = tk.Label(row3, text="Selecciona threshold y aplica binarización")
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
        button_frame.pack(fill=tk.X, pady=10)
        
        # Botón para remover seleccionado
        remove_btn = tk.Button(button_frame, text="Remover", 
                              command=self.remove_selected_from_list)
        remove_btn.pack(fill=tk.X, pady=2)
        
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
            
            for i, cl in enumerate(self.final_clusters):
                if len(cl) > 0:
                    cl_array = np.array(cl)
                    
                    # TODOS empiezan en AZUL, solo se vuelven ROJOS cuando se seleccionan
                    color = 'red' if i in self.selected_clusters else 'blue'
                    alpha = 1.0 if i in self.selected_clusters else 0.7
                    
                    scatter = ax4.scatter(cl_array[:, 1], np.array(self.var_im).shape[0] - cl_array[:, 0], 
                                        marker='o', s=2, alpha=alpha, color=color,
                                        picker=True, pickradius=5)
                    
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
        
        # Guardar referencia para save
        self.current_fig = fig
        self.current_canvas = canvas
    
    def apply_binarization(self):
        """Aplicar binarización con el threshold actual"""
        th = self.threshold_var.get()
        self.res_labels = apply_binarization(self.processed_image, th)
        self.info_label.config(text=f"Binarización aplicada con threshold {th}")
        self.update_display()
    
    def find_clusters(self):
        """Encontrar clusters básicos"""
        if self.res_labels is None:
            messagebox.showwarning("Advertencia", "Primero aplica la binarización")
            return
        
    def process_clusters_basic(self):
        """Procesar clusters - versión básica (filtrado por tamaño)"""
        if self.res_clusters is None:
            messagebox.showwarning("Advertencia", "Primero encuentra los clusters básicos")
            return
        
        min_size = self.min_size_var.get()
        max_size = self.max_size_var.get()
        
        # LIMPIAR selecciones anteriores al procesar
        self.selected_clusters = []
        
        # Versión simplificada sin detección de picos
        self.final_clusters = []
        for cl in self.res_clusters:
            if min_size <= len(cl) <= max_size:
                self.final_clusters.append(cl)
        
        self.info_label.config(text=f"Procesados (Básico): {len(self.final_clusters)} clusters finales")
        self.update_display()
        
        # Actualizar lista de selección (debería estar vacía)
        self.update_selection_list()
    
    def process_clusters_advanced(self):
        """Procesar clusters - versión avanzada con detección de picos 3D"""
        if self.res_clusters is None:
            messagebox.showwarning("Advertencia", "Primero encuentra los clusters básicos")
            return
        
        min_size = self.min_size_var.get()
        max_size = self.max_size_var.get()
        
        try:
            # Usar algoritmo avanzado original de Jose
            self.final_clusters, clusters_min_size = process_clusters_advanced(
                self.var_im, self.res_clusters, min_size, max_size
            )
            
            self.info_label.config(text=f"Procesados (Avanzado): {len(self.final_clusters)} clusters finales con detección de picos 3D")
            self.update_display()
            
        except Exception as e:
            # Si falla, usar método básico como respaldo
            messagebox.showwarning("Advertencia", f"Error en procesamiento avanzado: {e}\nUsando método básico como respaldo")
    def find_clusters(self):
        """Encontrar clusters básicos"""
        if self.res_labels is None:
            messagebox.showwarning("Advertencia", "Primero aplica la binarización")
            return
        
        pixels = extract_pixels_from_binary(self.res_labels)
        self.res_clusters = list(neighboring_groups(pixels))
        self.info_label.config(text=f"Encontrados {len(self.res_clusters)} clusters básicos")
    
    def on_cluster_click(self, event):
        """FUNCIÓN CLAVE - Manejar click en cluster (CON PROTECCIÓN ANTI-AUTO-CLICK)"""
        
        # PROTECCIÓN: Solo procesar clicks reales del usuario
        # Si no hay botón presionado, es un evento automático - ignorar
        if not hasattr(event, 'mouseevent') or event.mouseevent is None:
            return
            
        # PROTECCIÓN: Solo clicks del botón izquierdo del mouse
        if hasattr(event.mouseevent, 'button') and event.mouseevent.button != 1:
            return
        
        print(f"Click detectado en: {event.artist}")  # Debug
        
        # Encontrar qué cluster fue clickeado
        clicked_cluster = None
        for scatter, cluster_idx in self.scatter_objects:
            if event.artist == scatter:
                clicked_cluster = cluster_idx
                print(f"Cluster clickeado: {cluster_idx}")  # Debug
                break
        
        if clicked_cluster is not None:
            if clicked_cluster in self.selected_clusters:
                # Deseleccionar
                self.selected_clusters.remove(clicked_cluster)
                print(f"Deseleccionado cluster {clicked_cluster}")
            else:
                # Seleccionar
                self.selected_clusters.append(clicked_cluster)
                print(f"Seleccionado cluster {clicked_cluster}")
            
            # Actualizar visualización y lista
            self.update_cluster_colors()
            self.update_selection_list()
        else:
            print("No se pudo identificar el cluster clickeado")
    
    def update_cluster_colors(self):
        """Actualizar colores de clusters según selección"""
        for scatter, cluster_idx in self.scatter_objects:
            if cluster_idx in self.selected_clusters:
                scatter.set_color('red')
                scatter.set_alpha(1.0)
            else:
                scatter.set_color('blue')
                scatter.set_alpha(0.8)
        
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
            import pandas as pd
            df = pd.DataFrame(self.time_series_data).T
            df.columns = self.cluster_labels
            df.index.name = 'Frame'
            df.to_csv(filename)
            messagebox.showinfo("Éxito", f"Series temporales exportadas a {filename}")
    
    def export_coordinates(self):
        """Exportar coordenadas de clusters seleccionados"""
        filename = asksaveasfilename(
            initialfile='coordenadas_clusters.txt',
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write("# Coordenadas de Clusters Seleccionados\n")
                f.write(f"# Método de variabilidad: {self.method_name}\n")
                f.write(f"# Threshold usado: {self.threshold_var.get()}\n")
                f.write(f"# Total de clusters: {len(self.selected_clusters)}\n\n")
                
                for cluster_idx in self.selected_clusters:
                    cluster_points = self.final_clusters[cluster_idx]
                    f.write(f"Cluster {cluster_idx} ({len(cluster_points)} puntos):\n")
                    for point in cluster_points:
                        f.write(f"{point[0]},{point[1]}\n")
                    f.write("\n")
            
            messagebox.showinfo("Éxito", f"Coordenadas exportadas a {filename}")
    
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
            with open(filename, 'w') as f:
                f.write("REPORTE DE ANÁLISIS DE CLUSTERS\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Método de variabilidad: {self.method_name}\n")
                f.write(f"Threshold usado: {self.threshold_var.get()}\n")
                f.write(f"Rango de tamaños: {self.min_size_var.get()} - {self.max_size_var.get()}\n")
                f.write(f"Clusters seleccionados: {len(self.selected_clusters)}\n")
                f.write(f"Método de correlación: {self.current_method}\n\n")
                
                f.write("DETALLES DE CLUSTERS:\n")
                f.write("-" * 25 + "\n")
                for cluster_idx in self.selected_clusters:
                    cluster_size = len(self.final_clusters[cluster_idx])
                    f.write(f"Cluster {cluster_idx}: {cluster_size} píxeles\n")
                
                f.write(f"\nMATRIZ DE CORRELACIÓN ({self.current_method}):\n")
                f.write("-" * 35 + "\n")
                f.write(self.current_corr_matrix.to_string())
                f.write("\n\n")
                
                f.write("ESTADÍSTICAS DE CORRELACIÓN:\n")
                f.write("-" * 30 + "\n")
                correlations = self.current_corr_matrix.values[np.triu_indices_from(self.current_corr_matrix.values, k=1)]
                f.write(f"Correlación promedio: {np.mean(correlations):.3f}\n")
                f.write(f"Correlación máxima: {np.max(correlations):.3f}\n")
                f.write(f"Correlación mínima: {np.min(correlations):.3f}\n")
                f.write(f"Desviación estándar: {np.std(correlations):.3f}\n")
            
            messagebox.showinfo("Éxito", f"Reporte generado en {filename}")
    
    def show_time_series(self):
        """Mostrar serie temporal del primer cluster"""
        if self.final_clusters is None or len(self.final_clusters) == 0:
            messagebox.showwarning("Advertencia", "Primero procesa los clusters")
            return
        
        # Usar el primer cluster como ejemplo
        cluster_points = self.final_clusters[0]
        if len(cluster_points) == 0:
            messagebox.showwarning("Advertencia", "El cluster seleccionado está vacío")
            return
        
        time_series = extract_time_series(self.img_array, cluster_points)
        
        # Crear ventana para serie temporal
        ts_window = tk.Toplevel(self.window)
        ts_window.title(f"Serie Temporal - Cluster 0")
        ts_window.geometry("600x400")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.array(range(len(time_series))), time_series)
        ax.set_title(f'Serie Temporal - Cluster 0 ({len(cluster_points)} píxeles)')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Intensidad Promedio')
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=ts_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def save_image(self):
        """Guardar la imagen actual"""
        if hasattr(self, 'current_fig'):
            filename = asksaveasfilename(
                initialfile=f'{self.method_name.replace(" ", "_")}_analysis.png',
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All Files", "*.*")]
            )
            if filename:
                self.current_fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Éxito", f"Imagen guardada en {filename}")

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