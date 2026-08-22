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

def calculate_variability(img_array, method=1):
    """
    Calculates the variability of an image using different methods.

    """
    methods_info = {
        0: {"name": "Range", "default_th": 100},
        1: {"name": "Population Variance", "default_th": 120},
        2: {"name": "Sample Variance", "default_th": 200},
        3: {"name": "Population Standard Deviation", "default_th": 12},
        4: {"name": "Sample Standard Deviation", "default_th": 5},
        5: {"name": "Coefficient of Variation", "default_th": 5},
        6: {"name": "Interquartile Range (IQR)", "default_th": 20}
    }

    select = method

    if select == 0:
        # 1. Range
        var_im = np.max(img_array, axis=0) - np.min(img_array, axis=0)
        th = 100
    elif select == 1:
        # 2. Variance
        var_im = np.var(img_array, axis=0)  # Population
        th = 120
    elif select == 2:
        # 2. Variance
        var_im = np.var(img_array, axis=0, ddof=1)  # Sample
        th = 200
    elif select == 3:
        # 3. Standard deviation
        th = 12
        var_im = np.std(img_array, axis=0)  # Population
    elif select == 4:
        # 3. Standard deviation
        th = 5
        var_im = np.std(img_array, axis=0, ddof=1)  # Sample
    elif select == 5:
        # 4. Coefficient of variation
        th = 5
        var_im = np.std(img_array, axis=0, ddof=1)  # Sample
        media = np.mean(img_array, axis=0)
        var_im = (var_im / media) * 100
    elif select == 6:
        th = 20
        # 5. Interquartile range (IQR)
        q1 = np.percentile(img_array, 25, axis=0)
        q3 = np.percentile(img_array, 75, axis=0)
        var_im = q3 - q1
    else:
        raise ValueError("Method must be between 0 and 6")
    
    return var_im, th, methods_info[method]["name"]

def apply_image_processing(var_im):
    """
    Applies image processing: unsharp mask and filter.
    """
    result_1 = unsharp_mask(var_im, radius=20, amount=1)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv.filter2D(var_im, -1, kernel)
    return image

def apply_binarization(image, th):
    """
    Applies binarization
    """
    deconvolved_RL2 = np.reshape(image, (image.shape[0] * image.shape[1]))
    res_labels = [int(deconvolved_RL2[i] > th) for i in range(len(deconvolved_RL2))]
    res_labels = np.reshape(res_labels, (image.shape[0], image.shape[1]))
    return res_labels

def candidate_neighbors(node):
    """
    Original function to find neighbors.
    """
    return [(node[0] + 1, node[1]), (node[0], node[1] + 1), (node[0] + 1, node[1] + 1),
           (node[0] - 1, node[1]), (node[0], node[1] - 1), (node[0] - 1, node[1] - 1),
           (node[0] + 1, node[1] - 1), (node[0] - 1, node[1] + 1)]

def neighboring_groups(nodes):
    """
    Original function to group connected pixels.
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
    Extracts pixels from the binarized image.
    """
    pixels = []
    for i in range(res_labels.shape[0]):
        for j in range(res_labels.shape[1]):
            if res_labels[i][j] == True:
                pixels.append((i, j))
    return pixels

def find_peaks(Z, dz_dx, dz_dy, selected_points):
    """
    Detects peaks using sign changes in the derivatives.
    Original code by Jose.
    """
    peaks = []
    for i, j in selected_points:
        # Check that we are not on the edges
        if 1 <= i < Z.shape[0] - 1 and 1 <= j < Z.shape[1] - 1:
            # Check if it is a peak (local maximum)
            if dz_dx[i, j - 1] > 0 and dz_dx[i, j] <= 0 and dz_dy[i - 1, j] > 0 and dz_dy[i, j] <= 0:
                peaks.append((i, j))
    return peaks

def assign_points_to_peaks(Z, peaks, selected_points):
    """
    Assigns selected points to the nearest peaks.
    Original code by Jose.
    """
    import math

    # Initialize a list for the point sets per peak
    peak_sets = {i: [] for i in range(len(peaks))}

    # Create a label matrix to assign points to sets
    peak_map = np.zeros_like(Z, dtype=int) - 1
    gx = np.gradient(Z, axis=1)  # Partial derivative with respect to x
    gy = np.gradient(Z, axis=0)  # Partial derivative with respect to y

    # Label the peaks with unique IDs
    for label_id, (pi, pj) in enumerate(peaks):
        if 0 <= pi < Z.shape[0] and 0 <= pj < Z.shape[1]:
            peak_map[pi, pj] = label_id

    # Assign the selected points to their nearest peaks
    for i, j in selected_points:
        if not (0 <= i < Z.shape[0] and 0 <= j < Z.shape[1]):
            continue

        x, y = i, j
        seen = []

        # Gradient descent until a peak is found
        while (x, y) not in seen and (x, y) not in peaks:
            if len(seen) > 100:  # Avoid infinite loops
                break

            if not (0 <= x < Z.shape[0] and 0 <= y < Z.shape[1]):
                break

            # Get gradient direction
            if abs(gx[x, y]) > 1e-10 or abs(gy[x, y]) > 1e-10:
                dx, dy = int(np.sign(gx[x, y])), int(np.sign(gy[x, y]))
            else:
                break

            # Move in the direction of the gradient
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < Z.shape[0]) and (0 <= new_y < Z.shape[1]):
                seen.append((x, y))
                x, y = new_x, new_y
            else:
                break

        # Find the nearest peak
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

            # Assign the point to the peak's set
            if sel_peak is not None:
                peak_x, peak_y = sel_peak
                if 0 <= peak_x < peak_map.shape[0] and 0 <= peak_y < peak_map.shape[1]:
                    peak_id = peak_map[peak_x, peak_y]
                    if peak_id >= 0 and peak_id in peak_sets:
                        peak_sets[peak_id].append((i, j))
    
    return peak_sets

def process_clusters_advanced(var_im, res_clusters, min_size=20, max_size=200):
    """
    Processes clusters with advanced peak detection.
    Original code by Jose adapted with robust validations.
    """
    try:
        from sklearn import linear_model

        clusters_min_size = []
        final_cl = []

        # Calculate partial derivatives
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

        # Process clusters with peak detection
        for cl in clusters_min_size:
            try:
                if len(cl) == 0:
                    continue

                # Extract coordinates and values
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

                if len(z) < 3:  # We need at least 3 points
                    final_cl.append(cl)
                    continue

                z = np.array(z)
                y = np.array(y)
                x = np.array(x)

                # Classification with SGD One-Class SVM
                try:
                    clf = linear_model.SGDOneClassSVM(random_state=42, nu=0.131)
                    clf.fit(z.reshape(-1, 1))
                    y_pred = clf.predict(z.reshape(-1, 1))
                    y_res = [i for i, pred in enumerate(list(y_pred)) if pred == -1]
                except:
                    # If classification fails, use original cluster
                    final_cl.append(cl)
                    continue

                # Create grid for surface
                if len(x) > 0 and len(y) > 0:
                    min_x, max_x = int(min(x)), int(max(x))
                    min_y, max_y = int(min(y)), int(max(y))

                    if max_x > min_x and max_y > min_y:
                        Z = np.zeros((max_x - min_x + 1, max_y - min_y + 1))

                        # Fill grid with values
                        for i in range(len(z)):
                            grid_x = int(x[i] - min_x)
                            grid_y = int(y[i] - min_y)
                            if 0 <= grid_x < Z.shape[0] and 0 <= grid_y < Z.shape[1]:
                                Z[grid_x, grid_y] = z[i]

                        # Detect peaks
                        peaks = find_peaks(Z, dz_dx[min_x:max_x+1, min_y:max_y+1],
                                         dz_dy[min_x:max_x+1, min_y:max_y+1],
                                         [(i-min_x, j-min_y) for i, j in cl if min_x <= i <= max_x and min_y <= j <= max_y])

                        # Convert peaks back to original coordinates
                        original_peaks = [(p[0] + min_x, p[1] + min_y) for p in peaks]

                        if len(original_peaks) > 0:
                            # Assign points to peaks
                            relative_cl = [(i-min_x, j-min_y) for i, j in cl if min_x <= i <= max_x and min_y <= j <= max_y]
                            peak_sets = assign_points_to_peaks(Z, peaks, relative_cl)

                            # Convert back to original coordinates and add
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
                # If something fails in advanced processing, use original cluster
                pass  # Silent error
                final_cl.append(cl)
                continue

        return final_cl, clusters_min_size

    except Exception as e:
        pass  # Silent error
        # Fallback to basic method
        final_cl = []
        for cl in res_clusters:
            if min_size <= len(cl) <= max_size:
                final_cl.append(cl)
        return final_cl, []

def extract_time_series(img_array, cluster_points):
    """
    Extracts time series of a specific cluster.
    """
    final_array = np.array(cluster_points)
    sel_data_final = img_array[:, final_array[:,0], final_array[:,1]]
    sel_data_final_mean = np.mean(sel_data_final, axis=1)
    return sel_data_final_mean

def decompose_large_clusters(res_clusters, var_im, min_size=20, max_size=200):
    """
    Decomposes large clusters using KMeans with spatial + variability features.
    Clusters smaller than min_size are discarded, those between min_size and max_size
    are kept, and those larger than max_size are subdivided with KMeans.
    Based on the cluster-Copy4.ipynb notebook.
    """
    decomposed = []
    max_ax = np.max(var_im.shape)
    max_var = var_im.max()
    if max_var == 0:
        max_var = 1  # Avoid division by zero
    norm_var_im = (var_im / max_var) * max_ax

    for cl in res_clusters:
        cl_size = len(cl)
        if cl_size < min_size:
            continue  # Discard very small clusters
        elif cl_size <= max_size:
            decomposed.append(cl)  # Keep clusters of adequate size
        else:
            # Decompose large clusters with KMeans
            num_sub = (cl_size // max_size) + 1
            try:
                # Create 3D features: scaled coordinates + normalized variability
                new_cl = [
                    (cl[i][0] * 10, cl[i][1] * 10, norm_var_im[cl[i][0], cl[i][1]])
                    for i in range(cl_size)
                ]
                kmeans = KMeans(n_clusters=num_sub, random_state=0, n_init="auto").fit(new_cl)
                labels = kmeans.labels_
                for label_val in np.unique(labels):
                    sub_cluster = np.array(cl)[labels == label_val]
                    # Convert back to list of tuples
                    sub_cluster_list = [tuple(p) for p in sub_cluster.tolist()]
                    if len(sub_cluster_list) >= min_size:
                        decomposed.append(sub_cluster_list)
            except Exception as e:
                pass  # Silent error
                decomposed.append(cl)  # Keep original if it fails
    
    return decomposed

class VariabilityAnalysisWindow:
    """
    Window for variability analysis with cluster selection.
    """
    def __init__(self, img_array, method, main_window):
        self.img_array = img_array
        self.method = method
        self.main_window = main_window

        # Calculate initial variability
        self.var_im, self.default_th, self.method_name = calculate_variability(img_array, method)
        self.processed_image = apply_image_processing(self.var_im)

        # Variables for clustering
        self.res_labels = None
        self.res_clusters = None
        self.final_clusters = None
        self.selected_clusters = []
        self.scatter_objects = []
        self.rect_selector = None
        self.selection_mode = 'add'  # 'add' or 'remove'
        self.cluster_colors = []  # Random colors per cluster

        # Variables for analysis
        self.time_series_data = []
        self.cluster_labels = []

        self.create_window()
        self.update_display()

    def _generate_cluster_colors(self, n_clusters):
        """Generates random colors for each cluster, excluding red (reserved for selection)."""
        import colorsys
        colors = []
        rng = np.random.RandomState(42)
        for _ in range(n_clusters):
            # Generate colors with hue far from red (0.0)
            # Red is at hue ~0.0 and ~1.0, we avoid the range [0.0, 0.05] and [0.92, 1.0]
            hue = rng.uniform(0.08, 0.88)
            sat = rng.uniform(0.5, 1.0)
            val = rng.uniform(0.6, 1.0)
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append(rgb)
        return colors
    
    def create_window(self):
        """Create the main window"""
        self.window = tk.Toplevel(self.main_window)
        self.window.title(f"Full Analysis - {self.method_name}")
        self.window.geometry("1400x800")

        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Menu bar
        self.window.config(menu=self._create_menu_bar())

        # Main frame
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Parameter bar (threshold, sizes) and info
        self.create_controls(main_frame)

        # Horizontal frame for visualization and selection
        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Frame for visualization (left side)
        self.viz_frame = tk.Frame(content_frame)
        self.viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for selection (right side)
        self.selection_frame = tk.Frame(content_frame, width=200, relief=tk.RAISED, borderwidth=1)
        self.selection_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.selection_frame.pack_propagate(False)

        self.create_selection_panel()

    def _create_menu_bar(self):
        """Create the menu bar for the analysis window."""
        menu_bar = tk.Menu(self.window, tearoff=False)

        # Clustering menu
        menu_clustering = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Clustering", menu=menu_clustering)
        menu_clustering.add_command(label="Process Cluster (Basic)", command=self.process_clusters_basic)
        menu_clustering.add_command(label="Process Cluster (Advanced)", command=self.process_clusters_advanced)
        menu_clustering.add_separator()
        menu_clustering.add_command(label="Decompose Large Clusters", command=self.decompose_clusters)

        # Selection menu
        menu_seleccion = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Selection", menu=menu_seleccion)
        menu_seleccion.add_command(label="Select All", command=self.select_all_clusters)
        menu_seleccion.add_command(label="Clear Selection", command=self.clear_selection)

        # Visualization menu
        menu_visual = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Visualization", menu=menu_visual)
        menu_visual.add_command(label="3D View", command=self.show_3d_surface)

        # Export menu
        menu_exportar = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Export", menu=menu_exportar)
        menu_exportar.add_command(label="Save Image", command=self.save_image)
        menu_exportar.add_command(label="Save .npy", command=self.save_selected_npy)
        menu_exportar.add_separator()
        menu_exportar.add_command(label="Use Selected (Correlations)", command=self.use_selected_clusters)

        return menu_bar

    def create_controls(self, parent):
        """Create parameter bar (spinboxes) and info."""
        control_frame = tk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Threshold
        tk.Label(control_frame, text="Threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.threshold_var = tk.IntVar(value=self.default_th)
        ttk.Spinbox(control_frame, from_=1, to=1000, textvariable=self.threshold_var, width=10).pack(side=tk.LEFT, padx=(0, 15))

        # Minimum size
        tk.Label(control_frame, text="Min Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.min_size_var = tk.IntVar(value=20)
        ttk.Spinbox(control_frame, from_=1, to=500, textvariable=self.min_size_var, width=8).pack(side=tk.LEFT, padx=(0, 15))

        # Maximum size
        tk.Label(control_frame, text="Max Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.max_size_var = tk.IntVar(value=200)
        ttk.Spinbox(control_frame, from_=1, to=1000, textvariable=self.max_size_var, width=8).pack(side=tk.LEFT, padx=(0, 15))

        self.info_label = tk.Label(control_frame, text="Use the menu to process clusters")
        self.info_label.pack(side=tk.LEFT, padx=10)

    def create_selection_panel(self):
        """Create side selection panel"""
        # Title
        title_label = tk.Label(self.selection_frame, text="Selected Clusters",
                              font=("Arial", 12, "bold"))
        title_label.pack(pady=10)

        # List of selected items
        self.selected_listbox = tk.Listbox(self.selection_frame, height=15)
        self.selected_listbox.pack(fill=tk.BOTH, expand=True, padx=10)

        # Frame for panel buttons
        button_frame = tk.Frame(self.selection_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        # Button to remove selected item
        remove_btn = tk.Button(button_frame, text="Remove",
                              command=self.remove_selected_from_list)
        remove_btn.pack(fill=tk.X, pady=2)

        # Separator
        ttk.Separator(self.selection_frame, orient='horizontal').pack(fill='x', padx=10, pady=5)

        # Region selection controls
        region_label = tk.Label(self.selection_frame, text="Region Selection",
                               font=("Arial", 10, "bold"))
        region_label.pack(pady=(5, 2))

        region_info = tk.Label(self.selection_frame,
                              text="Right-click + drag\non the cluster plot",
                              font=("Arial", 8), fg='gray',
                              justify=tk.CENTER)
        region_info.pack(pady=(0, 5))

        mode_frame = tk.Frame(self.selection_frame)
        mode_frame.pack(fill=tk.X, padx=10)

        self.selection_mode_var = tk.StringVar(value='add')
        tk.Radiobutton(mode_frame, text="Add", variable=self.selection_mode_var,
                       value='add', command=self._update_selection_mode).pack(side=tk.LEFT, expand=True)
        tk.Radiobutton(mode_frame, text="Remove", variable=self.selection_mode_var,
                       value='remove', command=self._update_selection_mode).pack(side=tk.LEFT, expand=True)

        # Information
        self.selection_info = tk.Label(self.selection_frame,
                                     text="Click on clusters\nto select",
                                     justify=tk.CENTER)
        self.selection_info.pack(pady=10)
    
    def update_display(self):
        """Update the visualization"""
        # Clear previous frame
        for widget in self.viz_frame.winfo_children():
            widget.destroy()

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # 1. Original variability image
        im1 = ax1.imshow(self.var_im, cmap='viridis')
        ax1.set_title(f'{self.method_name}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)

        # 2. Processed image
        im2 = ax2.imshow(self.processed_image, cmap='viridis')
        ax2.set_title('Processed Image (Unsharp + Filter)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)

        # 3. Binarization (if it exists)
        if self.res_labels is not None:
            ax3.imshow(self.res_labels, cmap='gray')
            ax3.set_title(f'Binarization (th={self.threshold_var.get()})')
        else:
            ax3.text(0.5, 0.5, 'Apply binarization', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Binarization')
        ax3.axis('off')

        # 4. Clusters (if they exist) - WITH FIXED INTERACTIVITY
        if self.final_clusters is not None:
            # Clear previous scatter objects
            self.scatter_objects = []

            # Generate random colors if we don't have them or the count changed
            if len(self.cluster_colors) != len(self.final_clusters):
                self.cluster_colors = self._generate_cluster_colors(len(self.final_clusters))

            for i, cl in enumerate(self.final_clusters):
                if len(cl) > 0:
                    cl_array = np.array(cl)

                    # Red if selected, random color otherwise
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

                    # Store reference for clicks
                    self.scatter_objects.append((scatter, i))

            ax4.set_title(f'Final Clusters ({len(self.final_clusters)} found) - Click to select')
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')

        else:
            ax4.text(0.5, 0.5, 'Process clusters', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Clusters - Click to select')

        plt.tight_layout()

        # Add to the window
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # CONNECT CLICK EVENT - VERY IMPORTANT
        if self.final_clusters is not None:
            canvas.mpl_connect('pick_event', self.on_cluster_click)

            # Connect rectangle selector for region selection (right button)
            self.rect_selector = RectangleSelector(
                ax4, self._on_region_selected,
                useblit=True,
                button=[3],  # Right mouse button only
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=False,
                props=dict(facecolor='yellow', edgecolor='black', alpha=0.3, linewidth=1.5)
            )

        # Store reference for saving
        self.current_fig = fig
        self.current_canvas = canvas
    
    def _apply_binarization(self):
        """Binarize the processed image with the current threshold."""
        th = self.threshold_var.get()
        self.res_labels = apply_binarization(self.processed_image, th)

    def _find_clusters(self):
        """Find connected components in the binarized image."""
        pixels = extract_pixels_from_binary(self.res_labels)
        self.res_clusters = list(neighboring_groups(pixels))

    def process_clusters_basic(self):
        """Binarize, find connected components, and filter by size."""
        min_size = self.min_size_var.get()
        max_size = self.max_size_var.get()
        th = self.threshold_var.get()

        self._apply_binarization()
        self._find_clusters()

        self.selected_clusters = []
        self.cluster_colors = []

        self.final_clusters = [cl for cl in self.res_clusters if min_size <= len(cl) <= max_size]

        self.info_label.config(
            text=f"Basic: {len(self.final_clusters)} clusters (th={th}, size {min_size}–{max_size})"
        )
        self.update_display()
        self.update_selection_list()

    def process_clusters_advanced(self):
        """Binarize, find connected components, and process with 3D peak detection."""
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
                text=f"Advanced: {len(self.final_clusters)} clusters (th={th}, 3D peaks)"
            )
            self.update_display()
        except Exception as e:
            messagebox.showwarning("Warning", f"Error in advanced processing: {e}\nUsing basic method as fallback")

    def decompose_clusters(self):
        """Decompose large clusters using KMeans"""
        if self.res_clusters is None:
            messagebox.showwarning("Warning", "First process the clusters with 'Process Cluster (Basic)' or '(Advanced)'")
            return

        min_size = self.min_size_var.get()
        max_size = self.max_size_var.get()

        # Count large clusters before decomposing
        large_count = sum(1 for cl in self.res_clusters if len(cl) > max_size)

        # Decompose
        self.res_clusters = decompose_large_clusters(
            self.res_clusters, self.var_im, min_size, max_size
        )

        # Clear selections and final clusters
        self.selected_clusters = []
        self.final_clusters = None
        self.cluster_colors = []  # Regenerate colors

        self.info_label.config(
            text=f"Decomposed {large_count} large clusters → {len(self.res_clusters)} total clusters"
        )
        self.update_display()
        self.update_selection_list()

    def select_all_clusters(self):
        """Select all clusters"""
        if self.final_clusters is None or len(self.final_clusters) == 0:
            messagebox.showwarning("Warning", "First process the clusters")
            return

        self.selected_clusters = list(range(len(self.final_clusters)))
        self.update_cluster_colors()
        self.update_selection_list()

    def _update_selection_mode(self):
        """Update region selection mode"""
        self.selection_mode = self.selection_mode_var.get()

    def _on_region_selected(self, eclick, erelease):
        """Callback for rectangular region selection"""
        if self.final_clusters is None:
            return

        # Get rectangle coordinates in plot space
        x_min = min(eclick.xdata, erelease.xdata)
        x_max = max(eclick.xdata, erelease.xdata)
        y_min = min(eclick.ydata, erelease.ydata)
        y_max = max(eclick.ydata, erelease.ydata)

        img_height = np.array(self.var_im).shape[0]

        # Find clusters whose centroid falls within the rectangle
        changed = False
        for i, cl in enumerate(self.final_clusters):
            if len(cl) == 0:
                continue
            cl_array = np.array(cl)
            # Calculate centroid in plot coordinates (same as in scatter)
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
        """KEY FUNCTION - Handle click on cluster (WITH ANTI-AUTO-CLICK PROTECTION)"""

        # PROTECTION: Only process real user clicks
        # If no button is pressed, it's an automatic event - ignore
        if not hasattr(event, 'mouseevent') or event.mouseevent is None:
            return

        # PROTECTION: Only left mouse button clicks
        if hasattr(event.mouseevent, 'button') and event.mouseevent.button != 1:
            return


        # Find which cluster was clicked
        clicked_cluster = None
        for scatter, cluster_idx in self.scatter_objects:
            if event.artist == scatter:
                clicked_cluster = cluster_idx
                break

        if clicked_cluster is not None:
            if clicked_cluster in self.selected_clusters:
                # Deselect
                self.selected_clusters.remove(clicked_cluster)
            else:
                # Select
                self.selected_clusters.append(clicked_cluster)

            # Update visualization and list
            self.update_cluster_colors()
            self.update_selection_list()

    def update_cluster_colors(self):
        """Update cluster colors based on selection: red=selected, random color=not selected.
        Also shows the cluster ID above selected clusters."""
        # Clear previous annotations
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

                # Show cluster ID at the centroid
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

        # Refresh canvas
        if hasattr(self, 'current_canvas'):
            self.current_canvas.draw()

    def update_selection_list(self):
        """Update list of selected clusters"""
        self.selected_listbox.delete(0, tk.END)

        for cluster_idx in sorted(self.selected_clusters):
            cluster_size = len(self.final_clusters[cluster_idx]) if self.final_clusters else 0
            self.selected_listbox.insert(tk.END, f"Cluster {cluster_idx} ({cluster_size} pts)")

        # Update information
        count = len(self.selected_clusters)
        self.selection_info.config(text=f"{count} clusters\nselected")

    def remove_selected_from_list(self):
        """Remove selected cluster from the list"""
        selection = self.selected_listbox.curselection()
        if selection:
            # Get cluster index from the text
            item_text = self.selected_listbox.get(selection[0])
            cluster_idx = int(item_text.split()[1])  # "Cluster X (...)" -> X

            if cluster_idx in self.selected_clusters:
                self.selected_clusters.remove(cluster_idx)
                self.update_cluster_colors()
                self.update_selection_list()

    def clear_selection(self):
        """Clear the entire selection"""
        self.selected_clusters = []
        self.update_cluster_colors()
        self.update_selection_list()

    def save_selected_npy(self):
        """Save selected clusters as an .npy file"""
        if not self.selected_clusters:
            messagebox.showwarning("Warning", "No clusters selected")
            return

        if self.final_clusters is None:
            messagebox.showwarning("Warning", "First process the clusters")
            return

        filename = asksaveasfilename(
            initialfile='selected_clusters.npy',
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All Files", "*.*")]
        )
        
        if filename:
            # Build 2D array (n_frames, 1 + n_clusters):
            #   column 0 = frame indices (skipped by _load_data with [:,1:])
            #   columns 1+ = mean time series for each selected cluster
            time_series_list = []
            for cluster_idx in sorted(self.selected_clusters):
                ts = extract_time_series(self.img_array, self.final_clusters[cluster_idx])
                time_series_list.append(ts)

            n_frames = len(time_series_list[0])
            frame_col = np.arange(n_frames, dtype=float).reshape(-1, 1)
            ts_matrix = np.column_stack(time_series_list).astype(float)
            data = np.hstack([frame_col, ts_matrix])

            np.save(filename, data)
            messagebox.showinfo("Success", f"Clusters saved to {filename}")

    def _on_window_close(self):
        """Close the analysis window and free resources"""
        plt.close('all')
        self.window.destroy()

    def use_selected_clusters(self):
        """Use selected clusters for analysis"""
        if not self.selected_clusters:
            messagebox.showwarning("Warning", "No clusters selected")
            return

        # Create advanced analysis window
        self.show_correlation_analysis()

    def show_correlation_analysis(self):
        """Show correlation analysis of selected clusters"""
        # Create analysis window
        corr_window = tk.Toplevel(self.window)
        corr_window.title("Correlation Analysis - Selected Clusters")
        corr_window.geometry("1200x800")

        # Main frame
        main_frame = tk.Frame(corr_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top controls frame
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Information
        info_text = f"Analyzing correlations between {len(self.selected_clusters)} selected clusters"
        info_label = tk.Label(control_frame, text=info_text, font=("Arial", 12, "bold"))
        info_label.pack(pady=5)

        # Correlation buttons
        corr_button_frame = tk.Frame(control_frame)
        corr_button_frame.pack(fill=tk.X, pady=5)

        tk.Label(corr_button_frame, text="Correlation Type:").pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(corr_button_frame, text="Pearson",
                 command=lambda: self.calculate_correlation('pearson')).pack(side=tk.LEFT, padx=5)
        tk.Button(corr_button_frame, text="Kendall",
                 command=lambda: self.calculate_correlation('kendall')).pack(side=tk.LEFT, padx=5)
        tk.Button(corr_button_frame, text="Spearman",
                 command=lambda: self.calculate_correlation('spearman')).pack(side=tk.LEFT, padx=5)

        # Separator
        tk.Frame(corr_button_frame, height=2, bg="gray").pack(side=tk.LEFT, fill=tk.X, padx=20)

        # Export buttons
        tk.Button(corr_button_frame, text="Export Time Series",
                 command=self.export_time_series).pack(side=tk.LEFT, padx=5)
        tk.Button(corr_button_frame, text="Export Coordinates",
                 command=self.export_coordinates).pack(side=tk.LEFT, padx=5)
        tk.Button(corr_button_frame, text="Generate Report",
                 command=self.generate_report).pack(side=tk.LEFT, padx=5)

        # Frame for visualization
        self.corr_viz_frame = tk.Frame(main_frame)
        self.corr_viz_frame.pack(fill=tk.BOTH, expand=True)

        # Show initial time series
        self.show_initial_time_series()
    
    def show_initial_time_series(self):
        """Show time series of selected clusters"""
        # Clear frame
        for widget in self.corr_viz_frame.winfo_children():
            widget.destroy()

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Extract time series of selected clusters
        self.time_series_data = []
        self.cluster_labels = []

        for cluster_idx in self.selected_clusters:
            if len(self.final_clusters[cluster_idx]) > 0:
                time_series = extract_time_series(self.img_array, self.final_clusters[cluster_idx])
                self.time_series_data.append(time_series)
                self.cluster_labels.append(f'Cluster {cluster_idx}')

        # Plot individual time series
        for i, (ts, label) in enumerate(zip(self.time_series_data, self.cluster_labels)):
            ax1.plot(ts, label=label, alpha=0.8)

        ax1.set_title('Time Series of Selected Clusters')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Average Intensity')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Preliminary correlation heatmap
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
            ax2.set_title('Preliminary Correlation Matrix (Pearson)')

            # Add values on the map
            for i in range(len(self.cluster_labels)):
                for j in range(len(self.cluster_labels)):
                    text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontweight="bold")

            plt.colorbar(im, ax=ax2)
        else:
            ax2.text(0.5, 0.5, 'Select at least 2 clusters\nfor correlation analysis',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Correlation Matrix')

        plt.tight_layout()

        # Add to the window
        canvas = FigureCanvasTkAgg(fig, master=self.corr_viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Store reference
        self.current_corr_fig = fig

    def calculate_correlation(self, method):
        """Calculate correlation using the specified method"""
        if len(self.time_series_data) < 2:
            messagebox.showwarning("Warning", "You need at least 2 selected clusters")
            return

        import pandas as pd

        # Create DataFrame with the time series
        df = pd.DataFrame(self.time_series_data).T
        df.columns = self.cluster_labels

        # Calculate correlation according to the method
        if method == 'pearson':
            corr_matrix = df.corr(method='pearson')
        elif method == 'kendall':
            corr_matrix = df.corr(method='kendall')
        elif method == 'spearman':
            corr_matrix = df.corr(method='spearman')

        # Update visualization
        self.update_correlation_display(corr_matrix, method.capitalize())
    
    def update_correlation_display(self, corr_matrix, method_name):
        """Update the correlation visualization"""
        # Clear frame
        for widget in self.corr_viz_frame.winfo_children():
            widget.destroy()

        # Create new figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Time series
        for i, (ts, label) in enumerate(zip(self.time_series_data, self.cluster_labels)):
            ax1.plot(ts, label=label, alpha=0.8)
        ax1.set_title('Time Series of Selected Clusters')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Intensity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Correlation matrix
        im2 = ax2.imshow(corr_matrix.values, cmap='RdBu', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(self.cluster_labels)))
        ax2.set_yticks(range(len(self.cluster_labels)))
        ax2.set_xticklabels(self.cluster_labels, rotation=45)
        ax2.set_yticklabels(self.cluster_labels)
        ax2.set_title(f'Correlation Matrix ({method_name})')

        # Add values in the matrix
        for i in range(len(self.cluster_labels)):
            for j in range(len(self.cluster_labels)):
                text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight="bold")

        plt.colorbar(im2, ax=ax2)

        # 3. Histogram of correlations
        correlations = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        ax3.hist(correlations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Distribution of Correlations')
        ax3.set_xlabel('Correlation Value')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        # 4. Reordered correlation map
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform

            # Create distance matrix
            distance_matrix = 1 - np.abs(corr_matrix.values)
            condensed_distances = squareform(distance_matrix)

            # Hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method='ward')
            dendro = dendrogram(linkage_matrix, labels=self.cluster_labels, ax=ax4)
            ax4.set_title('Cluster Dendrogram')
            ax4.set_xlabel('Clusters')
            ax4.set_ylabel('Distance')
        except Exception as e:
            ax4.text(0.5, 0.5, 'Error generating dendrogram', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Dendrogram')

        plt.tight_layout()

        # Add to the window
        canvas = FigureCanvasTkAgg(fig, master=self.corr_viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Store for export
        self.current_corr_fig = fig
        self.current_corr_matrix = corr_matrix
        self.current_method = method_name
    
    def export_time_series(self):
        """Export time series to a CSV or Excel file"""
        if not hasattr(self, 'time_series_data') or not self.time_series_data:
            messagebox.showwarning("Warning", "No data to export")
            return

        filename = asksaveasfilename(
            initialfile='cluster_time_series.csv',
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All Files", "*.*")]
        )

        if filename:
            import pandas as pd
            df = pd.DataFrame(self.time_series_data).T
            df.columns = self.cluster_labels
            df.index.name = 'Frame'
            if filename.lower().endswith(('.xlsx', '.xls')):
                df.to_excel(filename, engine='xlsxwriter')
            else:
                df.to_csv(filename)
            messagebox.showinfo("Success", f"Time series exported to {filename}")

    def export_coordinates(self):
        """Export coordinates of selected clusters"""
        filename = asksaveasfilename(
            initialfile='cluster_coordinates.txt',
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All Files", "*.*")]
        )

        if filename:
            with open(filename, 'w') as f:
                f.write("# Coordinates of Selected Clusters\n")
                f.write(f"# Variability method: {self.method_name}\n")
                f.write(f"# Threshold used: {self.threshold_var.get()}\n")
                f.write(f"# Total clusters: {len(self.selected_clusters)}\n\n")

                for cluster_idx in self.selected_clusters:
                    cluster_points = self.final_clusters[cluster_idx]
                    f.write(f"Cluster {cluster_idx} ({len(cluster_points)} points):\n")
                    for point in cluster_points:
                        f.write(f"{point[0]},{point[1]}\n")
                    f.write("\n")

            messagebox.showinfo("Success", f"Coordinates exported to {filename}")

    def generate_report(self):
        """Generate complete analysis report"""
        if not hasattr(self, 'current_corr_matrix'):
            messagebox.showwarning("Warning", "First calculate a correlation")
            return

        filename = asksaveasfilename(
            initialfile='cluster_analysis_report.txt',
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All Files", "*.*")]
        )

        if filename:
            with open(filename, 'w') as f:
                f.write("CLUSTER ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Variability method: {self.method_name}\n")
                f.write(f"Threshold used: {self.threshold_var.get()}\n")
                f.write(f"Size range: {self.min_size_var.get()} - {self.max_size_var.get()}\n")
                f.write(f"Selected clusters: {len(self.selected_clusters)}\n")
                f.write(f"Correlation method: {self.current_method}\n\n")

                f.write("CLUSTER DETAILS:\n")
                f.write("-" * 25 + "\n")
                for cluster_idx in self.selected_clusters:
                    cluster_size = len(self.final_clusters[cluster_idx])
                    f.write(f"Cluster {cluster_idx}: {cluster_size} pixels\n")

                f.write(f"\nCORRELATION MATRIX ({self.current_method}):\n")
                f.write("-" * 35 + "\n")
                f.write(self.current_corr_matrix.to_string())
                f.write("\n\n")

                f.write("CORRELATION STATISTICS:\n")
                f.write("-" * 30 + "\n")
                correlations = self.current_corr_matrix.values[np.triu_indices_from(self.current_corr_matrix.values, k=1)]
                f.write(f"Average correlation: {np.mean(correlations):.3f}\n")
                f.write(f"Maximum correlation: {np.max(correlations):.3f}\n")
                f.write(f"Minimum correlation: {np.min(correlations):.3f}\n")
                f.write(f"Standard deviation: {np.std(correlations):.3f}\n")

            messagebox.showinfo("Success", f"Report generated at {filename}")
    
    def show_3d_surface(self):
        """Show 3D visualization of the variability surface"""
        from mpl_toolkits.mplot3d import Axes3D

        # Create window for 3D visualization
        window_3d = tk.Toplevel(self.window)
        window_3d.title(f"3D Surface - {self.method_name}")
        window_3d.geometry("800x600")

        # Subsample the image so the visualization is faster
        step = max(1, min(self.var_im.shape) // 100)  # Maximum 100 points per dimension
        var_im_sub = self.var_im[::step, ::step]

        # Create meshgrid
        x = np.arange(0, var_im_sub.shape[1])
        y = np.arange(0, var_im_sub.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = var_im_sub

        # Create 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

        # Configure axes
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel(f'{self.method_name}')
        ax.set_title(f'Variability Surface - {self.method_name}')

        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=self.method_name)

        # Add to the window
        canvas = FigureCanvasTkAgg(fig, master=window_3d)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Frame for controls
        control_frame = tk.Frame(window_3d)
        control_frame.pack(fill=tk.X, pady=5)

        # Button to save
        def save_3d():
            filename = asksaveasfilename(
                initialfile=f'{self.method_name.replace(" ", "_")}_3D.png',
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                           ("SVG files", "*.svg"), ("EPS files", "*.eps"),
                           ("All Files", "*.*")]
            )
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"3D image saved to {filename}")

        tk.Button(control_frame, text="Save 3D Image", command=save_3d).pack(side=tk.LEFT, padx=10)
        tk.Label(control_frame, text="Use the mouse to rotate the view").pack(side=tk.LEFT, padx=10)

    def save_image(self):
        """Save the current image"""
        if hasattr(self, 'current_fig'):
            filename = asksaveasfilename(
                initialfile=f'{self.method_name.replace(" ", "_")}_analysis.png',
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                           ("SVG files", "*.svg"), ("EPS files", "*.eps"),
                           ("All Files", "*.*")]
            )
            if filename:
                self.current_fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Image saved to {filename}")

def show_variability_analysis(img_array, method, main_window):
    """
    Main function to display the complete variability analysis.
    """
    if img_array is None or len(img_array) == 0:
        messagebox.showwarning("Warning", "No image loaded")
        return

    # Create the analysis window
    analysis_window = VariabilityAnalysisWindow(img_array, method, main_window)

def get_variability_methods():
    """
    Returns list of available methods for the menu.
    """
    return [
        "Range",
        "Population Variance",
        "Sample Variance",
        "Population Standard Deviation",
        "Sample Standard Deviation",
        "Coefficient of Variation",
        "Interquartile Range (IQR)"
    ]
