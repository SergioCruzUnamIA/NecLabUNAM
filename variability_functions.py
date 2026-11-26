"""
Cell detection window ("Full Analysis - <method>").

This module is user interface only: every computation is delegated to
neclab_core (variability maps, detection, signal extraction, correlation,
file formats). Cells are kept as a label image whose values are the cells'
permanent IDs, so the IDs shown here are the same ones written to every
exported file.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import colorsys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import asksaveasfilename, askopenfilename

from neclab_core import correlation, formats, segmentation, signals, variability

# Menu label -> split method passed to segmentation.detect_cells.
_DETECTION_MODES = {
    "basic": ("Basic", "none"),
    "advanced": ("Advanced (3D peaks)", "peaks"),
    "kmeans": ("KMeans split", "kmeans"),
}


class VariabilityAnalysisWindow:
    """
    Window for variability analysis, cell detection and cell selection.
    """
    def __init__(self, img_array, method, main_window):
        self.img_array = img_array
        self.method = method
        self.main_window = main_window

        # Variability map and its sharpened version (thresholded for detection)
        self.var_im, self.default_th, self.method_name = variability.calculate_variability(img_array, method)
        self.processed_image = variability.sharpen(self.var_im)

        # Detection results
        self.mask = None            # boolean candidate-pixel mask (last detection run)
        self.labels = None          # label image: 0 = background, n = cell with ID n
        self.cell_pix = {}          # {cell_id: (N, 2) array of (row, col)}
        self.detection_info = ""    # how the current cells were obtained
        self.selected_clusters = []  # selected cell IDs, in selection order
        self.scatter_objects = []
        self.rect_selector = None
        self.selection_mode = 'add'  # 'add' or 'remove'
        self.cluster_colors = {}    # {cell_id: rgb}

        # Correlation analysis of the selected cells
        self.signals_df = None      # frames x selected cells ("cell_<id>" columns)

        self.create_window()
        self.update_display()

    # ------------------------------------------------------------ helpers

    @property
    def cell_id_list(self):
        return sorted(self.cell_pix)

    def _set_cells(self, labels, info, mask=None):
        """Replace the current cells (clears the selection)."""
        self.labels = labels
        self.mask = mask
        self.cell_pix = segmentation.cell_pixels(labels)
        self.detection_info = info
        self.selected_clusters = []
        self.cluster_colors = self._generate_cluster_colors(self.cell_id_list)
        self.update_display()
        self.update_selection_list()

    def _generate_cluster_colors(self, ids):
        """Random color per cell ID, excluding red (reserved for selection)."""
        rng = np.random.RandomState(42)
        colors = {}
        for cid in ids:
            # Hue far from red: avoid [0.0, 0.05] and [0.92, 1.0]
            hue = rng.uniform(0.08, 0.88)
            sat = rng.uniform(0.5, 1.0)
            val = rng.uniform(0.6, 1.0)
            colors[cid] = colorsys.hsv_to_rgb(hue, sat, val)
        return colors

    def _centroid_plot_xy(self, cid):
        """Centroid of a cell in plot coordinates (x = column, y = flipped row)."""
        pts = self.cell_pix[cid]
        return np.mean(pts[:, 1]), self.var_im.shape[0] - np.mean(pts[:, 0])

    def _require_cells(self):
        if not self.cell_pix:
            messagebox.showwarning("Warning", "First detect cells (Clustering menu) or load them")
            return False
        return True

    def _selected_sorted(self):
        return sorted(self.selected_clusters)

    # ------------------------------------------------------------- layout

    def create_window(self):
        """Create the main window"""
        self.window = tk.Toplevel(self.main_window)
        self.window.title(f"Full Analysis - {self.method_name}")
        self.window.geometry("1400x800")
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
        self.window.config(menu=self._create_menu_bar())

        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_controls(main_frame)

        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        self.viz_frame = tk.Frame(content_frame)
        self.viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.selection_frame = tk.Frame(content_frame, width=200, relief=tk.RAISED, borderwidth=1)
        self.selection_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.selection_frame.pack_propagate(False)

        self.create_selection_panel()

    def _create_menu_bar(self):
        """Create the menu bar for the analysis window."""
        menu_bar = tk.Menu(self.window, tearoff=False)

        menu_clustering = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Clustering", menu=menu_clustering)
        menu_clustering.add_command(label="Process Cluster (Basic)", command=self.process_clusters_basic)
        menu_clustering.add_command(label="Process Cluster (Advanced)", command=self.process_clusters_advanced)
        menu_clustering.add_separator()
        menu_clustering.add_command(label="Decompose Large Clusters (KMeans)", command=self.decompose_clusters)
        menu_clustering.add_separator()
        menu_clustering.add_command(label="Load Cells...", command=self.load_cells)

        menu_seleccion = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Selection", menu=menu_seleccion)
        menu_seleccion.add_command(label="Select All", command=self.select_all_clusters)
        menu_seleccion.add_command(label="Clear Selection", command=self.clear_selection)

        menu_visual = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Visualization", menu=menu_visual)
        menu_visual.add_command(label="3D View", command=self.show_3d_surface)

        menu_exportar = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Export", menu=menu_exportar)
        menu_exportar.add_command(label="Save Image", command=self.save_image)
        menu_exportar.add_command(label="Save Signals (CSV / XLSX / NPY)...", command=self.save_selected_signals)
        menu_exportar.add_command(label="Save Cells (Label Image)...", command=self.save_cells)
        menu_exportar.add_separator()
        menu_exportar.add_command(label="Use Selected (Correlations)", command=self.use_selected_clusters)

        return menu_bar

    def create_controls(self, parent):
        """Create parameter bar (spinboxes) and info."""
        control_frame = tk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(control_frame, text="Threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.threshold_var = tk.IntVar(value=self.default_th)
        ttk.Spinbox(control_frame, from_=1, to=1000, textvariable=self.threshold_var, width=10).pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(control_frame, text="Min Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.min_size_var = tk.IntVar(value=20)
        ttk.Spinbox(control_frame, from_=1, to=500, textvariable=self.min_size_var, width=8).pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(control_frame, text="Max Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.max_size_var = tk.IntVar(value=200)
        ttk.Spinbox(control_frame, from_=1, to=1000, textvariable=self.max_size_var, width=8).pack(side=tk.LEFT, padx=(0, 15))

        self.info_label = tk.Label(control_frame, text="Use the menu to process clusters")
        self.info_label.pack(side=tk.LEFT, padx=10)

    def create_selection_panel(self):
        """Create side selection panel"""
        tk.Label(self.selection_frame, text="Selected Cells",
                 font=("Arial", 12, "bold")).pack(pady=10)

        self.selected_listbox = tk.Listbox(self.selection_frame, height=15)
        self.selected_listbox.pack(fill=tk.BOTH, expand=True, padx=10)

        button_frame = tk.Frame(self.selection_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(button_frame, text="Remove",
                  command=self.remove_selected_from_list).pack(fill=tk.X, pady=2)

        ttk.Separator(self.selection_frame, orient='horizontal').pack(fill='x', padx=10, pady=5)

        tk.Label(self.selection_frame, text="Region Selection",
                 font=("Arial", 10, "bold")).pack(pady=(5, 2))
        tk.Label(self.selection_frame, text="Right-click + drag\non the cluster plot",
                 font=("Arial", 8), fg='gray', justify=tk.CENTER).pack(pady=(0, 5))

        mode_frame = tk.Frame(self.selection_frame)
        mode_frame.pack(fill=tk.X, padx=10)
        self.selection_mode_var = tk.StringVar(value='add')
        tk.Radiobutton(mode_frame, text="Add", variable=self.selection_mode_var,
                       value='add', command=self._update_selection_mode).pack(side=tk.LEFT, expand=True)
        tk.Radiobutton(mode_frame, text="Remove", variable=self.selection_mode_var,
                       value='remove', command=self._update_selection_mode).pack(side=tk.LEFT, expand=True)

        self.selection_info = tk.Label(self.selection_frame,
                                       text="Click on cells\nto select",
                                       justify=tk.CENTER)
        self.selection_info.pack(pady=10)

    def update_display(self):
        """Redraw the four panels."""
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        if getattr(self, 'current_fig', None) is not None:
            plt.close(self.current_fig)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # 1. Variability map
        im1 = ax1.imshow(self.var_im, cmap='viridis')
        ax1.set_title(f'{self.method_name}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)

        # 2. Sharpened map (the one thresholded)
        im2 = ax2.imshow(self.processed_image, cmap='viridis')
        ax2.set_title('Processed Image (Sharpened)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)

        # 3. Candidate-pixel mask
        if self.mask is not None:
            ax3.imshow(self.mask, cmap='gray')
            ax3.set_title(f'Binarization (th={self.threshold_var.get()})')
        elif self.labels is not None:
            ax3.imshow(self.labels > 0, cmap='gray')
            ax3.set_title('Loaded cells (mask)')
        else:
            ax3.text(0.5, 0.5, 'Apply binarization', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Binarization')
        ax3.axis('off')

        # 4. Cells (clickable)
        self.scatter_objects = []
        self._cluster_annotations = []
        height = self.var_im.shape[0]
        if self.labels is not None:
            for cid in self.cell_id_list:
                pts = self.cell_pix[cid]
                selected = cid in self.selected_clusters
                scatter = ax4.scatter(pts[:, 1], height - pts[:, 0],
                                      marker='o', s=2,
                                      alpha=1.0 if selected else 0.7,
                                      color='red' if selected else self.cluster_colors.get(cid, 'blue'),
                                      picker=True, pickradius=5,
                                      zorder=3 if selected else 2)
                self.scatter_objects.append((scatter, cid))
            ax4.set_title(f'Cells ({len(self.cell_pix)} found) - Click to select')
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.set_xlim(0, self.var_im.shape[1])
            ax4.set_ylim(0, height)
        else:
            ax4.text(0.5, 0.5, 'Process clusters', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cells - Click to select')

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if self.labels is not None:
            canvas.mpl_connect('pick_event', self.on_cluster_click)
            # Right-button drag selects every cell whose centroid is inside the rectangle
            self.rect_selector = RectangleSelector(
                ax4, self._on_region_selected,
                useblit=True,
                button=[3],
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=False,
                props=dict(facecolor='yellow', edgecolor='black', alpha=0.3, linewidth=1.5)
            )

        self.current_fig = fig
        self.current_canvas = canvas
        if self.selected_clusters:
            self.update_cluster_colors()

    # ---------------------------------------------------------- detection

    def _run_detection(self, mode):
        label, split = _DETECTION_MODES[mode]
        try:
            th = self.threshold_var.get()
            min_size = self.min_size_var.get()
            max_size = self.max_size_var.get()
        except tk.TclError:
            messagebox.showerror("Error", "Threshold, Min Size and Max Size must be numbers")
            return
        try:
            labels, mask = segmentation.detect_cells(
                self.var_im, self.processed_image, th, min_size, max_size, split=split)
        except Exception as e:
            messagebox.showerror("Error", f"Cell detection failed:\n{e}")
            return
        n = len(segmentation.cell_ids(labels))
        info = f"{label}: {n} cells (th={th}, size {min_size}–{max_size})"
        self._set_cells(labels, info, mask=mask)
        self.info_label.config(text=info)

    def process_clusters_basic(self):
        """Threshold, connected regions, keep regions within Min/Max Size."""
        self._run_detection("basic")

    def process_clusters_advanced(self):
        """Like basic, but regions above Max Size are split around variability peaks."""
        self._run_detection("advanced")

    def decompose_clusters(self):
        """Like basic, but regions above Max Size are split with KMeans."""
        self._run_detection("kmeans")

    def load_cells(self):
        """Load cells saved with "Save Cells", or a legacy coordinates .txt."""
        filename = askopenfilename(
            parent=self.window,
            title="Load Cells",
            filetypes=[("Cells", "*.tif *.tiff *.npy *.txt"), ("Label image", "*.tif *.tiff *.npy"),
                       ("Coordinates (legacy)", "*.txt"), ("All Files", "*.*")]
        )
        if not filename:
            return
        try:
            labels = formats.load_cells(filename, shape=self.var_im.shape)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load cells:\n{e}", parent=self.window)
            return
        n = len(segmentation.cell_ids(labels))
        info = f"Loaded {n} cells from {os.path.basename(filename)}"
        self._set_cells(labels, info, mask=None)
        self.info_label.config(text=info)

    # ---------------------------------------------------------- selection

    def select_all_clusters(self):
        if not self._require_cells():
            return
        self.selected_clusters = self.cell_id_list
        self.update_cluster_colors()
        self.update_selection_list()

    def _update_selection_mode(self):
        self.selection_mode = self.selection_mode_var.get()

    def _on_region_selected(self, eclick, erelease):
        """Callback for rectangular region selection"""
        if self.labels is None:
            return
        x_min, x_max = sorted((eclick.xdata, erelease.xdata))
        y_min, y_max = sorted((eclick.ydata, erelease.ydata))

        changed = False
        for cid in self.cell_id_list:
            cx, cy = self._centroid_plot_xy(cid)
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                if self.selection_mode == 'add' and cid not in self.selected_clusters:
                    self.selected_clusters.append(cid)
                    changed = True
                elif self.selection_mode == 'remove' and cid in self.selected_clusters:
                    self.selected_clusters.remove(cid)
                    changed = True
        if changed:
            self.update_cluster_colors()
            self.update_selection_list()

    def on_cluster_click(self, event):
        """Toggle the clicked cell (left button, real clicks only)."""
        if not hasattr(event, 'mouseevent') or event.mouseevent is None:
            return
        if getattr(event.mouseevent, 'button', 1) != 1:
            return
        for scatter, cid in self.scatter_objects:
            if event.artist == scatter:
                if cid in self.selected_clusters:
                    self.selected_clusters.remove(cid)
                else:
                    self.selected_clusters.append(cid)
                self.update_cluster_colors()
                self.update_selection_list()
                return

    def update_cluster_colors(self):
        """Red + ID label for selected cells, their own color for the rest."""
        for ann in getattr(self, '_cluster_annotations', []):
            try:
                ann.remove()
            except Exception:
                pass
        self._cluster_annotations = []

        for scatter, cid in self.scatter_objects:
            if cid in self.selected_clusters:
                scatter.set_color('red')
                scatter.set_alpha(1.0)
                scatter.set_zorder(3)
                cx, cy = self._centroid_plot_xy(cid)
                ann = scatter.axes.annotate(
                    str(cid), (cx, cy),
                    fontsize=7, fontweight='bold', color='white',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.8),
                    zorder=5
                )
                self._cluster_annotations.append(ann)
            else:
                scatter.set_color(self.cluster_colors.get(cid, 'blue'))
                scatter.set_alpha(0.7)
                scatter.set_zorder(2)

        if hasattr(self, 'current_canvas'):
            self.current_canvas.draw()

    def update_selection_list(self):
        self.selected_listbox.delete(0, tk.END)
        for cid in self._selected_sorted():
            self.selected_listbox.insert(tk.END, f"Cell {cid} ({len(self.cell_pix[cid])} pts)")
        self.selection_info.config(text=f"{len(self.selected_clusters)} cells\nselected")

    def remove_selected_from_list(self):
        selection = self.selected_listbox.curselection()
        if not selection:
            return
        cid = self._selected_sorted()[selection[0]]
        if cid in self.selected_clusters:
            self.selected_clusters.remove(cid)
            self.update_cluster_colors()
            self.update_selection_list()

    def clear_selection(self):
        self.selected_clusters = []
        self.update_cluster_colors()
        self.update_selection_list()

    # ------------------------------------------------------------- export

    def _selected_signals(self):
        """Signals of the selected cells (frames x "cell_<id>")."""
        return signals.extract_signals(self.img_array, self.labels, self._selected_sorted())

    def save_selected_signals(self):
        """Save the mean signal of each selected cell (shared NecLab signals format)."""
        if not self._require_cells():
            return
        if not self.selected_clusters:
            messagebox.showwarning("Warning", "No cells selected")
            return
        filename = asksaveasfilename(
            parent=self.window,
            initialfile='cell_signals.csv',
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"),
                       ("NumPy files", "*.npy"), ("All Files", "*.*")]
        )
        if not filename:
            return
        try:
            formats.write_signals(self._selected_signals(), filename)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save signals:\n{e}", parent=self.window)
            return
        messagebox.showinfo("Success", f"Signals saved to {filename}", parent=self.window)

    def save_cells(self, only_selected=None):
        """Save the cells as a label image (+ cell table .csv next to it).

        only_selected: True/False, or None to ask when there is a selection.
        """
        if not self._require_cells():
            return
        if only_selected is None:
            only_selected = False
            if self.selected_clusters:
                answer = messagebox.askyesnocancel(
                    "Save Cells",
                    f"Save only the {len(self.selected_clusters)} selected cells?\n\n"
                    f"Yes = selected cells only, No = all {len(self.cell_pix)} cells",
                    parent=self.window)
                if answer is None:
                    return
                only_selected = answer
        labels = (segmentation.subset_labels(self.labels, self.selected_clusters)
                  if only_selected else self.labels)
        filename = asksaveasfilename(
            parent=self.window,
            initialfile='cells.tif',
            defaultextension=".tif",
            filetypes=[("Label image (TIFF)", "*.tif"), ("Label image (NumPy)", "*.npy"),
                       ("All Files", "*.*")]
        )
        if not filename:
            return
        try:
            img_path, table_path = formats.save_cells(labels, filename)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save cells:\n{e}", parent=self.window)
            return
        messagebox.showinfo("Success", f"Cells saved to:\n{img_path}\n{table_path}", parent=self.window)

    def _on_window_close(self):
        plt.close('all')
        self.window.destroy()

    # -------------------------------------------------------- correlation

    def use_selected_clusters(self):
        if not self.selected_clusters:
            messagebox.showwarning("Warning", "No cells selected")
            return
        self.show_correlation_analysis()

    def show_correlation_analysis(self):
        """Window with the selected cells' signals and their correlation."""
        corr_window = tk.Toplevel(self.window)
        corr_window.title("Correlation Analysis - Selected Cells")
        corr_window.geometry("1200x800")

        main_frame = tk.Frame(corr_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        info_text = f"Analyzing correlations between {len(self.selected_clusters)} selected cells"
        tk.Label(control_frame, text=info_text, font=("Arial", 12, "bold")).pack(pady=5)

        corr_button_frame = tk.Frame(control_frame)
        corr_button_frame.pack(fill=tk.X, pady=5)

        tk.Label(corr_button_frame, text="Correlation Type:").pack(side=tk.LEFT, padx=(0, 10))
        for method in correlation.CORRELATION_METHODS:
            tk.Button(corr_button_frame, text=method.capitalize(),
                      command=lambda m=method: self.calculate_correlation(m)).pack(side=tk.LEFT, padx=5)

        tk.Frame(corr_button_frame, height=2, bg="gray").pack(side=tk.LEFT, fill=tk.X, padx=20)

        tk.Button(corr_button_frame, text="Export Time Series",
                  command=self.export_time_series).pack(side=tk.LEFT, padx=5)
        tk.Button(corr_button_frame, text="Export Cells",
                  command=lambda: self.save_cells(only_selected=True)).pack(side=tk.LEFT, padx=5)
        tk.Button(corr_button_frame, text="Generate Report",
                  command=self.generate_report).pack(side=tk.LEFT, padx=5)

        self.corr_viz_frame = tk.Frame(main_frame)
        self.corr_viz_frame.pack(fill=tk.BOTH, expand=True)

        self.show_initial_time_series()

    @property
    def cluster_labels(self):
        return [f"Cell {signals.parse_cell_column(c)}" for c in self.signals_df.columns]

    def _plot_signals(self, ax):
        for col, label in zip(self.signals_df.columns, self.cluster_labels):
            ax.plot(self.signals_df[col].to_numpy(), label=label, alpha=0.8)
        ax.set_title('Time Series of Selected Cells')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Average Intensity')
        ax.grid(True, alpha=0.3)

    def _plot_corr_matrix(self, ax, corr_matrix, title):
        labels = self.cluster_labels
        im = ax.imshow(corr_matrix.values, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        ax.set_title(title)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontweight="bold")
        plt.colorbar(im, ax=ax)

    def _show_corr_figure(self, fig):
        for widget in self.corr_viz_frame.winfo_children():
            widget.destroy()
        if getattr(self, 'current_corr_fig', None) is not None:
            plt.close(self.current_corr_fig)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.corr_viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_corr_fig = fig

    def show_initial_time_series(self):
        """Signals of the selected cells + a preliminary Pearson matrix."""
        self.signals_df = self._selected_signals()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self._plot_signals(ax1)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if self.signals_df.shape[1] > 1:
            corr_matrix = correlation.correlation_matrix(self.signals_df, 'pearson')
            self._plot_corr_matrix(ax2, corr_matrix, 'Preliminary Correlation Matrix (Pearson)')
        else:
            ax2.text(0.5, 0.5, 'Select at least 2 cells\nfor correlation analysis',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Correlation Matrix')

        self._show_corr_figure(fig)

    def calculate_correlation(self, method):
        if self.signals_df is None or self.signals_df.shape[1] < 2:
            messagebox.showwarning("Warning", "You need at least 2 selected cells")
            return
        corr_matrix = correlation.correlation_matrix(self.signals_df, method)
        self.update_correlation_display(corr_matrix, method.capitalize())

    def update_correlation_display(self, corr_matrix, method_name):
        """Signals, correlation matrix, histogram of correlations, dendrogram."""
        from scipy.cluster.hierarchy import dendrogram

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        self._plot_signals(ax1)
        ax1.set_ylabel('Intensity')
        ax1.legend()

        self._plot_corr_matrix(ax2, corr_matrix, f'Correlation Matrix ({method_name})')

        values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        ax3.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Distribution of Correlations')
        ax3.set_xlabel('Correlation Value')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        try:
            dendrogram(correlation.abs_correlation_ward_linkage(corr_matrix),
                       labels=self.cluster_labels, ax=ax4)
            ax4.set_title('Cell Dendrogram')
            ax4.set_xlabel('Cells')
            ax4.set_ylabel('Distance')
        except Exception:
            ax4.text(0.5, 0.5, 'Error generating dendrogram', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Dendrogram')

        self._show_corr_figure(fig)
        self.current_corr_matrix = corr_matrix
        self.current_method = method_name

    def export_time_series(self):
        """Save the selected cells' signals (shared NecLab signals format)."""
        if self.signals_df is None or self.signals_df.empty:
            messagebox.showwarning("Warning", "No data to export")
            return
        filename = asksaveasfilename(
            initialfile='cell_signals.csv',
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"),
                       ("NumPy files", "*.npy"), ("All Files", "*.*")]
        )
        if not filename:
            return
        try:
            formats.write_signals(self.signals_df, filename)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save signals:\n{e}")
            return
        messagebox.showinfo("Success", f"Time series exported to {filename}")

    def generate_report(self):
        """Plain-text report of the detection settings, cells and correlations."""
        if not hasattr(self, 'current_corr_matrix'):
            messagebox.showwarning("Warning", "First calculate a correlation")
            return
        filename = asksaveasfilename(
            initialfile='cell_analysis_report.txt',
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All Files", "*.*")]
        )
        if not filename:
            return
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("CELL ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Variability method: {self.method_name}\n")
            f.write(f"Cells: {self.detection_info}\n")
            f.write(f"Selected cells: {len(self.selected_clusters)}\n")
            f.write(f"Correlation method: {self.current_method}\n\n")

            f.write("CELL DETAILS:\n")
            f.write("-" * 25 + "\n")
            for cid in self._selected_sorted():
                f.write(f"Cell {cid}: {len(self.cell_pix[cid])} pixels\n")

            f.write(f"\nCORRELATION MATRIX ({self.current_method}):\n")
            f.write("-" * 35 + "\n")
            f.write(self.current_corr_matrix.to_string())
            f.write("\n\n")

            f.write("CORRELATION STATISTICS:\n")
            f.write("-" * 30 + "\n")
            values = self.current_corr_matrix.values[np.triu_indices_from(self.current_corr_matrix.values, k=1)]
            f.write(f"Average correlation: {np.mean(values):.3f}\n")
            f.write(f"Maximum correlation: {np.max(values):.3f}\n")
            f.write(f"Minimum correlation: {np.min(values):.3f}\n")
            f.write(f"Standard deviation: {np.std(values):.3f}\n")
        messagebox.showinfo("Success", f"Report generated at {filename}")

    # ----------------------------------------------------------- 3D / image

    def show_3d_surface(self):
        """3D view of the variability surface."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

        window_3d = tk.Toplevel(self.window)
        window_3d.title(f"3D Surface - {self.method_name}")
        window_3d.geometry("800x600")

        # Subsample so the view stays responsive (at most ~100 points per side)
        step = max(1, min(self.var_im.shape) // 100)
        Z = self.var_im[::step, ::step]
        X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel(f'{self.method_name}')
        ax.set_title(f'Variability Surface - {self.method_name}')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=self.method_name)

        canvas = FigureCanvasTkAgg(fig, master=window_3d)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(window_3d)
        control_frame.pack(fill=tk.X, pady=5)

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
        """Save the four-panel figure."""
        if getattr(self, 'current_fig', None) is None:
            return
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
    """Open the analysis window for `img_array` (the raw stack) and `method`."""
    if img_array is None or len(img_array) == 0:
        messagebox.showwarning("Warning", "No image loaded")
        return None
    return VariabilityAnalysisWindow(img_array, method, main_window)


def get_variability_methods():
    """Names of the variability methods, in menu order."""
    return variability.method_names()
