"""
Cell detection view ("Cell Detection - <name>"), a tab of the main window,
with its "Signals & Correlation" and "3D View" tabs.

This module is user interface only: every computation is delegated to
neclab_core (variability maps, detection, signal extraction, correlation,
file formats, validation). Cells are kept as a label image whose values
are the cells' permanent IDs, so the IDs shown here are the same ones
written to every exported file.

The view covers pipeline stages 2-4 (variability map, candidate mask,
cells) and can start from any of them: from the recording (the map is
computed), from a saved map, from a saved mask or from saved cells. What
needs data that isn't loaded is switched off with a message saying what
to load; e.g. signals need the recording (Data -> Attach Recording).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import colorsys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import asksaveasfilename, askopenfilename

from neclab_core import analysis as core_analysis
from neclab_core import correlation, events, normalization, segmentation, signals, stages, variability
from neclab_core.validation import ValidationError
from neclab_core.project import DONE, LOADED
from ui_tabs import add_tab

# Detection mode -> (label, split method passed to neclab_core.segmentation).
_DETECTION_MODES = {
    "basic": ("Basic", "none"),
    "advanced": ("Advanced (3D peaks)", "peaks"),
    "kmeans": ("KMeans split", "kmeans"),
}

_DEFAULT_THRESHOLDS = dict(variability.VARIABILITY_METHODS)


def default_threshold_for(name, processed_image):
    """Starting threshold for a map: the method's default when the map is
    one of the 7 variability methods, else the map's 95th percentile."""
    if name in _DEFAULT_THRESHOLDS:
        return _DEFAULT_THRESHOLDS[name]
    if processed_image is None:
        return 100
    finite = np.asarray(processed_image, dtype=float)
    finite = finite[np.isfinite(finite)]
    return max(1, int(round(float(np.percentile(finite, 95))))) if finite.size else 100


def ask_choice(parent, title, prompt, options):
    """Small modal list dialog; returns the chosen option or None."""
    if len(options) == 1:
        return options[0]
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    tk.Label(dialog, text=prompt, anchor='w').pack(fill='x', padx=10, pady=(10, 4))
    box = tk.Listbox(dialog, height=min(10, len(options)), width=40, exportselection=False)
    for opt in options:
        box.insert(tk.END, opt)
    box.selection_set(0)
    box.pack(fill='both', expand=True, padx=10)
    result = {'value': None}

    def ok(_event=None):
        sel = box.curselection()
        if sel:
            result['value'] = options[sel[0]]
        dialog.destroy()

    box.bind('<Double-Button-1>', ok)
    buttons = tk.Frame(dialog)
    buttons.pack(pady=8)
    tk.Button(buttons, text="OK", width=10, command=ok).pack(side=tk.LEFT, padx=4)
    tk.Button(buttons, text="Cancel", width=10, command=dialog.destroy).pack(side=tk.LEFT, padx=4)
    parent.wait_window(dialog)
    return result['value']


def show_warnings(parent, warnings, title="Loaded with warnings"):
    if warnings:
        messagebox.showwarning(title, "\n\n".join(warnings), parent=parent)


class VariabilityAnalysisWindow:
    """
    Window for variability analysis, cell detection and cell selection.

    Args:
        img_array: the raw recording (frames, height, width), or None.
        method: variability method index, used when img_array is given and
            var_im isn't.
        main_window: parent Tk window.
        var_im, map_name: a variability map to start from (stage 2).
        mask: a candidate-pixel mask to start from (stage 3).
        labels: cells to start from (stage 4).
        source: short description of where the starting data came from.
        project: neclab_core.project.Project that this window reports what
            it computes or loads to (stages 0 and 2-7), or None.
        on_recording_attached: callback(stack, path, metadata) when a
            recording is attached here, so the main window can show it.
        restore: saved view state to restore (from a project):
            {"detection_mask", "threshold", "min_size", "max_size", "selected"}.
        notebook: the main window's ttk.Notebook: the view, its signals &
            correlation analysis and its 3D views are tabs of it. Without
            one they are separate windows.
    """
    def __init__(self, img_array, method, main_window, var_im=None, map_name=None,
                 mask=None, labels=None, source="", project=None, on_recording_attached=None,
                 restore=None, notebook=None):
        self.img_array = img_array
        self.method = method
        self.main_window = main_window
        self.notebook = notebook
        self.tab = None             # ui_tabs.Tab of this view (self.window is its frame)
        self.corr_tab = None        # the signals & correlation analysis tab, when open
        self.extra_tabs = []        # 3D views opened from here
        self.project = project
        self.on_recording_attached = on_recording_attached
        restore = restore or {}

        computed_map = var_im is None and img_array is not None and method is not None
        if computed_map:
            var_im, _th, map_name = variability.calculate_variability(img_array, method)
        self._set_map_data(var_im, map_name)
        if computed_map:
            self._report("features", {"maps": {self.method_name: self.var_im}},
                         params={"method": self.method_name})

        self.loaded_mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.mask = restore.get("detection_mask")  # mask from the last threshold-based detection
        self.labels = None          # label image: 0 = background, n = cell with ID n
        self.cell_pix = {}          # {cell_id: (N, 2) array of (row, col)}
        self.detection_info = ""    # how the current cells were obtained
        self.selected_clusters = []  # selected cell IDs, in selection order
        self.scatter_objects = []
        self.rect_selector = None
        self.selection_mode = 'add'
        self.cluster_colors = {}
        self.signals_df = None      # frames x selected cells ("cell_<id>" columns), stage 5 (raw)
        self._signal_cache = {}     # {cell_id: signal} for the current cells + recording
        self._auto_signals = False  # stages 5-6 follow the selection once the window is set up
        self.norm_df = None         # the same, normalized (stage 5; = signals_df when "none")
        self.norm_description = "raw"
        self.lowest_points_used = {}  # {column: frame indices of the lowest points used}
        self.events_df = None       # peaks of the normalized signals (stage 5 events), or None
        self.analysis_result = None      # neclab_core.analysis.AnalysisResult behind norm_df/events_df
        self.current_corr_matrix = None  # correlation of the normalized signals (stage 6), or None
        self._corr_basis = None          # the normalized signals current_corr_matrix was computed from
        self.current_method = None       # its method, for display ("Pearson")
        self.corr_window = None
        self._analyzer = core_analysis.Analyzer()   # normalization + events, each cell's peaks cached
        self._load_analysis_settings()
        self.current_fig = None

        self.create_window()
        for key, var in (("threshold", self.threshold_var), ("min_size", self.min_size_var),
                         ("max_size", self.max_size_var)):
            if restore.get(key) is not None:
                var.set(restore[key])
        if labels is not None:
            n = len(segmentation.cell_ids(labels))
            detection_mask = self.mask
            self._set_cells(np.asarray(labels, dtype=np.int32), f"Loaded {n} cells {source}".strip(),
                            mask=detection_mask)
            selected = [int(c) for c in restore.get("selected") or [] if int(c) in self.cell_pix]
            if selected:
                self.selected_clusters = selected
                self.update_cluster_colors()
                self.update_selection_list()
            self.info_label.config(text=self.detection_info)
        else:
            if self.loaded_mask is not None:
                self.info_label.config(text=f"Using loaded mask {source} (threshold not used)".strip())
            elif source:
                self.info_label.config(text=f"Loaded {source}")
            self.update_display()
        self._auto_signals = True
        self._update_pipeline(selection_changed=False)   # restored selection: only stages that don't match

    # ------------------------------------------------------------ helpers

    def _set_map_data(self, var_im, name):
        self.var_im = None if var_im is None else np.asarray(var_im)
        self.processed_image = None if var_im is None else variability.sharpen(self.var_im)
        self.method_name = name or ("Variability map" if var_im is not None else "Loaded data")
        self.default_th = default_threshold_for(self.method_name, self.processed_image)

    def _report(self, stage_key, items, params=None, origin="computed", path=None):
        """Record a stage's new data in the project (if there is one)."""
        if self.project is None:
            return
        source = {"path": os.path.abspath(path)} if path else {}
        self.project.set_stage(stage_key, items, params=params, origin=origin, source=source)

    def _report_selection(self):
        if self.project is not None:
            self.project.set_ui("selected_cells", self._selected_sorted())

    @property
    def frame_shape(self):
        for image in (self.var_im, self.loaded_mask, self.labels):
            if image is not None:
                return tuple(np.asarray(image).shape)
        if self.img_array is not None:
            return tuple(np.asarray(self.img_array).shape[1:])
        return None

    def _context(self):
        return stages.Context(stack=self.img_array,
                              maps=None if self.var_im is None else {self.method_name: self.var_im},
                              mask=self.loaded_mask, labels=self.labels)

    @property
    def cell_id_list(self):
        return sorted(self.cell_pix)

    def _set_cells(self, labels, info, mask=None):
        """Replace the current cells (clears the selection and the results
        of a previous correlation analysis, which were for the old cells)."""
        self.labels = labels
        self.mask = mask
        self.cell_pix = segmentation.cell_pixels(labels)
        self._signal_cache = {}
        self._analyzer.clear()
        self.detection_info = info
        self.selected_clusters = []
        self.cluster_colors = self._generate_cluster_colors(self.cell_id_list)
        self._reset_correlation_results()
        self.update_display()
        self.update_selection_list()

    def _reset_correlation_results(self):
        """Close the correlation window and forget its results: they belong
        to the previous cells (from PR #52)."""
        self._close_correlation_window()
        self.signals_df = None
        self.norm_df = None
        self.norm_description = "raw"
        self.lowest_points_used = {}
        self.events_df = None
        self.analysis_result = None
        self.current_corr_matrix = None
        self._corr_basis = None
        self.current_method = None

    def _close_correlation_window(self):
        corr_window = getattr(self, 'corr_window', None)
        if corr_window is not None:
            try:
                corr_window.destroy()      # also removes its tab
            except tk.TclError:
                pass
        self.corr_window = None
        self.corr_tab = None
        if getattr(self, 'current_corr_fig', None) is not None:
            plt.close(self.current_corr_fig)
        self.current_corr_fig = None

    def _generate_cluster_colors(self, ids):
        """Random color per cell ID, excluding red (reserved for selection)."""
        rng = np.random.RandomState(42)
        colors = {}
        for cid in ids:
            hue = rng.uniform(0.08, 0.88)   # avoid red: [0.0, 0.05] and [0.92, 1.0]
            sat = rng.uniform(0.5, 1.0)
            val = rng.uniform(0.6, 1.0)
            colors[cid] = colorsys.hsv_to_rgb(hue, sat, val)
        return colors

    def _centroid_plot_xy(self, cid):
        """Centroid of a cell in plot coordinates (x = column, y = flipped row)."""
        pts = self.cell_pix[cid]
        return np.mean(pts[:, 1]), self.frame_shape[0] - np.mean(pts[:, 0])

    def _require_cells(self):
        if not self.cell_pix:
            messagebox.showwarning("Warning", "First detect cells (Clustering menu) or load them "
                                   "(Data → Load Cells...)", parent=self.window)
            return False
        return True

    def _require_stack(self, action="This"):
        if self.img_array is None:
            messagebox.showwarning(
                "Recording Needed",
                f"{action} needs the recording.\n\n"
                "Use Data → Attach Recording... to load it (its frames must be the same size).",
                parent=self.window)
            return False
        return True

    def _require_map(self, action="This"):
        if self.var_im is None:
            messagebox.showwarning(
                "Variability Map Needed",
                f"{action} needs a variability map.\n\n"
                "Use Data → Load Variability Map..., or Data → Attach Recording... to compute one.",
                parent=self.window)
            return False
        return True

    def _selected_sorted(self):
        return sorted(self.selected_clusters)

    def _ask_open(self, item_key, title):
        return askopenfilename(parent=self.window, title=title,
                               filetypes=stages.open_filetypes(item_key))

    def _ask_save(self, item_key, title, initialfile=None):
        it = stages.item(item_key)
        default_ext = os.path.splitext(it.default_name)[1]
        return asksaveasfilename(parent=self.window, title=title,
                                 initialfile=initialfile or it.default_name,
                                 defaultextension=default_ext,
                                 filetypes=stages.save_filetypes(item_key))

    def _load(self, item_key, title, context=None):
        """Ask for a file and load/validate it. Returns stages.Loaded or None."""
        filename = self._ask_open(item_key, title)
        if not filename:
            return None
        try:
            loaded = stages.load_item(item_key, filename, context or self._context())
        except (ValidationError, ValueError, OSError) as e:
            messagebox.showerror("Can't Load File", f"{os.path.basename(filename)}:\n\n{e}", parent=self.window)
            return None
        show_warnings(self.window, loaded.warnings)
        return loaded

    def _save(self, item_key, data, title, initialfile=None):
        filename = self._ask_save(item_key, title, initialfile)
        if not filename:
            return
        try:
            written = stages.save_item(item_key, data, filename)
        except (ValidationError, ValueError, OSError) as e:
            messagebox.showerror("Can't Save", str(e), parent=self.window)
            return
        messagebox.showinfo("Saved", "Saved to:\n" + "\n".join(map(str, written)), parent=self.window)

    # ------------------------------------------------------------- layout

    def create_window(self):
        """The view: a tab of the main window (or a window, without a
        notebook) with its menus in the tab's bar."""
        self.tab = add_tab(self.notebook, self._title_text(), parent=self.main_window,
                           on_close=self._on_window_close, geometry="1400x800")
        self.window = self.tab.frame
        self._create_menu_bar()

        main_frame = tk.Frame(self.tab.body)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.create_controls(main_frame)

        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # The selection panel is packed first so it keeps its width when
        # the tab is narrower than the plots would like.
        self.selection_frame = tk.Frame(content_frame, width=200, relief=tk.RAISED, borderwidth=1)
        self.selection_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.selection_frame.pack_propagate(False)
        self.create_selection_panel()

        self.viz_frame = tk.Frame(content_frame)
        self.viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _title_text(self):
        suffix = "" if self.img_array is not None else " (no recording)"
        return f"Cell Detection - {self.method_name}{suffix}"

    def _update_title(self):
        self.tab.set_title(self._title_text())

    def show(self):
        """Bring this view to the front."""
        self.tab.show()

    def exists(self):
        return self.tab is not None and self.tab.exists()

    def _create_menu_bar(self):
        """The view's menus (Data, Clustering, Selection, Visualization,
        Export), in its tab's bar."""
        menu_data = self.tab.add_menu("Data", lambda m: None)
        menu_data.add_command(label="Attach Recording...", command=self.attach_recording)
        menu_data.add_separator()
        menu_data.add_command(label="Load Variability Map...", command=self.load_map)
        menu_data.add_command(label="Load Mask...", command=self.load_mask)
        menu_data.add_command(label="Load Cells...", command=self.load_cells)
        menu_data.add_separator()
        menu_data.add_command(label="Stop Using Loaded Mask", command=self.clear_loaded_mask)

        menu_clustering = self.tab.add_menu("Clustering", lambda m: None)
        menu_clustering.add_command(label="Process Cluster (Basic)", command=self.process_clusters_basic)
        menu_clustering.add_command(label="Process Cluster (Advanced)", command=self.process_clusters_advanced)
        menu_clustering.add_separator()
        menu_clustering.add_command(label="Decompose Large Clusters (KMeans)", command=self.decompose_clusters)

        menu_seleccion = self.tab.add_menu("Selection", lambda m: None)
        menu_seleccion.add_command(label="Select All", command=self.select_all_clusters)
        menu_seleccion.add_command(label="Clear Selection", command=self.clear_selection)

        menu_visual = self.tab.add_menu("Visualization", lambda m: None)
        menu_visual.add_command(label="3D View", command=self.show_3d_surface)

        menu_exportar = self.tab.add_menu("Export", lambda m: None)
        menu_exportar.add_command(label="Save Image", command=self.save_image)
        menu_exportar.add_separator()
        menu_exportar.add_command(label="Save Variability Map...", command=self.save_map)
        menu_exportar.add_command(label="Save Mask...", command=self.save_mask)
        menu_exportar.add_command(label="Save Cells (Label Image)...", command=self.save_cells)
        menu_exportar.add_command(label="Save Signals (CSV / XLSX / NPY)...", command=self.save_selected_signals)
        menu_exportar.add_separator()
        menu_exportar.add_command(label="Use Selected (Correlations)", command=self.use_selected_clusters)

    def create_controls(self, parent):
        control_frame = tk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(control_frame, text="Threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.threshold_var = tk.IntVar(value=self.default_th)
        ttk.Spinbox(control_frame, from_=1, to=10 ** 9, textvariable=self.threshold_var,
                    width=10).pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(control_frame, text="Min Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.min_size_var = tk.IntVar(value=20)
        ttk.Spinbox(control_frame, from_=1, to=500, textvariable=self.min_size_var, width=8).pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(control_frame, text="Max Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.max_size_var = tk.IntVar(value=200)
        ttk.Spinbox(control_frame, from_=1, to=1000, textvariable=self.max_size_var, width=8).pack(side=tk.LEFT, padx=(0, 15))

        self.info_label = tk.Label(control_frame, text="Use the menu to process clusters")
        self.info_label.pack(side=tk.LEFT, padx=10)

    def create_selection_panel(self):
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

    @staticmethod
    def _placeholder(ax, text, title):
        ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.set_title(title)
        ax.axis('off')

    def update_display(self):
        """Redraw the four panels."""
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        if self.current_fig is not None:
            plt.close(self.current_fig)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # 1. Variability map
        if self.var_im is not None:
            im1 = ax1.imshow(self.var_im, cmap='viridis')
            ax1.set_title(f'{self.method_name}')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1)
        else:
            self._placeholder(ax1, 'No variability map\n(Data → Load Variability Map\nor Attach Recording)',
                              'Variability Map')

        # 2. Sharpened map (the one thresholded)
        if self.processed_image is not None:
            im2 = ax2.imshow(self.processed_image, cmap='viridis')
            ax2.set_title('Processed Image (Sharpened)')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2)
        else:
            self._placeholder(ax2, 'No variability map', 'Processed Image (Sharpened)')

        # 3. Candidate-pixel mask
        if self.loaded_mask is not None:
            ax3.imshow(self.loaded_mask, cmap='gray')
            ax3.set_title('Loaded mask (threshold not used)')
            ax3.axis('off')
        elif self.mask is not None:
            ax3.imshow(self.mask, cmap='gray')
            ax3.set_title(f'Binarization (th={self.threshold_var.get()})')
            ax3.axis('off')
        elif self.labels is not None:
            ax3.imshow(self.labels > 0, cmap='gray')
            ax3.set_title('Loaded cells (mask)')
            ax3.axis('off')
        else:
            self._placeholder(ax3, 'Apply binarization', 'Binarization')

        # 4. Cells (clickable)
        self.scatter_objects = []
        self._cluster_annotations = []
        if self.labels is not None:
            height, width = self.frame_shape
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
            ax4.set_xlim(0, width)
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

    # ------------------------------------------------- data (enter a stage)

    def attach_recording(self):
        """Load the recording these maps/masks/cells belong to (needed for signals)."""
        ctx = self._context()
        ctx.stack = None
        loaded = self._load("stack", "Attach Recording", ctx)
        if loaded is None:
            return
        self.img_array = loaded.data
        self._signal_cache = {}
        self._analyzer.clear()
        if self.on_recording_attached is not None:
            self.on_recording_attached(loaded.data, loaded.path, loaded.metadata)
        elif self.project is not None:
            self.project.set_recording(loaded.data, loaded.path, loaded.metadata)
        msg = f"Recording attached: {loaded.data.shape[0]} frames"
        if self.var_im is None:
            names = variability.method_names()
            choice = ask_choice(self.window, "Variability Map",
                                "Compute a variability map from the recording?\n"
                                "(Cancel to keep working without one.)", names)
            if choice is not None:
                var_im, _th, name = variability.calculate_variability(self.img_array, names.index(choice))
                self._set_map_data(var_im, name)
                self._report("features", {"maps": {name: var_im}}, params={"method": name})
                self.threshold_var.set(self.default_th)
                msg += f", {name} computed"
        self._update_title()
        self.info_label.config(text=msg)
        self.update_display()
        self._update_pipeline()   # stages 5-6 of the cells already selected

    def load_map(self):
        """Start from a saved variability map (stage 2)."""
        ctx = self._context()
        ctx.maps = None
        loaded = self._load("maps", "Load Variability Map", ctx)
        if loaded is None:
            return
        name = ask_choice(self.window, "Choose Map", "The file has several maps. Which one?", list(loaded.data))
        if name is None:
            return
        self._set_map_data(loaded.data[name], name)
        self._report("features", {"maps": {name: self.var_im}}, params={"map": name},
                     origin="imported", path=loaded.path)
        self.threshold_var.set(self.default_th)
        self._update_title()
        self.info_label.config(text=f"Loaded map '{name}' from {os.path.basename(loaded.path)}")
        self.update_display()

    def load_mask(self):
        """Start from a saved candidate-pixel mask (stage 3)."""
        ctx = self._context()
        ctx.mask = None
        loaded = self._load("mask", "Load Mask", ctx)
        if loaded is None:
            return
        self.loaded_mask = loaded.data
        self._report("mask", {"mask": self.loaded_mask}, origin="imported", path=loaded.path)
        self.info_label.config(
            text=f"Using mask {os.path.basename(loaded.path)} (threshold not used) - run a Clustering method")
        self.update_display()

    def clear_loaded_mask(self):
        if self.loaded_mask is None:
            return
        self.loaded_mask = None
        self.info_label.config(text="Loaded mask cleared: detection uses the threshold again")
        self.update_display()

    def load_cells(self):
        """Start from saved cells (stage 4): label image, ImageJ ROIs or legacy .txt."""
        ctx = self._context()
        ctx.labels = None
        filename = self._ask_open("cells", "Load Cells")
        if not filename:
            return
        try:
            if filename.lower().endswith(('.roi', '.zip', '.txt')) and ctx.frame_shape is None:
                raise ValidationError("ImageJ ROIs and coordinate files need the image size: "
                                      "attach the recording or load a map first.")
            loaded = stages.load_item("cells", filename, ctx)
        except (ValidationError, ValueError, OSError) as e:
            messagebox.showerror("Can't Load File", f"{os.path.basename(filename)}:\n\n{e}", parent=self.window)
            return
        show_warnings(self.window, loaded.warnings)
        n = len(segmentation.cell_ids(loaded.data))
        info = f"Loaded {n} cells from {os.path.basename(filename)}"
        self._report("cells", {"cells": loaded.data}, origin="imported", path=filename)
        self._set_cells(loaded.data, info, mask=None)
        self.info_label.config(text=info)

    # ---------------------------------------------------------- detection

    def _run_detection(self, mode):
        label, split = _DETECTION_MODES[mode]
        try:
            th = self.threshold_var.get()
            min_size = self.min_size_var.get()
            max_size = self.max_size_var.get()
        except tk.TclError:
            messagebox.showerror("Error", "Threshold, Min Size and Max Size must be numbers", parent=self.window)
            return

        if self.loaded_mask is None and self.processed_image is None:
            messagebox.showwarning(
                "Nothing to Detect From",
                "Load a variability map or a mask (Data menu), or attach the recording.",
                parent=self.window)
            return
        if split != "none" and not self._require_map(f"The {label} method"):
            return

        try:
            if self.loaded_mask is not None:
                labels = segmentation.detect_cells_from_mask(
                    self.loaded_mask, self.var_im, min_size, max_size, split=split)
                mask, where = None, "loaded mask"
            else:
                labels, mask = segmentation.detect_cells(
                    self.var_im, self.processed_image, th, min_size, max_size, split=split)
                where = f"th={th}"
        except Exception as e:
            messagebox.showerror("Error", f"Cell detection failed:\n{e}", parent=self.window)
            return
        n = len(segmentation.cell_ids(labels))
        info = f"{label}: {n} cells ({where}, size {min_size}–{max_size})"
        if mask is not None:  # thresholded here (a loaded mask is already recorded)
            self._report("mask", {"mask": mask}, params={"threshold": th, "map": self.method_name})
        self._report("cells", {"cells": labels},
                     params={"method": label, "split": split, "min_size": min_size, "max_size": max_size,
                             "from": "loaded mask" if mask is None else "threshold"})
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
        self._report_selection()
        self._update_pipeline()
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

    # ------------------------------------------------- export (leave a stage)

    def save_map(self):
        if not self._require_map("Saving a variability map"):
            return
        safe = self.method_name.replace(" ", "_")
        self._save("maps", {self.method_name: self.var_im}, "Save Variability Map",
                   initialfile=f"{safe}.npz")

    def save_mask(self):
        mask = self.loaded_mask if self.loaded_mask is not None else self.mask
        if mask is None:
            messagebox.showwarning("No Mask", "Run a Clustering method first: the mask is made "
                                   "from the threshold when cells are detected.", parent=self.window)
            return
        self._save("mask", mask, "Save Mask")

    def _extract_selected(self):
        """Signals of the selected cells (frames x "cell_<id>", in ID order).

        Each cell's signal is computed once and cached until the cells or
        the recording change, so adding one cell to the selection only
        extracts that cell.
        """
        ids = self._selected_sorted()
        missing = [cid for cid in ids if cid not in self._signal_cache]
        if missing:
            new = signals.extract_signals(self.img_array, self.labels, missing)
            for col in new.columns:
                self._signal_cache[signals.parse_cell_column(col)] = new[col]
        if ids:
            df = pd.concat([self._signal_cache[cid] for cid in ids], axis=1)
        else:
            df = pd.DataFrame(index=pd.RangeIndex(self.img_array.shape[0], name="frame"))
        df.index = pd.RangeIndex(self.img_array.shape[0], name="frame")
        return df

    def _selected_signals(self):
        """Raw signals of the selected cells (frames x "cell_<id>")."""
        return self._extract_selected()

    # ---- stages 5 (signals, normalization & events) and 6 (correlation) follow the selection

    def _load_analysis_settings(self):
        """Normalization, peak and correlation settings: those the project
        was saved with, else the defaults (no normalization, no peaks,
        Pearson)."""
        rec = self.project.get("signals") if self.project is not None else None
        self.analysis_settings = core_analysis.settings_from_params(rec.params if rec is not None else None)
        self.peak_params = {}       # {method: params} last used for each peak method
        if self.analysis_settings["peak_method"] in events.PEAK_METHODS:
            self.peak_params[self.analysis_settings["peak_method"]] = dict(self.analysis_settings["peak_params"])
        rec = self.project.get("correlation") if self.project is not None else None
        method = str(rec.params.get("method") or "").lower() if rec is not None else ""
        self.corr_method = method if method in correlation.CORRELATION_METHODS else "pearson"

    def _needs_update(self, key):
        """True if stage `key` should be (re)computed: not run, out of date
        or missing. A stage that is up to date (computed or loaded) is kept."""
        return self.project is not None and self.project.status(key) not in (DONE, LOADED)

    def _clear_computed(self, key):
        rec = self.project.get(key) if self.project is not None else None
        if rec is not None and rec.origin == "computed":
            self.project.clear_stage(key)

    def _signals_stage_matches(self):
        """True if stage 5 is up to date and already holds the selected
        cells, normalized and searched for events with the current settings
        (e.g. a project was just reopened): then it is left as is, so the
        correlation computed from it stays up to date."""
        rec = self.project.get("signals") if self.project is not None else None
        if rec is None or not rec.available or self._needs_update("signals"):
            return False
        if (list(rec.params.get("cells") or []) != self._selected_sorted()
                or rec.items.get("signals") is None or rec.items.get("normalized_signals") is None):
            return False
        saved = core_analysis.settings_from_params(rec.params)
        current = {**self.analysis_settings,
                   "peak_params": dict(self.peak_params.get(self.analysis_settings["peak_method"], {}))}
        return core_analysis.settings_key(saved) == core_analysis.settings_key(current)

    def _update_pipeline(self, selection_changed=True):
        """Stages 5 and 6 follow the selection. Whenever the selected cells
        change (and the recording is loaded), their signals are extracted,
        normalized and searched for events with the last settings applied
        in the correlation tab (or the project's), and recorded as stage 5;
        their correlation, with the last method chosen, is recomputed and
        recorded as stage 6, and an open correlation tab is redrawn. The
        correlation is only recomputed here, when the selection changes (or
        with a correlation button): applying another normalization doesn't.
        Stages that are still up to date (e.g. a project just reopened,
        selection_changed=False) are left as they are. With no cells
        selected both are cleared; with a single cell there is no
        correlation, so stage 6 is cleared."""
        if not self._auto_signals or self.labels is None or self.img_array is None:
            return
        window_open = self._corr_window_open()
        if not self.selected_clusters:
            self.signals_df = self.norm_df = self.events_df = None
            self.lowest_points_used = {}
            self.current_corr_matrix = self._corr_basis = None
            self._clear_computed("signals")
            self._clear_computed("correlation")
            if window_open:
                self._redraw_correlation()
            return
        need_signals = self.project is not None and not self._signals_stage_matches()
        need_corr = need_signals or (selection_changed and self._needs_update("correlation"))
        if not (need_signals or need_corr or window_open):
            return
        if not self._compute_results(corr=need_corr or not self._corr_matches_selection()):
            return
        self._record_results(signals=need_signals, corr=need_corr)
        if window_open:
            self._redraw_correlation()

    def _note(self, text):
        """A message about the automatic stages, in the window's info line."""
        self.info_label.config(text=text)

    def _compute_results(self, interactive=False, corr=True):
        """Signals of the selected cells, their normalization and events
        (stage 5) and, if `corr`, their correlation (stage 6), in memory.
        `interactive`: problems are shown in dialogs (the correlation tab's
        buttons) instead of the info line. False if the signals can't be
        extracted."""
        try:
            self.signals_df = self._extract_selected()
        except Exception as e:      # e.g. recording and cells don't match
            self.signals_df = None
            self._note(f"Could not extract signals: {e}")
            return False
        self._run_analysis(interactive)
        if corr:
            self._compute_correlation()
        return True

    def _run_analysis(self, interactive=False):
        """Normalization and events of self.signals_df with self.analysis_settings. If a
        signal can't be normalized (e.g. its lowest-points baseline reaches
        zero), the raw signals are used and the user is told."""
        parent = self.corr_window if self._corr_window_open() else self.window
        try:
            result = self._analyzer.run(self.signals_df, self.analysis_settings)
        except ValueError as e:
            raw = {**self.analysis_settings, "method": "none", "smooth": False, "subtract": False}
            if interactive:
                messagebox.showwarning("Can't Normalize", f"{e}\n\nShowing the raw signals instead.",
                                       parent=parent)
                self.analysis_settings = raw
                self._set_controls_from_settings()
            else:
                self._note(f"Can't normalize ({e}): using the raw signals")
            result = self._analyzer.run(self.signals_df, raw)
        if result.events_error:
            if interactive:
                messagebox.showwarning("Can't Detect Peaks", result.events_error, parent=parent)
            else:
                self._note(result.events_error)
        self.analysis_result = result
        self.norm_df = result.norm_df
        self.events_df = result.events_df
        self.norm_description = result.description
        self.lowest_points_used = result.lowest_points_used
        self._update_peak_label()

    def _compute_correlation(self):
        """Stage 6: correlation of the normalized signals (2+ cells)."""
        if self.norm_df is not None and self.norm_df.shape[1] >= 2:
            self.current_corr_matrix = correlation.correlation_matrix(self.norm_df, self.corr_method)
            self.current_method = self.corr_method.capitalize()
            self._corr_basis = self.norm_df
        else:
            self.current_corr_matrix = self._corr_basis = None

    def _corr_matches_selection(self):
        """True if the correlation in memory is for the selected cells."""
        m = self.current_corr_matrix
        return m is not None and [str(c) for c in m.columns] == [
            signals.cell_column(c) for c in self._selected_sorted()]

    @property
    def corr_is_stale(self):
        """True if the correlation shown was computed from other normalized
        signals than the current ones (another normalization was applied
        since): it is updated when the selection changes, or with a
        correlation button."""
        if self.current_corr_matrix is None or self.norm_df is None:
            return False
        return self._corr_basis is None or not self._corr_basis.equals(self.norm_df)

    def _correlation_for_view(self):
        """The correlation to show when the correlation tab opens, without
        recomputing it: the one in memory, or else the project's (stage 6)
        if it is for the selected cells. Only if there is none for these
        cells is it computed (and recorded)."""
        if self._corr_matches_selection():
            return
        rec = self.project.get("correlation") if self.project is not None else None
        stored = self.project.data("correlation") if rec is not None else None
        ids = [signals.cell_column(c) for c in self._selected_sorted()]
        if stored is not None and [str(c) for c in stored.columns] == ids and len(ids) >= 2:
            self.current_corr_matrix = stored
            method = str(rec.params.get("method") or self.corr_method)
            self.current_method = method.capitalize()
            self._corr_basis = self.norm_df if not self._needs_update("correlation") else None
            return
        self._compute_correlation()
        self._record_results(signals=False, corr=self._needs_update("correlation"))

    def _record_results(self, signals=True, corr=True):
        """Record stage 5 and/or 6 from the results in memory."""
        if self.project is None or self.signals_df is None:
            return
        if signals:
            result = self.analysis_result
            self._report("signals", {"signals": self.signals_df, **result.items},
                         params={"cells": self._selected_sorted(), "measure": "mean of cell pixels",
                                 **result.params})
        if corr:
            if self.current_corr_matrix is not None:
                self._report_correlation(self.current_corr_matrix, self.corr_method)
            else:
                self._clear_computed("correlation")

    def _report_correlation(self, corr_matrix, method):
        self._report("correlation", {"correlation": corr_matrix},
                     params={"method": method.lower(), "signals": self.norm_description,
                             "clustering": correlation.LINKAGE_DESCRIPTION})

    def save_selected_signals(self):
        """Save the mean signal of each selected cell (shared NecLab signals format)."""
        if not self._require_cells() or not self._require_stack("Extracting signals"):
            return
        if not self.selected_clusters:
            messagebox.showwarning("Warning", "No cells selected", parent=self.window)
            return
        self._save("signals", self._selected_signals(), "Save Signals", initialfile="cell_signals.csv")

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
        self._save("cells", labels, "Save Cells")

    def _on_window_close(self):
        """Close this view with the tabs it opened (signals & correlation, 3D)."""
        self._close_correlation_window()
        for tab in self.extra_tabs:
            tab.close()
        self.extra_tabs = []
        if self.current_fig is not None:
            plt.close(self.current_fig)
            self.current_fig = None
        self.tab.close()

    close = _on_window_close

    # -------------------------------------------------------- correlation

    def use_selected_clusters(self):
        if not self.selected_clusters:
            messagebox.showwarning("Warning", "No cells selected", parent=self.window)
            return
        if not self._require_stack("The correlation analysis"):
            return
        self.show_correlation_analysis()

    def show_correlation_analysis(self):
        """Window with the selected cells' signals and their correlation.
        Only one at a time: opening it again replaces the previous one."""
        self._close_correlation_window()
        self.corr_tab = add_tab(self.notebook, f"Signals & Correlation - {self.method_name}",
                                parent=self.window, after=self.window if self.notebook is not None else None,
                                on_close=self._close_correlation_window)
        corr_window = self.corr_tab.frame
        self.corr_window = corr_window

        main_frame = tk.Frame(self.corr_tab.body)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.corr_info_label = tk.Label(control_frame, text="", font=("Arial", 12, "bold"), wraplength=1000)
        self.corr_info_label.pack(pady=5)

        # The controls are grouped by stage, in short rows, so that they all
        # fit in the tab (it is narrower than a separate window was).
        def row(parent):
            f = tk.Frame(parent)
            f.pack(fill=tk.X, pady=2)
            return f

        # ---- Stage 5: signal processing (normalization + peak detection).
        # The controls start from the last settings applied (or the project's).
        proc = tk.LabelFrame(control_frame, text=" Signal processing: normalization & peak detection (stage 5) ",
                             padx=8, pady=4)
        proc.pack(fill=tk.X, pady=(0, 4))
        settings = self.analysis_settings
        method, points = settings["method"], int(settings["lowest_points"])
        smooth, subtract = bool(settings["smooth"]), bool(settings["subtract"])
        self._norm_keys = list(normalization.NORMALIZATION_METHODS)
        self._norm_labels = list(normalization.NORMALIZATION_METHODS.values())

        norm_frame = row(proc)
        tk.Label(norm_frame, text="Normalization:").pack(side=tk.LEFT, padx=(0, 5))
        self.norm_method_var = tk.StringVar(value=normalization.NORMALIZATION_METHODS[method])
        ttk.Combobox(norm_frame, textvariable=self.norm_method_var, values=self._norm_labels,
                     state='readonly', width=36).pack(side=tk.LEFT, padx=(0, 15))
        tk.Label(norm_frame, text="Lowest points:").pack(side=tk.LEFT, padx=(0, 5))
        self.norm_points_var = tk.IntVar(value=points)
        ttk.Spinbox(norm_frame, from_=2, to=50, textvariable=self.norm_points_var,
                    width=5).pack(side=tk.LEFT, padx=(0, 15))
        tk.Button(norm_frame, text="Apply", command=self.apply_normalization).pack(side=tk.LEFT, padx=5)

        options_frame = row(proc)
        self.norm_smooth_var = tk.BooleanVar(value=smooth)
        tk.Checkbutton(options_frame, text="Smoothing",
                       variable=self.norm_smooth_var).pack(side=tk.LEFT, padx=(0, 10))
        self.norm_subtract_var = tk.BooleanVar(value=subtract)
        tk.Checkbutton(options_frame, text="Subtract lowest-points baseline (lowest points → 0)",
                       variable=self.norm_subtract_var).pack(side=tk.LEFT, padx=(0, 10))
        self.norm_show_points_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Show lowest points", variable=self.norm_show_points_var,
                       command=self._redraw_correlation).pack(side=tk.LEFT, padx=(0, 10))

        # Event (peak) detection on the normalized signals
        peak_frame = row(proc)
        tk.Label(peak_frame, text="Peaks:").pack(side=tk.LEFT, padx=(0, 5))
        self.peak_method_var = tk.StringVar(value=settings["peak_method"])
        peak_box = ttk.Combobox(peak_frame, textvariable=self.peak_method_var,
                                values=[events.NO_PEAKS] + list(events.PEAK_METHODS),
                                state='readonly', width=24)
        peak_box.pack(side=tk.LEFT, padx=(0, 10))
        peak_box.bind('<<ComboboxSelected>>', lambda e: self._on_peak_method_change(ask=True))
        tk.Button(peak_frame, text="Parameters...",
                  command=lambda: self._on_peak_method_change(ask=True)).pack(side=tk.LEFT, padx=5)
        self.peak_show_var = tk.BooleanVar(value=True)
        tk.Checkbutton(peak_frame, text="Show peaks", variable=self.peak_show_var,
                       command=self._redraw_correlation).pack(side=tk.LEFT, padx=(10, 10))
        self.peak_info_label = tk.Label(peak_frame, text="", fg='gray')
        self.peak_info_label.pack(side=tk.LEFT, padx=5)

        # ---- Stage 6: correlation, and the exports
        corr_box = tk.LabelFrame(control_frame, text=" Correlation (stage 6) & export ", padx=8, pady=4)
        corr_box.pack(fill=tk.X, pady=(0, 4))
        corr_button_frame = row(corr_box)
        tk.Label(corr_button_frame, text="Correlation Type:").pack(side=tk.LEFT, padx=(0, 10))
        for method in correlation.CORRELATION_METHODS:
            tk.Button(corr_button_frame, text=method.capitalize(),
                      command=lambda m=method: self.calculate_correlation(m)).pack(side=tk.LEFT, padx=5)
        tk.Label(corr_button_frame, text=f"Dendrogram: {correlation.LINKAGE_DESCRIPTION}",
                 fg='gray').pack(side=tk.LEFT, padx=15)

        export_frame = row(corr_box)
        tk.Label(export_frame, text="Export:").pack(side=tk.LEFT, padx=(0, 10))
        for text, command in (("Time Series", self.export_time_series),
                              ("Normalized", self.export_normalized),
                              ("Events", self.export_events),
                              ("Correlation", self.export_correlation),
                              ("Cells", lambda: self.save_cells(only_selected=True)),
                              ("Generate Report", self.generate_report)):
            tk.Button(export_frame, text=text, command=command).pack(side=tk.LEFT, padx=4)

        self.corr_viz_frame = tk.Frame(main_frame)
        self.corr_viz_frame.pack(fill=tk.BOTH, expand=True)

        # Stages 5-6 normally already follow the selection: compute stage 5
        # in memory for the plots (recording it only if it doesn't match),
        # and show the existing correlation rather than recomputing it.
        if self._compute_results(interactive=True, corr=False):
            if self.project is not None and not self._signals_stage_matches():
                self._record_results(signals=True, corr=False)
            self._correlation_for_view()
        self._redraw_correlation()

    @property
    def cluster_labels(self):
        return [f"Cell {signals.parse_cell_column(c)}" for c in self.signals_df.columns]

    # ---- normalization & events (stage 5): the window's controls

    def _corr_window_open(self):
        try:
            return self.corr_window is not None and bool(self.corr_window.winfo_exists())
        except tk.TclError:
            return False

    def _read_controls(self):
        """Settings from the correlation window's controls into
        self.analysis_settings (on Apply or when a peak method is chosen)."""
        if not self._corr_window_open():
            return
        label = self.norm_method_var.get()
        method = self._norm_keys[self._norm_labels.index(label)] if label in self._norm_labels else "none"
        try:
            points = int(self.norm_points_var.get())
        except (tk.TclError, ValueError):
            points = 0
        if points < 2:
            points = normalization.DEFAULT_LOWEST_POINTS
        peak_method = self.peak_method_var.get()
        if peak_method not in events.PEAK_METHODS:
            peak_method = events.NO_PEAKS
        self.analysis_settings = {
            "method": method, "lowest_points": points,
            "smooth": bool(self.norm_smooth_var.get()), "subtract": bool(self.norm_subtract_var.get()),
            "peak_method": peak_method, "peak_params": dict(self.peak_params.get(peak_method, {})),
        }

    def _set_controls_from_settings(self):
        if not self._corr_window_open():
            return
        s = self.analysis_settings
        self.norm_method_var.set(normalization.NORMALIZATION_METHODS[s["method"]])
        self.norm_points_var.set(int(s["lowest_points"]))
        self.norm_smooth_var.set(bool(s["smooth"]))
        self.norm_subtract_var.set(bool(s["subtract"]))
        self.peak_method_var.set(s["peak_method"])

    def _on_peak_method_change(self, ask=False):
        """A peak method was chosen (or Parameters... clicked): ask for its
        parameters, then detect the events (stage 5)."""
        method = self.peak_method_var.get()
        if method in events.PEAK_METHODS and (ask or method not in self.peak_params):
            from peak_functions import show_parameter_dialog
            title, specs = events.PEAK_METHODS[method]
            current = {**events.default_params(method), **self.peak_params.get(method, {})}
            specs = [{**spec, 'default': current[spec['key']]} for spec in specs]
            new_params = show_parameter_dialog(self.corr_window, title, specs)
            if new_params:
                self.peak_params[method] = new_params
            elif method not in self.peak_params:   # cancelled before ever choosing parameters
                self.peak_method_var.set(self.analysis_settings["peak_method"])
                return
        self.apply_normalization()

    def _update_peak_label(self):
        label = getattr(self, 'peak_info_label', None)
        if label is None or not self._corr_window_open():
            return
        if self.events_df is None:
            label.config(text="")
        else:
            n_cells = self.events_df["cell_id"].nunique() if len(self.events_df) else 0
            label.config(text=f"{len(self.events_df)} peaks in {n_cells} of {self.norm_df.shape[1]} cells")

    def export_events(self):
        """Save the detected peaks (stage 5 events format)."""
        if self.events_df is None:
            messagebox.showwarning("No Events", "Choose a peak method (Peaks) first.",
                                   parent=self.corr_window if self._corr_window_open() else self.window)
            return
        self._save("events", self.events_df, "Export Events", initialfile="events.csv")

    def _redraw_correlation(self):
        """Redraw the correlation window from the results in memory (after
        a selection change, Apply, a correlation button, or toggling
        'Show lowest points' / 'Show peaks')."""
        if not self._corr_window_open():
            return
        n = len(self.selected_clusters)
        text = f"Analyzing correlations between {n} selected cells"
        if self.corr_is_stale:
            text += ("  —  correlation not updated for this normalization: it updates when the "
                     "selection changes, or click a correlation type")
        self.corr_info_label.config(text=text, fg='#b45309' if self.corr_is_stale else 'black')
        self._update_peak_label()
        if self.signals_df is None or self.norm_df is None or self.norm_df.shape[1] == 0:
            for widget in self.corr_viz_frame.winfo_children():
                widget.destroy()
            if getattr(self, 'current_corr_fig', None) is not None:
                plt.close(self.current_corr_fig)
                self.current_corr_fig = None
            tk.Label(self.corr_viz_frame, fg='gray', font=("Arial", 12),
                     text="No cells selected.\n\nSelect cells in the cell detection tab "
                          "(click, drag or Select All): this tab follows the selection.").pack(expand=True)
            return
        if self.current_corr_matrix is not None:
            self.update_correlation_display(self.current_corr_matrix, self.current_method)
        else:
            self.show_initial_time_series()

    def apply_normalization(self):
        """Apply the tab's normalization and peak settings: recompute and
        record stage 5. The correlation (stage 6) isn't recomputed: that
        happens when the selection changes, or with a correlation button.
        If the normalized signals didn't change (e.g. only the peaks did),
        the correlation still holds and is recorded again so it stays up
        to date; otherwise it is shown as not updated, and stage 6 is out
        of date."""
        if self.signals_df is None:
            return
        self._read_controls()
        self._run_analysis(interactive=True)
        self._record_results(signals=True, corr=False)
        if self.current_corr_matrix is not None and not self.corr_is_stale:
            self._record_results(signals=False, corr=True)
        self._redraw_correlation()

    def _plot_signals(self, ax):
        show_points = bool(getattr(self, 'norm_show_points_var', None) and self.norm_show_points_var.get())
        points_used = getattr(self, 'lowest_points_used', {}) if show_points else {}
        points_labelled = False
        show_peaks = self.events_df is not None and bool(
            getattr(self, 'peak_show_var', None) is None or self.peak_show_var.get())
        peaks = events.events_by_column(self.events_df, self.norm_df.columns) if show_peaks else {}
        peaks_labelled = False
        for col, label in zip(self.norm_df.columns, self.cluster_labels):
            values = self.norm_df[col].to_numpy()
            line, = ax.plot(values, label=label, alpha=0.8)
            px = points_used.get(col)
            if px is not None and len(px):
                ax.scatter(px, values[px], s=28, color=line.get_color(), edgecolors='black', linewidths=0.6,
                           zorder=4, label=None if points_labelled else 'Lowest points used')
                points_labelled = True
            pk = peaks.get(col)
            if pk is not None and len(pk):
                ax.scatter(pk, values[pk], s=45, marker='^', color=line.get_color(), edgecolors='crimson',
                           linewidths=1.0, zorder=5, label=None if peaks_labelled else 'Peaks')
                peaks_labelled = True
        title = f'Time Series of Selected Cells ({self.norm_description})'
        if self.events_df is not None:
            title += f' - {len(self.events_df)} peaks ({self.analysis_result.settings["peak_method"]})'
        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Average Intensity' if self.norm_description == "raw" else 'Normalized value')
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
        """Signals of the selected cells, when there are too few for a correlation."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self._plot_signals(ax1)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.text(0.5, 0.5, 'Select at least 2 cells\nfor correlation analysis',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Correlation Matrix')
        self._show_corr_figure(fig)

    def calculate_correlation(self, method):
        """A correlation button: use `method` from now on (also when the
        selection changes), record stage 6 and redraw."""
        if self.norm_df is None or self.norm_df.shape[1] < 2:
            messagebox.showwarning("Warning", "You need at least 2 selected cells",
                                   parent=self.corr_window if self._corr_window_open() else self.window)
            return
        self.corr_method = method.lower()
        self._compute_correlation()
        self._record_results(signals=False, corr=True)
        self._redraw_correlation()

    def update_correlation_display(self, corr_matrix, method_name):
        """Signals, correlation matrix, histogram of correlations, dendrogram."""
        from scipy.cluster.hierarchy import dendrogram

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        self._plot_signals(ax1)
        ax1.legend()

        self._plot_corr_matrix(ax2, corr_matrix, f'Correlation Matrix ({method_name})')

        values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        ax3.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Distribution of Correlations')
        ax3.set_xlabel('Correlation Value')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        try:
            dendrogram(correlation.cluster_linkage(corr_matrix), labels=self.cluster_labels, ax=ax4)
            ax4.set_title(f'Cell Dendrogram ({correlation.LINKAGE_DESCRIPTION})')
            ax4.set_xlabel('Cells')
            ax4.set_ylabel('Distance')
        except ValueError as e:
            ax4.text(0.5, 0.5, f"Can't build the dendrogram:\n{e}", ha='center', va='center',
                     transform=ax4.transAxes, wrap=True)
            ax4.set_title('Dendrogram')

        self._show_corr_figure(fig)

    def export_time_series(self):
        """Save the selected cells' signals (shared NecLab signals format)."""
        if self.signals_df is None or self.signals_df.empty:
            messagebox.showwarning("Warning", "No data to export")
            return
        self._save("signals", self.signals_df, "Export Time Series", initialfile="cell_signals.csv")

    def export_normalized(self):
        """Save the normalized signals shown in the plot (stage 5 format)."""
        if self.signals_df is None or self.signals_df.empty:
            messagebox.showwarning("Warning", "No data to export")
            return
        self._save("normalized_signals", self.norm_df, "Export Normalized Signals",
                   initialfile="cell_signals_norm.csv")

    def export_correlation(self):
        """Save the current correlation matrix (stage 6 format)."""
        if getattr(self, 'current_corr_matrix', None) is None:
            messagebox.showwarning("Warning", "Select at least 2 cells to compute a correlation")
            return
        method = getattr(self, 'current_method', 'pearson').lower()
        self._save("correlation", self.current_corr_matrix, "Export Correlation",
                   initialfile=f"correlation_{method}.csv")

    def generate_report(self):
        """Plain-text report of the detection settings, cells and correlations."""
        if getattr(self, 'current_corr_matrix', None) is None:
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
            f.write(f"Signals: {self.norm_description}\n")
            if self.events_df is not None:
                settings = self.analysis_result.settings
                method, params = settings["peak_method"], core_analysis.peak_params(settings)
                f.write(f"Peaks: {method} ({', '.join(f'{k}={v}' for k, v in params.items())}), "
                        f"{len(self.events_df)} in total\n")
            f.write(f"Correlation method: {self.current_method}\n")
            f.write(f"Clustering: {correlation.LINKAGE_DESCRIPTION}\n\n")

            f.write("CELL DETAILS:\n")
            f.write("-" * 25 + "\n")
            counts = (self.events_df["cell_id"].value_counts() if self.events_df is not None
                      and len(self.events_df) else {})
            for cid in self._selected_sorted():
                line = f"Cell {cid}: {len(self.cell_pix[cid])} pixels"
                if self.events_df is not None:
                    line += f", {int(counts.get(cid, 0))} peaks"
                f.write(line + "\n")

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
        if not self._require_map("The 3D view"):
            return
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

        tab = add_tab(self.notebook, f"3D View - {self.method_name}", parent=self.window,
                      after=self.window if self.notebook is not None else None, geometry="800x600")
        self.extra_tabs = [t for t in self.extra_tabs if t.exists()] + [tab]
        window_3d = tab.body

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

        control_frame = tk.Frame(window_3d)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        canvas = FigureCanvasTkAgg(fig, master=window_3d)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tab.frame.bind("<Destroy>", lambda e: plt.close(fig) if e.widget is tab.frame else None, add="+")

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
        if self.current_fig is None:
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


def show_variability_analysis(img_array, method, main_window, project=None, on_recording_attached=None,
                              notebook=None):
    """Open the cell detection view for `img_array` (the raw stack) and
    `method` (a tab of `notebook`, if given)."""
    if img_array is None or len(img_array) == 0:
        messagebox.showwarning("Warning", "No image loaded")
        return None
    return VariabilityAnalysisWindow(img_array, method, main_window, project=project,
                                     on_recording_attached=on_recording_attached, notebook=notebook)


def open_detection_window(main_window, stack=None, var_im=None, map_name=None,
                          mask=None, labels=None, source="", project=None,
                          on_recording_attached=None, restore=None, notebook=None):
    """Open the cell detection view at stage 2, 3 or 4 from loaded data
    (a tab of `notebook`, if given). `stack` is the recording when it is
    open and the same frame size."""
    return VariabilityAnalysisWindow(stack, None, main_window, var_im=var_im, map_name=map_name,
                                     mask=mask, labels=labels, source=source, project=project,
                                     on_recording_attached=on_recording_attached, restore=restore,
                                     notebook=notebook)


def get_variability_methods():
    """Names of the variability methods, in menu order."""
    return variability.method_names()
