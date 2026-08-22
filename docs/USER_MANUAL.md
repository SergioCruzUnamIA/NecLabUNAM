# User Manual — NecLab

Complete user manual for NecLab, the microscopy image analysis and data visualization tool developed at the LansBiodyt laboratory, Facultad de Ciencias, UNAM.

For installation instructions, see the [README](../README.md). This manual assumes the application is already installed and running (`python interface3.py`, or the precompiled executable).

> 🇲🇽 Versión en español: [MANUAL_USUARIO.md](MANUAL_USUARIO.md)

---

## Table of contents

1. [Application overview](#1-application-overview)
2. [Menu bar](#2-menu-bar)
3. [Tab: Image Processing](#3-tab-image-processing)
4. [Variability analysis and cell detection](#4-variability-analysis-and-cell-detection)
5. [Tab: Data Visualization](#5-tab-data-visualization)
6. [Tab: Multiple Files](#6-tab-multiple-files)
7. [Tab: Dendrogram](#7-tab-dendrogram)
8. [Tab: Time Series](#8-tab-time-series)
9. [Loading a precomputed correlation matrix](#9-loading-a-precomputed-correlation-matrix)
10. [Keyboard shortcuts](#10-keyboard-shortcuts)
11. [Automatic updates](#11-automatic-updates)
12. [Method glossary](#12-method-glossary)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Application overview

NecLab has two main workflows, organized as tabs inside a single window:

- **Image processing**: load an OME-TIFF stack, adjust its display, and analyze the temporal variability of each pixel to detect and segment cells as spatial clusters.
- **Data visualization and analysis**: load numeric series (a single `.npy`/`.csv` file, or several Excel files at once) to detect peaks, smooth signals, normalize, compute correlations, and build dendrograms.

Both workflows share the same main window, which has a menu bar at the top and a tabbed `Notebook` below it. Some tabs are present from startup (Image Processing, Data Visualization); others are created automatically the first time the corresponding feature is used (Multiple Files, Dendrogram, Time Series). The application's UI is entirely in English.

---

## 2. Menu bar

### File

| Item | Action |
|---|---|
| **Open OME-TIFF** (`Ctrl+O`) | Loads an OME-TIFF image stack, switches to the "Image Processing" tab, and enables the Image and Variability Analysis menus. |
| **Open Data (.npy / .csv / .xlsx)** | Loads a numeric data file into the "Data Visualization" tab. For CSV files, a dialog lets you pick the initial column (ROI); for `.npy` files, columns are auto-named "Column 1", "Column 2", etc. |
| **Load Correlation Matrix** | Loads an already-computed correlation matrix (CSV/XLSX, must be square) and displays it as a dendrogram + heatmap. See [section 9](#9-loading-a-precomputed-correlation-matrix). |
| **Open Multiple Files (.xls / .csv)** | Starts the "Multiple Files" workflow: pick several Excel files and the sheets to load from each. See [section 6](#6-tab-multiple-files). |
| **Exit** | Closes the application (asks for confirmation and closes every open plot window). |

### Image

Disabled until an OME-TIFF image is loaded.

| Item | Action |
|---|---|
| **Auto Contrast** | Applies automatic autocontrast to every frame of the stack. |
| **Histogram** | Shows a histogram of per-pixel variance across the whole stack, in a separate window. |
| **Binarize** | Opens a threshold dialog and binarizes the variance image. |
| **Restore Original** | Reverts any adjustment and returns to the image exactly as loaded. |

### Variability Analysis

Disabled until an image is loaded. Populated dynamically with the 7 variability methods (see [section 4](#4-variability-analysis-and-cell-detection)). Each entry opens a full "Full Analysis — &lt;method&gt;" window for that method.

### Visualization

| Item | Action |
|---|---|
| **Dendrogram** | Disabled until data is loaded via `File → Open Data`. Once loaded, opens/activates the permanent "Dendrogram" tab (section 7). |
| **Time Series** | Disabled until data is loaded via `File → Open Data`. Once loaded, opens the "Time Series" tab (section 8). |

> **Note:** these two items only become active once data has been loaded with `File → Open Data (.npy / .csv)`. If you only used "Open Multiple Files (.xls)" or "Load Correlation Matrix", they stay disabled — use those tabs' own workflows for dendrograms and correlations instead.

### Help

| Item | Action |
|---|---|
| **Check for Updates** | Compares the local version against the latest commit on `main` on GitHub and offers to update the application's `.py` files. See [section 11](#11-automatic-updates). |
| **About NecLab** | Shows a dialog with the application name, repository, and contact email (sergio.cruz@ciencias.unam.mx). |

---

## 3. Tab: Image Processing

Visible from startup. Made up of an image panel (left) and a controls panel (right):

**NAVIGATION**
- **Frame**: slider to move through the frames/slices of the loaded stack. Shows "Frame: n / N".

**IMAGE ADJUSTMENTS**
- **Brightness** and **Contrast**: sliders from -100 to 100.
- **Auto Contrast**: applies autocontrast automatically.
- **Reset Adjustments**: returns brightness/contrast to their defaults.

**PROCESSING**
- **Threshold**: "Apply" checkbox to enable binarization, plus a 0–255 threshold slider.

**INFORMATION**
- Read-only panel showing the dimensions, frame count, and data type of the loaded image.

### Typical workflow

1. `File → Open OME-TIFF` (or `Ctrl+O`) and select the file.
2. Scroll through frames with the "Frame" control to inspect the stack.
3. Adjust brightness/contrast, or use "Auto Contrast", to aid visual inspection.
4. Optionally apply a threshold from the Image menu or the Processing panel.
5. Move to the **Variability Analysis** menu to start cell detection.

---

## 4. Variability analysis and cell detection

Choosing a method from the **Variability Analysis** menu opens a new 1400×800 window, "Full Analysis — &lt;method&gt;", with its own menu bar.

### 4.1 Available variability methods

| # | Method | Description | Default threshold |
|---|---|---|---|
| 1 | Range | Max − min per pixel over time | 100 |
| 2 | Population Variance | `np.var` with `ddof=0` | 120 |
| 3 | Sample Variance | `np.var` with `ddof=1` | 200 |
| 4 | Population Standard Deviation | `ddof=0` | 12 |
| 5 | Sample Standard Deviation | `ddof=1` | 5 |
| 6 | Coefficient of Variation | `std(ddof=1) / mean × 100` | 5 |
| 7 | Interquartile Range (IQR) | Q3 − Q1 | 20 |

### 4.2 Analysis window controls

- **Threshold** (1–1000): binarization threshold for the variability image; starts at the chosen method's default.
- **Min Size** / **Max Size** (1–500 / 1–1000): minimum and maximum cluster size (in pixels) to keep.

### 4.3 Clustering menu

| Item | Action |
|---|---|
| **Process Cluster (Basic)** | Detects spatial clusters using the threshold alone. |
| **Process Cluster (Advanced)** | Same as basic, but filters by Min Size / Max Size. |
| **Decompose Large Clusters** | Splits oversized clusters into sub-clusters, respecting Min Size/Max Size. |

### 4.4 Selecting clusters

- Click clusters in the plot to select them (they highlight in red).
- **Region Selection**: hold the right mouse button and drag over the plot to add/remove several clusters at once inside a rectangle; use the Add/Remove toggle to pick the mode.
- **Selection** menu: `Select All`, `Clear Selection`.
- The list of selected clusters appears in the side panel, each with a "Remove" button.

### 4.5 3D visualization

**Visualization → 3D View** menu: opens a 3D surface (matplotlib) of the variability image, which can be rotated with the mouse.

### 4.6 Export

**Export** menu:
- **Save Image**: exports the current analysis image.
- **Save .npy**: exports the mean time series of the selected clusters as a `(frames, 1 + n_clusters)` array, where the first column is the frame index.
- **Use Selected (Correlations)**: opens the "Correlation Analysis — Selected Clusters" window, with buttons to compute Pearson, Kendall, or Spearman correlation across the chosen clusters' series.

---

## 5. Tab: Data Visualization

Visible from startup, but only fully active once data is loaded via `File → Open Data (.npy / .csv)`.

### 5.1 Tab's local menu

A dedicated menu bar sits at the top of the tab:

- **View**
  - `Smoothing`: toggles convex-envelope smoothing on the signal.
  - `Show points`: shows/hides the signal's data points.
  - `Show Labels (Correlation)`: shows/hides axis labels on the correlation heatmap.
- **Save**
  - `Save Data Image...`: saves the data plot as an image.
  - `Save Correlation...`: saves the correlation heatmap as an image.
  - `Save Correlation Data...`: exports the correlation matrix (CSV/XLSX).
  - `Save Peaks CSV...`: exports the detected peaks (see 5.4).

### 5.2 Side panel

- **DATA COLUMNS**: multi-select list of every column in the loaded file.
- **PEAK FINDER**: combo box with the 7 peak-detection methods (see the [glossary](#12-method-glossary)) and a "Points:" spinbox (2–50) for the number of points used by the convex-envelope smoothing.
- **CORRELATION**: combo box to choose the method (`pearson`, `kendall`, `spearman`).
- **SELECTION**: single-select list, with "Add to Selection" / "Remove from Selection" buttons. At least 2 columns must be in the selection to draw the correlation heatmap.

### 5.3 Plot area

Clicking a column builds three stacked panels:

1. **Top**: the raw signal for the chosen column, with peak markers (if a detection method is active) and the smoothing baseline (if "Smoothing" is on).
2. **Middle**: the "processed" view — either the smoothed signal alone, or the diagnostic plot of the selected peak-detection method.
3. **Bottom**: the correlation heatmap (Pearson/Kendall/Spearman) for the columns in the Selection list.

### 5.4 Exporting detected peaks

`Save → Save Peaks CSV...` runs the active peak-detection method over every column in the Selection list and writes a CSV with a `TIME` column plus one 0/1 flag column per selected data column (1 = a peak was detected at that time index). Requires having already run a detection method once, and at least one column in the Selection list.

---

## 6. Tab: Multiple Files

Created the first time you use `File → Open Multiple Files (.xls)`. Lets you load and compare several sheets from one or more Excel files at once.

### 6.1 Loading files

1. `File → Open Multiple Files (.xls)`.
2. Select one or more Excel files in the dialog.
3. For each file, a "Select Sheets to Load" dialog appears with a checkbox per sheet (plus "Select All"/"Deselect All").
4. Sheets are read **without a header row** (columns are identified only by position: Column 1, Column 2, ...). Only columns present in **every** loaded sheet are shown.
5. Loading runs in a background thread with a progress window; wait for it to close before interacting with the tab.

### 6.2 Tab's local menu

- **View**
  - `Show Data Names` (default off): shows/hides per-sheet x-axis labels on both plots.
  - `Smoothing` (default **on**): applies convex-envelope smoothing to both the line plot and the heatmap.
  - `Smoothing Points...`: opens a dialog with a spinbox (2–50, default 2) for the smoothing point count.
  - `Shared Color Scale (heatmap)` (default off): uses one shared color scale across every sheet instead of auto-scaling each sheet independently.
  - **Normalization** (choose one):
    | Mode | Description |
    |---|---|
    | By Column (minimum of that column in each sheet) | divides by that column's own minimum, per sheet (default) |
    | By Sheet (minimum of the entire sheet) | divides by the minimum across the whole sheet |
    | By Column Across All Sheets (shared minimum) | divides by that column's minimum pooled across every loaded sheet |
    | Global (minimum of all columns and sheets) | divides by the minimum of the entire loaded dataset |
- **Plot**
  - `Axis Limits (Top Plot)...`: manually sets X/Y limits for the line plot ("Auto" button to revert).
  - `Color Limits (Heatmap)...`: manually sets the heatmap's color range, showing the range currently in use ("Auto" button to revert).
  - `Save Plot Image...`: saves the line plot (PNG/PDF/TIFF/SVG/EPS).
  - `Save Heatmap Image...`: saves the heatmap.
  - `Save Smoothed Data (XLSX/CSV)...`: exports **every** common column from **every** loaded sheet, processed exactly as shown on screen (interpolated, normalized per the active mode, smoothed if enabled), into a single file, each source sheet's block separated by 20 blank rows. Runs in a background thread with a progress window.
- **Data**
  - `Edit Classifications...`: add, rename, or delete classification labels used to tag each loaded sheet; changes propagate to every column.
  - `Save Classifications...` / `Load Classifications...`: persist/restore, per data column, which classification label is assigned to each sheet (XLSX/CSV).

### 6.3 Tab layout

- Left panel: "DATA (COLUMNS)" list of the available common columns.
- Draggable horizontal split between the left panel and the plot area (enforced minimum width).
- Draggable vertical split between:
  - **Top panel**: combined line plot — the chosen column, plotted for every loaded sheet, offset horizontally with dashed separators between sheets; classification combo boxes aligned above each sheet's segment; a "▶" button to advance to the next column.
  - **Bottom panel**: one heatmap tile per sheet (columns = data columns, rows = samples).

Both panels redraw to fill their space on window resize or when a split is released.

---

## 7. Tab: Dendrogram

Created the first time data is loaded (via `File → Open Data`) and `Visualization → Dendrogram` is used.

- Side panel: "DATA COLUMNS" list (same as Data Visualization), "SELECTION" list, "Add to Selection"/"Remove from Selection" buttons, and "Save Dendrogram Image" / "Save Dendrogram CSV" buttons.
- Plot area: top panel shows the raw signal of the chosen column; bottom panel shows the dendrogram of the columns in the Selection list (minimum 2), computed with `AgglomerativeClustering` (`distance_threshold=0, n_clusters=None`) on the transposed selection, drawn with `scipy.cluster.hierarchy.dendrogram`.
- "Save Dendrogram CSV" exports the cluster labels together with the linkage matrix (merge steps, children, distances).

---

## 8. Tab: Time Series

Created via `Visualization → Time Series` (requires data loaded through `File → Open Data`).

- Its own data-column list and selection list.
- "Show Labels" checkbox.
- A single-signal preview (top) and a multi-signal overlay with a per-signal vertical offset (bottom), for visually comparing several columns at once.
- Buttons: "Save Image", "Save CSV", "Close Tab".

---

## 9. Loading a precomputed correlation matrix

`File → Load Correlation Matrix` loads a correlation matrix computed externally (CSV or XLSX). It must be a square matrix; a warning is shown if any value falls outside the [-1, 1] range.

It is rendered directly inside the "Data Visualization" tab's main plot area as a dendrogram (top) plus a correlation heatmap (bottom), with its own buttons: "Save Dendrogram", "Save Correlation Matrix" (CSV), "Save Correlation Matrix Image", "Save All".

---

## 10. Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Open OME-TIFF |

There is no drag-and-drop support; all file loading goes through the operating system's file dialogs.

---

## 11. Automatic updates

`Help → Check for Updates`:

1. Queries, in a background thread, the latest commit on the `main` branch of the `sergiocruzunamia/neclabunam` GitHub repository, and compares it against the locally recorded version.
2. If a newer version is found, offers to download the current versions of the 8 core `.py` files (`interface3.py`, `peak_functions.py`, `visualization_helpers.py`, `corr_dendo_functions.py`, `variability_functions.py`, `image_loader.py`, `image_processing.py`, `multi_xls_helpers.py`) and overwrite them locally.
3. Offers to restart the application automatically afterward.

Requires an internet connection. This feature does **not** update Python dependencies or the packaged executable itself — only the `.py` source files — so it only applies to installations run from source (installation option B/C), not to precompiled executables.

---

## 12. Method glossary

### Peak detection (Peak Finder)

| Method | Technique | Parameters (default) |
|---|---|---|
| Elliptic Envelope | `sklearn.covariance.EllipticEnvelope` with `ElasticNet` detrending | Contamination (0.01) |
| Peak Caller | Custom rise/fall percentage heuristic | Rise % (5), Fall % (5), Max Lookback (10 pts), Max Lookahead (10 pts) |
| Local Outlier Factor | `sklearn.neighbors.LocalOutlierFactor` | N Neighbors (20) |
| Peak Function 4 | Elliptic Envelope + SVR | Contamination (0.01) |
| Isolation Forest | `sklearn.ensemble.IsolationForest` | Contamination (0.05) |
| Linear Model | `SGDOneClassSVM` | Nu (0.131) |
| Peak Function 7 | Lasso + Local Outlier Factor | N Neighbors (20) |

Smoothing (independent of the peak method) uses a convex envelope: it builds a piecewise-linear baseline from the signal's genuine local-convexity "lowest points" (the "Points" parameter, 2–50) and subtracts it from the raw signal. Used in both Data Visualization and Multiple Files.

### Variability methods

See the table in [section 4.1](#41-available-variability-methods).

### Correlation methods

`pearson` (linear), `kendall` (rank-based, tau), `spearman` (rank-based, rho) — available in Data Visualization, Multiple Files (implicitly via the dendrogram), and in the Variability Analysis cluster correlation window.

---

## 13. Troubleshooting

See the "Troubleshooting" section of the [README](../README.md#troubleshooting) for installation and startup errors.

**Usage-specific issues:**

- **"Save Peaks CSV..." is disabled**: run a peak-detection method on a column first, and add at least one column to the Selection list.
- **The correlation heatmap doesn't appear**: you need at least 2 columns in the Selection list.
- **"Load Correlation Matrix" rejects my file**: the matrix must be square (same number of rows and columns, with matching headers).
- **Some columns from my sheets don't show up in Multiple Files**: only columns present in **every** loaded sheet are shown (matched by position, since sheets are read without headers). If one sheet has fewer columns than the others, the "extra" columns from the other sheets won't be displayed.
- **Visualization → Dendrogram / Time Series are disabled**: they only activate after loading data via `File → Open Data (.npy / .csv)`; they are not enabled by "Open Multiple Files (.xls)" or "Load Correlation Matrix".

If the problem persists, contact sergio.cruz@ciencias.unam.mx or open an issue on the GitHub repository.
