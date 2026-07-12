# User Manual — NecLab

Complete user manual for NecLab, the microscopy image analysis and data visualization tool developed at the LansBiodyt laboratory, Facultad de Ciencias, UNAM.

For installation instructions, see the [README](../README.md). This manual assumes the application is already installed and running (`python interface3.py`, or the precompiled executable).

> 🇲🇽 Versión en español: [MANUAL_USUARIO.md](MANUAL_USUARIO.md)

---

## Table of contents

1. [Application overview](#1-application-overview)
2. [Menu bar](#2-menu-bar)
3. [Tab: Procesamiento de Imágenes (Image Processing)](#3-tab-procesamiento-de-imágenes-image-processing)
4. [Variability analysis and cell detection](#4-variability-analysis-and-cell-detection)
5. [Tab: Visualización de Datos (Data Visualization)](#5-tab-visualización-de-datos-data-visualization)
6. [Tab: Datos Múltiples (Multiple Files)](#6-tab-datos-múltiples-multiple-files)
7. [Tab: Dendograma (Dendrogram)](#7-tab-dendograma-dendrogram)
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

Both workflows share the same main window, which has a menu bar at the top and a tabbed `Notebook` below it. Some tabs are present from startup (Procesamiento de Imágenes, Visualización de Datos); others are created automatically the first time the corresponding feature is used (Datos Múltiples, Dendograma, Time Series). Note that most UI labels inside the app remain in Spanish (the interface itself is not translated) — this manual describes each Spanish label in English so you can follow along regardless.

---

## 2. Menu bar

### Archivo (File)

| Item | Action |
|---|---|
| **Abrir OME-TIFF** (Open OME-TIFF, `Ctrl+O`) | Loads an OME-TIFF image stack, switches to the "Procesamiento de Imágenes" tab, and enables the Imagen and Análisis de Variabilidad menus. |
| **Abrir Datos (.npy / .csv)** (Open Data) | Loads a numeric data file into the "Visualización de Datos" tab. For CSV files, a dialog lets you pick the initial column (ROI); for `.npy` files, columns are auto-named "Column 1", "Column 2", etc. |
| **Cargar Matriz de Correlacion** (Load Correlation Matrix) | Loads an already-computed correlation matrix (CSV/XLSX, must be square) and displays it as a dendrogram + heatmap. See [section 9](#9-loading-a-precomputed-correlation-matrix). |
| **Abrir Multiples Archivos (.xls)** (Open Multiple Files) | Starts the "Datos Múltiples" workflow: pick several Excel files and the sheets to load from each. See [section 6](#6-tab-datos-múltiples-multiple-files). |
| **Salir** (Exit) | Closes the application (asks for confirmation and closes every open plot window). |

### Imagen (Image)

Disabled until an OME-TIFF image is loaded.

| Item | Action |
|---|---|
| **Auto Contraste** (Auto Contrast) | Applies automatic autocontrast to every frame of the stack. |
| **Histogram** | Shows a histogram of per-pixel variance across the whole stack, in a separate window. |
| **Binarize** | Opens a threshold dialog and binarizes the variance image. |
| **Restaurar Original** (Restore Original) | Reverts any adjustment and returns to the image exactly as loaded. |

### Análisis de Variabilidad (Variability Analysis)

Disabled until an image is loaded. Populated dynamically with the 7 variability methods (see [section 4](#4-variability-analysis-and-cell-detection)). Each entry opens a full "Análisis Completo — &lt;method&gt;" window for that method.

### Visualizacion

| Item | Action |
|---|---|
| **Dendograma** | Disabled until data is loaded via `Archivo → Abrir Datos`. Once loaded, opens/activates the permanent "Dendograma" tab (section 7). |
| **Series de tiempo** (Time Series) | Disabled until data is loaded via `Archivo → Abrir Datos`. Once loaded, opens the "Time Series" tab (section 8). |

> **Note:** these two items only become active once data has been loaded with `Archivo → Abrir Datos (.npy / .csv)`. If you only used "Abrir Multiples Archivos (.xls)" or "Cargar Matriz de Correlacion", they stay disabled — use those tabs' own workflows for dendrograms and correlations instead.

### Help

| Item | Action |
|---|---|
| **Check for Updates** | Compares the local version against the latest commit on `main` on GitHub and offers to update the application's `.py` files. See [section 11](#11-automatic-updates). |
| **About NecLab** | Shows a dialog with the application name, repository, and contact email (sergio.cruz@ciencias.unam.mx). |

---

## 3. Tab: Procesamiento de Imágenes (Image Processing)

Visible from startup. Made up of an image panel (left) and a controls panel (right):

**NAVEGACIÓN (Navigation)**
- **Capa (Frame)**: slider to move through the frames/slices of the loaded stack. Shows "Frame: n / N".

**AJUSTES DE IMAGEN (Image Adjustments)**
- **Brillo** (Brightness) and **Contraste** (Contrast): sliders from -100 to 100.
- **Auto Contraste**: applies autocontrast automatically.
- **Resetear Ajustes** (Reset Adjustments): returns brightness/contrast to their defaults.

**PROCESAMIENTO (Processing)**
- **Threshold**: "Aplicar" (Apply) checkbox to enable binarization, plus a 0–255 threshold slider.

**INFORMACIÓN (Information)**
- Read-only panel showing the dimensions, frame count, and data type of the loaded image.

### Typical workflow

1. `Archivo → Abrir OME-TIFF` (or `Ctrl+O`) and select the file.
2. Scroll through frames with the "Capa" control to inspect the stack.
3. Adjust brightness/contrast, or use "Auto Contraste", to aid visual inspection.
4. Optionally apply a threshold from the Imagen menu or the Processing panel.
5. Move to the **Análisis de Variabilidad** menu to start cell detection.

---

## 4. Variability analysis and cell detection

Choosing a method from the **Análisis de Variabilidad** menu opens a new 1400×800 window, "Análisis Completo — &lt;method&gt;", with its own menu bar.

### 4.1 Available variability methods

| # | Method (Spanish label) | Description | Default threshold |
|---|---|---|---|
| 1 | Rango (Range) | Max − min per pixel over time | 100 |
| 2 | Varianza Poblacional (Population Variance) | `np.var` with `ddof=0` | 120 |
| 3 | Varianza Muestral (Sample Variance) | `np.var` with `ddof=1` | 200 |
| 4 | Desviación Estándar Poblacional (Population Std. Dev.) | `ddof=0` | 12 |
| 5 | Desviación Estándar Muestral (Sample Std. Dev.) | `ddof=1` | 5 |
| 6 | Coeficiente de Variación (Coefficient of Variation) | `std(ddof=1) / mean × 100` | 5 |
| 7 | Rango Intercuartílico (IQR) | Q3 − Q1 | 20 |

### 4.2 Analysis window controls

- **Threshold** (1–1000): binarization threshold for the variability image; starts at the chosen method's default.
- **Min Size** / **Max Size** (1–500 / 1–1000): minimum and maximum cluster size (in pixels) to keep.

### 4.3 Clustering menu

| Item | Action |
|---|---|
| **Procesar Cluster (Básico)** (Process Cluster — Basic) | Detects spatial clusters using the threshold alone. |
| **Procesar Cluster (Avanzado)** (Process Cluster — Advanced) | Same as basic, but filters by Min Size / Max Size. |
| **Descomponer Clusters Grandes** (Decompose Large Clusters) | Splits oversized clusters into sub-clusters, respecting Min Size/Max Size. |

### 4.4 Selecting clusters

- Click clusters in the plot to select them (they highlight in red).
- **Selección por Región** (Region Selection): hold the right mouse button and drag over the plot to add/remove several clusters at once inside a rectangle; use the Añadir/Quitar (Add/Remove) toggle to pick the mode.
- **Selección** menu: `Seleccionar Todos` (Select All), `Limpiar Selección` (Clear Selection).
- The list of selected clusters appears in the side panel, each with a "Remover" (Remove) button.

### 4.5 3D visualization

**Visualización → Vista 3D** menu: opens a 3D surface (matplotlib) of the variability image, which can be rotated with the mouse.

### 4.6 Export

**Exportar** menu:
- **Guardar Imagen** (Save Image): exports the current analysis image.
- **Guardar .npy** (Save .npy): exports the mean time series of the selected clusters as a `(frames, 1 + n_clusters)` array, where the first column is the frame index.
- **Usar Seleccionados (Correlaciones)** (Use Selected — Correlations): opens the "Análisis de Correlaciones — Clusters Seleccionados" window, with buttons to compute Pearson, Kendall, or Spearman correlation across the chosen clusters' series.

---

## 5. Tab: Visualización de Datos (Data Visualization)

Visible from startup, but only fully active once data is loaded via `Archivo → Abrir Datos (.npy / .csv)`.

### 5.1 Tab's local menu

A dedicated menu bar sits at the top of the tab:

- **Vista** (View)
  - `Smoothing`: toggles convex-envelope smoothing on the signal.
  - `Show points`: shows/hides the signal's data points.
  - `Show Labels (Correlación)`: shows/hides axis labels on the correlation heatmap.
- **Guardar** (Save)
  - `Save Data Image...`: saves the data plot as an image.
  - `Save Correlation...`: saves the correlation heatmap as an image.
  - `Save Correlation Data...`: exports the correlation matrix (CSV/XLSX).
  - `Save Peaks CSV...`: exports the detected peaks (see 5.4).

### 5.2 Side panel

- **COLUMNAS DE DATOS** (Data Columns): multi-select list of every column in the loaded file.
- **PEAK FINDER**: combo box with the 7 peak-detection methods (see the [glossary](#12-method-glossary)) and a "Points:" spinbox (2–50) for the number of points used by the convex-envelope smoothing.
- **CORRELACIÓN**: combo box to choose the method (`pearson`, `kendall`, `spearman`).
- **SELECCIÓN** (Selection): single-select list, with "Add to Selection" / "Remove from Selection" buttons. At least 2 columns must be in the selection to draw the correlation heatmap.

### 5.3 Plot area

Clicking a column builds three stacked panels:

1. **Top**: the raw signal for the chosen column, with peak markers (if a detection method is active) and the smoothing baseline (if "Smoothing" is on).
2. **Middle**: the "processed" view — either the smoothed signal alone, or the diagnostic plot of the selected peak-detection method.
3. **Bottom**: the correlation heatmap (Pearson/Kendall/Spearman) for the columns in the Selección list.

### 5.4 Exporting detected peaks

`Guardar → Save Peaks CSV...` runs the active peak-detection method over every column in the Selección list and writes a CSV with a `TIME` column plus one 0/1 flag column per selected data column (1 = a peak was detected at that time index). Requires having already run a detection method once, and at least one column in the Selección list.

---

## 6. Tab: Datos Múltiples (Multiple Files)

Created the first time you use `Archivo → Abrir Multiples Archivos (.xls)`. Lets you load and compare several sheets from one or more Excel files at once.

### 6.1 Loading files

1. `Archivo → Abrir Multiples Archivos (.xls)`.
2. Select one or more Excel files in the dialog.
3. For each file, a "Seleccionar Hojas a Cargar" (Select Sheets to Load) dialog appears with a checkbox per sheet (plus "Seleccionar Todo"/"Deseleccionar Todo" — Select All/Deselect All).
4. Sheets are read **without a header row** (columns are identified only by position: Column 1, Column 2, ...). Only columns present in **every** loaded sheet are shown.
5. Loading runs in a background thread with a progress window; wait for it to close before interacting with the tab.

### 6.2 Tab's local menu

- **Vista** (View)
  - `Mostrar Nombres de Datos` (Show Data Names, default off): shows/hides per-sheet x-axis labels on both plots.
  - `Smoothing` (default **on**): applies convex-envelope smoothing to both the line plot and the heatmap.
  - `Puntos de Smoothing...` (Smoothing Points): opens a dialog with a spinbox (2–50, default 2) for the smoothing point count.
  - `Escala de Color Compartida (mapa de calor)` (Shared Color Scale — heatmap, default off): uses one shared color scale across every sheet instead of auto-scaling each sheet independently.
  - **Normalización** (Normalization, choose one):
    | Mode | Description |
    |---|---|
    | Por Columna (By Column) | divides by that column's own minimum, per sheet (default) |
    | Por Hoja (By Sheet) | divides by the minimum across the whole sheet |
    | Por Columna en Todas las Hojas (By Column Across All Sheets) | divides by that column's minimum pooled across every loaded sheet |
    | Global | divides by the minimum of the entire loaded dataset |
- **Gráfica** (Plot)
  - `Límites de Ejes (Gráfica Superior)...` (Axis Limits — Top Plot): manually sets X/Y limits for the line plot ("Auto" button to revert).
  - `Límites de Color (Heatmap)...` (Color Limits — Heatmap): manually sets the heatmap's color range, showing the range currently in use ("Auto" button to revert).
  - `Save Plot Image...`: saves the line plot (PNG/PDF/TIFF/SVG/EPS).
  - `Save Heatmap Image...`: saves the heatmap.
  - `Save Smoothed Data (XLSX, Multiple Sheets)...`: exports **every** common column from **every** loaded sheet, processed exactly as shown on screen (interpolated, normalized per the active mode, smoothed if enabled), into a single `.xlsx` file with one sheet, each source sheet's block separated by 20 blank rows. Runs in a background thread with a progress window.
- **Datos** (Data)
  - `Editar Clasificaciones...` (Edit Classifications): add, rename, or delete classification labels used to tag each loaded sheet; changes propagate to every column.
  - `Save Classifications...` / `Load Classifications...`: persist/restore, per data column, which classification label is assigned to each sheet (XLSX/CSV).

### 6.3 Tab layout

- Left panel: "DATOS (CELDAS)" (Data — Cells) list of the available common columns.
- Draggable horizontal split between the left panel and the plot area (enforced minimum width).
- Draggable vertical split between:
  - **Top panel**: combined line plot — the chosen column, plotted for every loaded sheet, offset horizontally with dashed separators between sheets; classification combo boxes aligned above each sheet's segment; a "▶" button to advance to the next column.
  - **Bottom panel**: one heatmap tile per sheet (columns = data columns, rows = samples).

Both panels redraw to fill their space on window resize or when a split is released.

---

## 7. Tab: Dendograma (Dendrogram)

Created the first time data is loaded (via `Archivo → Abrir Datos`) and `Visualizacion → Dendograma` is used.

- Side panel: "COLUMNAS DE DATOS" list (same as Visualización de Datos), "SELECCIÓN" list, "Add to Selection"/"Remove from Selection" buttons, and "Save Dendrogram Image" / "Save Dendrogram CSV" buttons.
- Plot area: top panel shows the raw signal of the chosen column; bottom panel shows the dendrogram of the columns in Selección (minimum 2), computed with `AgglomerativeClustering` (`distance_threshold=0, n_clusters=None`) on the transposed selection, drawn with `scipy.cluster.hierarchy.dendrogram`.
- "Save Dendrogram CSV" exports the cluster labels together with the linkage matrix (merge steps, children, distances).

---

## 8. Tab: Time Series

Created via `Visualizacion → Series de tiempo` (requires data loaded through `Archivo → Abrir Datos`).

- Its own data-column list and selection list.
- "Show Labels" checkbox.
- A single-signal preview (top) and a multi-signal overlay with a per-signal vertical offset (bottom), for visually comparing several columns at once.
- Buttons: "Save Image", "Save CSV", "Close Tab".

---

## 9. Loading a precomputed correlation matrix

`Archivo → Cargar Matriz de Correlacion` loads a correlation matrix computed externally (CSV or XLSX). It must be a square matrix; a warning is shown if any value falls outside the [-1, 1] range.

It is rendered directly inside the "Visualización de Datos" tab's main plot area as a dendrogram (top) plus a correlation heatmap (bottom), with its own buttons: "Save Dendrogram", "Save Correlation Matrix" (CSV), "Save Correlation Matrix Image", "Save All".

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

Smoothing (independent of the peak method) uses a convex envelope: it builds a piecewise-linear baseline from the signal's genuine local-convexity "lowest points" (the "Points" parameter, 2–50) and subtracts it from the raw signal. Used in both Visualización de Datos and Datos Múltiples.

### Variability methods

See the table in [section 4.1](#41-available-variability-methods).

### Correlation methods

`pearson` (linear), `kendall` (rank-based, tau), `spearman` (rank-based, rho) — available in Visualización de Datos, Datos Múltiples (implicitly via the dendrogram), and in the Análisis de Variabilidad cluster correlation window.

---

## 13. Troubleshooting

See the "Solución de problemas" section of the [README](../README.md#solución-de-problemas) for installation and startup errors.

**Usage-specific issues:**

- **"Save Peaks CSV..." is disabled**: run a peak-detection method on a column first, and add at least one column to the Selección list.
- **The correlation heatmap doesn't appear**: you need at least 2 columns in the Selección list.
- **"Cargar Matriz de Correlacion" rejects my file**: the matrix must be square (same number of rows and columns, with matching headers).
- **Some columns from my sheets don't show up in Datos Múltiples**: only columns present in **every** loaded sheet are shown (matched by position, since sheets are read without headers). If one sheet has fewer columns than the others, the "extra" columns from the other sheets won't be displayed.
- **Visualizacion → Dendograma / Series de tiempo are disabled**: they only activate after loading data via `Archivo → Abrir Datos (.npy / .csv)`; they are not enabled by "Abrir Multiples Archivos (.xls)" or "Cargar Matriz de Correlacion".

If the problem persists, contact sergio.cruz@ciencias.unam.mx or open an issue on the GitHub repository.
