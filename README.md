# NecLab

Software for automated microscopy image analysis and for the visualization/analysis of data series, developed at the LansBiodyt laboratory, Facultad de Ciencias, UNAM.

📖 **Complete user manual:** [docs/MANUAL_USUARIO.md](docs/MANUAL_USUARIO.md) (Español) · [docs/USER_MANUAL.md](docs/USER_MANUAL.md) (English)

---

## Description

NecLab is a desktop application (Tkinter) with two main workflows:

1. **Microscopy image processing**: loading OME-TIFF stacks, brightness/contrast/threshold adjustment, 7 temporal variability analysis methods, cell detection via spatial clustering, and per-cell time series extraction/export.
2. **Numerical data visualization and analysis**: loading `.npy`/`.csv` files or multiple Excel workbooks, peak detection (7 methods), smoothing, normalization, correlations (Pearson/Kendall/Spearman), hierarchical clustering dendrograms, and export to CSV/XLSX.

Main features:

- Load and view OME-TIFF images, with frame-by-frame navigation
- Adjust brightness, contrast, and binarization threshold
- Apply 7 temporal variability analysis methods (Range, Variances, Standard Deviations, Coefficient of Variation, IQR)
- Detect and segment cells via spatial clustering (basic, advanced, and decomposition of large clusters)
- Extract time series for individual cells and analyze their correlations, with a 3D view of the variability surface
- Load individual `.npy`/`.csv` data or **multiple Excel files at once** ("Multiple Files"), with smoothing, 4 normalization modes, a heatmap, and sheet classification
- Detect peaks with 7 different methods (Elliptic Envelope, Peak Caller, Local Outlier Factor, Isolation Forest, linear models, etc.)
- Generate hierarchical clustering dendrograms and correlation matrices, or load an already-computed correlation matrix
- Export results as CSV/XLSX (time series, peaks, correlations, classifications, clusters) and save an image of each plot
- Check for code updates directly from the **Help → Check for Updates** menu

---

## System requirements

- Operating system: Windows, macOS, or Linux
- Python 3.11 (recommended; the project is built and tested with 3.11)
- Miniconda or Anaconda (for the recommended installation)

---

## Installation

There are three ways to get NecLab. For most lab users, **Option A** (precompiled executable) is recommended; for development, or for platforms without a published executable, use **Option B** (conda).

### Option A: Use the precompiled executable (no Python installation needed)

Every time a tagged release (`v1.0`, `v1.1`, etc.) is published on GitHub, executables for macOS and Windows are automatically generated via GitHub Actions.

1. Go to the repository's **Releases** tab: https://github.com/SergioCruzUnamIA/NecLabUNAM/releases
2. Download `NecLab-mac.zip` (macOS) or `NecLab.exe` (Windows) from the latest release
3. **macOS**: unzip and drag `NecLab.app` to the Applications folder. Since the app isn't signed by Apple, the first time you must right-click → "Open" (instead of double-clicking) and confirm in the security dialog
4. **Windows**: run `NecLab.exe` directly. If Windows Defender SmartScreen shows a warning, select "More info" → "Run anyway"

If there is no recent tagged release, you can also download the latest automatic build from the repository's **Actions** tab (artifacts `NecLab-mac` / `NecLab-windows`), though these require a GitHub account to download.

### Option B: Installation with Conda (recommended for development)

#### Step 1: Install Miniconda

If you don't have Miniconda installed, download it from:
https://docs.conda.io/en/latest/miniconda.html

Follow the installation instructions for your operating system.

#### Step 2: Download the repository

Option A - Clone with Git:
```
git clone https://github.com/SergioCruzUnamIA/NecLabUNAM.git
```

Option B - Download ZIP:
1. Go to the repository page on GitHub
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the file to your desired location

#### Step 3: Create the virtual environment

Open a terminal and navigate to the project folder:
```
cd path/to/NecLabUNAM
```

Create the virtual environment with the dependencies:
```
conda env create -f environment.yml
```

This process will download all the required libraries (including `customtkinter`, installed via pip inside the conda environment itself, since it isn't available on conda-forge). It may take several minutes depending on your internet connection.

#### Step 4: Activate the virtual environment

```
conda activate neclab_env
```

Once activated, you'll see `(neclab_env)` at the start of the command line.

### Option C: Installation with pip / venv

`requirements.txt` is a complete snapshot (`pip freeze`) of the *development and packaging* environment, not just the runtime one: in addition to the libraries the application uses, it includes tools for building executables (`pyinstaller`, `cx_Freeze`, `dmgbuild`, `mac-alias`) and the Jupyter stack used by the prototyping notebooks. Installing it works, but it's heavier than necessary just to run the app.

```
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the app

With the virtual environment activated, run:

```
python interface3.py
```

The first run may take 1 to 2 minutes while the libraries load.

---

## Quick usage guide

This section is only a summary. For detailed step-by-step instructions for each menu, tab, and export feature, see the **complete user manual**: [docs/MANUAL_USUARIO.md](docs/MANUAL_USUARIO.md) (Español) or [docs/USER_MANUAL.md](docs/USER_MANUAL.md) (English).

The application is organized into tabs:

- **Image Processing**: open an OME-TIFF image (`File → Open OME-TIFF`, or `Ctrl+O`), adjust brightness/contrast/threshold, and from the **Variability Analysis** menu choose one of the 7 methods to open the cell detection and clustering window.
- **Data Visualization**: open a `.npy`/`.csv` file (`File → Open Data`) to find peaks, apply smoothing, compute correlations, and generate dendrograms or time series.
- **Multiple Files** (appears when using `File → Open Multiple Files (.xls)`): load and compare several Excel sheets at once, with smoothing, normalization, a heatmap, and sheet classification.
- **Dendrogram** and **Time Series**: created automatically when using the corresponding options in the **Visualization** menu, once data has been loaded.

You can also load an already-computed correlation matrix with `File → Load Correlation Matrix`.

---

## Updates from within the app

`Help → Check for Updates` compares the local version against the latest commit on the `main` branch of the GitHub repository and, if a newer version is available, offers to download the updated `.py` files and restart the application automatically. Requires an internet connection.

`Help → About NecLab` shows contact and repository information.

---

## Project structure

```
NecLabUNAM/
├── interface3.py               # Main graphical interface (entry point)
├── variability_functions.py    # Variability methods and cell clustering
├── peak_functions.py           # Peak detection (7 methods) and smoothing
├── corr_dendo_functions.py     # Correlations, dendrograms, and time series
├── multi_xls_helpers.py        # Loading and processing for the Multiple Files tab
├── visualization_helpers.py    # Helper functions for data visualization
├── image_loader.py             # OME-TIFF image loading
├── image_processing.py         # Image processing
├── NecLab.spec                 # PyInstaller spec (Mac/Windows executables)
├── build_mac.sh                # Script to build NecLab.app locally
├── build_windows.bat           # Script to build NecLab.exe locally
├── .github/workflows/build.yml # Automatic executable build (GitHub Actions)
├── environment.yml             # Project dependencies (conda)
├── requirements.txt            # Full development/packaging dependencies (pip)
├── cell_detection_complete.ipynb  # Prototyping notebook (not part of the app)
├── signal_processing.ipynb        # Prototyping notebook (not part of the app)
├── docs/
│   ├── MANUAL_USUARIO.md       # Complete user manual (Español)
│   └── USER_MANUAL.md          # Full user manual (English)
└── README.md                   # This file
```

---

## Building standalone executables

The executables are generated with PyInstaller from `NecLab.spec`, which produces a `.app` on macOS or a `.exe` on Windows depending on the platform it's run on.

**Locally:**
```
pip install pyinstaller
./build_mac.sh          # macOS → dist/NecLab.app
build_windows.bat       # Windows → dist\NecLab.exe
```

**Automatically (GitHub Actions):** the `.github/workflows/build.yml` workflow builds both executables when a `v*` tag is pushed (and publishes them to the corresponding Release), or when pushing to the development branches configured in that file. It can also be triggered manually from the Actions tab ("Run workflow").

---

## Troubleshooting

**The program doesn't start / `ModuleNotFoundError: No module named 'customtkinter'`:**
- Check that the virtual environment is activated (you should see `(neclab_env)` in the terminal)
- If you installed with conda before this fix, `customtkinter` isn't installed via `conda install` because it doesn't exist on conda-forge; run `pip install customtkinter` inside the activated environment, or recreate the environment with `conda env create -f environment.yml`

**Library not found error:**
- Run: `pip install library_name`
- Or reinstall the environment: `conda env remove -n neclab_env` and repeat the installation

**The image doesn't display correctly:**
- Check that the file is a valid OME-TIFF format
- Try adjusting the brightness and contrast in the controls panel

**"Check for Updates" doesn't find anything / fails:**
- Requires an internet connection and access to `api.github.com`; if your network blocks GitHub, update manually with `git pull` or by downloading a new ZIP/executable

**macOS says the app is damaged or from an unidentified developer:**
- The app isn't signed or notarized by Apple. Right-click → "Open" instead of double-clicking the first time

**Loading many Excel files in "Multiple Files" is slow:**
- Loading and saving run on a background thread with a progress window, but very large files or files with many sheets will still take time; wait for the progress window to close before interacting with the tab

---

## Test data

To test the software you can download a sample image from:
https://drive.google.com/file/d/1EP7TQMWQglbhgoRdJm2bY-10s2cYMDa3/view

---

## Credits

Developed at the LansBiodyt laboratory, Facultad de Ciencias, UNAM.

Advisor: Dr. Sergio Rodolfo Cruz Gómez

---

## License

This project is under the MIT license included in the [LICENSE](LICENSE) file.
