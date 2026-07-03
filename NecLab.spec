# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for NecLab
# Run with: pyinstaller NecLab.spec
#
# Must be executed on the TARGET platform:
#   Mac  → pyinstaller NecLab.spec   (produces dist/NecLab.app)
#   Win  → pyinstaller NecLab.spec   (produces dist/NecLab.exe)

import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# ── Hidden imports that PyInstaller commonly misses ──────────────────────────
hidden = []

# sklearn pulls in many lazy-loaded submodules
hidden += collect_submodules('sklearn')
hidden += collect_submodules('scipy')
hidden += collect_submodules('scipy.special')
hidden += collect_submodules('scipy.linalg')

# matplotlib backends
hidden += [
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg',
]

# image/tiff stacks
hidden += collect_submodules('tifffile')
hidden += collect_submodules('imagecodecs')
hidden += collect_submodules('pyometiff')

# misc
hidden += [
    'PIL._tkinter_finder',
    'pkg_resources.py2_compat',
    'pandas._libs.tslibs.np_datetime',
    'pandas._libs.tslibs.nattype',
    'pandas._libs.tslibs.timestamps',
    'openpyxl',
    'xlrd',
]

# ── Data files bundled into the executable ───────────────────────────────────
datas = []
datas += collect_data_files('sklearn')
datas += collect_data_files('scipy')
datas += collect_data_files('matplotlib')
datas += collect_data_files('pyometiff')

# ── Analysis ─────────────────────────────────────────────────────────────────
a = Analysis(
    ['interface3.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['IPython', 'jupyter', 'notebook', 'ipykernel', 'debugpy'],
    noarchive=False,
)

pyz = PYZ(a.pure)

# ── Platform-specific output ─────────────────────────────────────────────────
if sys.platform == 'darwin':
    # Mac: produce a .app bundle
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='NecLab',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        # icon='NecLab.icns',  # uncomment and add icon file to use one
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='NecLab',
    )
    app = BUNDLE(
        coll,
        name='NecLab.app',
        # icon='NecLab.icns',
        bundle_identifier='mx.unam.neclab',
        info_plist={
            'NSHighResolutionCapable': True,
            'CFBundleShortVersionString': '1.0.0',
        },
    )

else:
    # Windows: produce a single .exe
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='NecLab',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,          # no black terminal window
        disable_windowed_traceback=False,
        # icon='NecLab.ico',    # uncomment and add icon file to use one
    )
