# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for qa-distance-picker.

Build:  pyinstaller qa-distance-picker.spec --noconfirm
Output: dist/qa-distance-picker/
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Base directory (where this spec file lives)
BASE = os.path.abspath('.')

# Collect diplib's native shared libraries (PyInstaller often misses these)
diplib_binaries = []
diplib_datas = []
try:
    import diplib
    diplib_dir = os.path.dirname(diplib.__file__)
    # Include all .so files from the diplib package (skip javaio — needs JVM)
    for fname in os.listdir(diplib_dir):
        fpath = os.path.join(diplib_dir, fname)
        if fname.endswith('.so') and os.path.isfile(fpath) and 'javaio' not in fname:
            diplib_binaries.append((fpath, 'diplib'))
except ImportError:
    pass

a = Analysis(
    ['app.py'],
    pathex=[os.path.join(BASE, 'driver')],
    binaries=[
        ('/usr/lib/libMVSDK.so', '.'),
    ] + diplib_binaries,
    datas=[
        ('config.yaml', '.'),
    ] + diplib_datas,
    hiddenimports=[
        'mvsdk',
        'calibration',
        'detect_lines',
        'PySide6.QtWidgets',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'shiboken6',
        'diplib',
        'scipy',
        'scipy.signal',
        'scipy.ndimage',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'sam2', 'open3d', 'matplotlib',
        'tkinter', 'IPython', 'notebook', 'jupyterlab',
        'PIL.ImageQt',
    ],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='qa-distance-picker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='qa-distance-picker',
)
