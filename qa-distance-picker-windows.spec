# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for qa-distance-picker (Windows build).

Build:  pyinstaller qa-distance-picker-windows.spec --noconfirm
Output: dist/qa-distance-picker/
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

BASE = os.path.abspath('.')

# Collect diplib native libraries (.pyd / .dll on Windows)
diplib_binaries = []
try:
    import diplib
    diplib_dir = os.path.dirname(diplib.__file__)
    for fname in os.listdir(diplib_dir):
        fpath = os.path.join(diplib_dir, fname)
        if os.path.isfile(fpath) and 'javaio' not in fname:
            if fname.endswith(('.pyd', '.dll')):
                diplib_binaries.append((fpath, 'diplib'))
except ImportError:
    pass

# MindVision SDK DLL — place MVSDK.dll next to this spec or in driver/
mvsdk_dll = os.path.join(BASE, 'driver', 'MVSDK.dll')
sdk_binaries = []
if os.path.exists(mvsdk_dll):
    sdk_binaries.append((mvsdk_dll, '.'))

a = Analysis(
    ['app.py'],
    pathex=[os.path.join(BASE, 'driver')],
    binaries=diplib_binaries + sdk_binaries,
    datas=[
        ('config.yaml', '.'),
    ],
    hiddenimports=[
        'mvsdk',
        'calibration',
        'detect_lines',
        'alignment',
        'debug_overlay',
        'PySide6.QtWidgets',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'shiboken6',
        'diplib',
        'scipy',
        'scipy.signal',
        'scipy.ndimage',
        'scipy.optimize',
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
