#!/usr/bin/env bash
# Build NecLab.app for macOS
# Requirements: pip install pyinstaller  (Python 3.9+, on a Mac)
set -e

echo "==> Cleaning previous builds..."
rm -rf build dist

echo "==> Running PyInstaller..."
pyinstaller NecLab.spec

echo ""
echo "==> Done! App bundle is at: dist/NecLab.app"
echo "    You can drag it to /Applications or double-click to run."
