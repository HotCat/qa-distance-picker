#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  Build script for qa-distance-picker
#
#  Usage:   bash build.sh
#  Output:  dist/qa-distance-picker/
#
#  Copy the entire dist/qa-distance-picker/ folder to any machine with the
#  same OS/architecture and run ./qa-distance-picker — no Python install needed.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ">>> Installing PyInstaller..."
pip install pyinstaller

echo ">>> Cleaning previous build..."
rm -rf build/ dist/

echo ">>> Building qa-distance-picker..."
pyinstaller qa-distance-picker.spec --noconfirm

echo ""
echo "========================================="
echo "  Build complete!"
echo "  Output: dist/qa-distance-picker/"
echo ""
echo "  Copy that folder to any machine and run:"
echo "    ./qa-distance-picker"
echo "========================================="
