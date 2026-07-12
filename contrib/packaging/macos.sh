#!/bin/bash

# make sure starting from project's root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.."

python3 -m pip install pyinstaller

make clean
rm -rf build/portable/
mkdir -p build/portable/
cp -R ./modmesh build/portable/
cp -R ./resources build/portable/
cd build/portable/

cat > modmesh.py <<'EOF'
import os
os.environ["QT3D_RENDERER"] = "opengl"
from modmesh.pilot import launch
if __name__ == "__main__":
    launch()
EOF

pyinstaller --onedir \
            --windowed \
            --add-data "modmesh:modmesh" \
            --hidden-import PySide6.Qt3DCore \
            --hidden-import PySide6.Qt3DRender \
            --hidden-import PySide6.Qt3DExtras \
            --hidden-import PySide6.Qt3DInput \
            --exclude-module PySide6.QtNetwork \
            --icon="resources/pilot/solvcon.icns" \
            modmesh.py