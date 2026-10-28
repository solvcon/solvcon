#!/bin/bash
#
# Prepare the pinned dependencies used to build the self-contained macOS Pilot
# bundle. The script creates isolated tools and relocatable SDK prefixes under
# build/bundle-deps for bundle-with-aqt.sh.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

QT_VERSION=6.11.1
PYTHON_VERSION=3.14.7
PYTHON_RELEASE=20260929
DEPS_ROOT="$BUNDLE_REPO_ROOT/build/bundle-deps"
TOOLS_PREFIX="$DEPS_ROOT/tools"
SDK_ROOT="$DEPS_ROOT/sdk"
# aqt always installs under <output>/<Qt version>/<architecture>.
SDK_PREFIX="$SDK_ROOT/$QT_VERSION/macos"
AQT_CONFIG=${AQT_CONFIG:-$BUNDLE_REPO_ROOT/thirdparty/aqt_settings.ini}

[[ "$(uname -s)" == "Darwin" ]] || {
    echo "macOS is required for macOS bundle dependencies" >&2
    exit 1
}

[[ "$(uname -m)" == "arm64" ]] || {
    echo "Apple Silicon macOS is required for macOS bundle dependencies" >&2
    exit 1
}

tmp=$(mktemp -d -t solvcon-bundle-deps)
trap 'rm -rf "$tmp"' EXIT

PYTHON_ARCHIVE="cpython-${PYTHON_VERSION}+${PYTHON_RELEASE}-aarch64-apple-darwin-install_only.tar.gz"
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PYTHON_RELEASE/$PYTHON_ARCHIVE"
curl --fail --location --retry 3 --output "$tmp/$PYTHON_ARCHIVE" \
    "$PYTHON_URL"
[[ "$(shasum -a 256 "$tmp/$PYTHON_ARCHIVE" | cut -d' ' -f1)" == \
    0211ccf04c5e0a0feb0b9bf1148030e1f6a31b8eae624f13919a7752a4e3dc79 ]] || {
    echo "Python archive SHA-256 mismatch" >&2
    exit 1
}

rm -rf "$TOOLS_PREFIX" "$SDK_ROOT"
mkdir -p "$TOOLS_PREFIX" "$SDK_PREFIX"
tar -xzf "$tmp/$PYTHON_ARCHIVE" --strip-components=1 -C "$TOOLS_PREFIX"
tar -xzf "$tmp/$PYTHON_ARCHIVE" --strip-components=1 -C "$SDK_PREFIX"

"$TOOLS_PREFIX/bin/python3" -I -m pip --isolated install \
    'pip-system-certs==5.3' 'aqtinstall==3.3.0' 'py7zr==1.1.0' \
    'cmake==4.4.3' 'pybind11==3.0.1'

# qtbase supplies Core, Gui, Widgets, plugins, and deployment tools. Svg is a
# separate base archive, while ShaderTools is an extra module.
AQT_CONFIG="$AQT_CONFIG" "$TOOLS_PREFIX/bin/python3" -I -m aqt install-qt \
    mac desktop "$QT_VERSION" clang_64 --outputdir "$SDK_ROOT" \
    --archives qtbase qtsvg --modules qtshadertools

for component in Core Gui Widgets Svg ShaderTools; do
    config="$SDK_PREFIX/lib/cmake/Qt6$component/Qt6${component}Config.cmake"
    [[ -f "$config" ]] || {
        echo "AQT installation is missing Qt6$component" >&2
        exit 1
    }
done

"$TOOLS_PREFIX/bin/python3" -I -m pip --isolated \
    --python "$SDK_PREFIX/bin/python3" install --only-binary=:all: \
    "PySide6-Essentials==$QT_VERSION" "shiboken6==$QT_VERSION" \
    'numpy==2.5.3' 'matplotlib==3.11.2' 'jsonschema==4.26.0' \
    zstandard lz4

PYSIDE_DIR=$("$SDK_PREFIX/bin/python3" -I -c \
    'import os, PySide6; print(os.path.dirname(PySide6.__file__))')
PYSIDE_QT_LIB="$PYSIDE_DIR/Qt/lib"
[[ -d "$PYSIDE_QT_LIB" ]] || {
    echo "PySide6 wheel is missing its Qt runtime" >&2
    exit 1
}

# Pilot links against AQT Qt, while PySide6 normally resolves Qt from its
# wheel. Redirect PySide6 before the build so both sides load the SDK's Qt
# instead of conflicting copies.
rm -rf "$PYSIDE_QT_LIB"
ln -s ../../../.. "$PYSIDE_QT_LIB"

"$SDK_PREFIX/bin/python3" -I - <<'PY'
import importlib.util
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets
import shiboken6

if importlib.util.find_spec("PySide6.QtMultimedia") is not None:
    raise SystemExit("unexpected PySide6.QtMultimedia binding")
PY

echo "Tools prefix: $TOOLS_PREFIX"
echo "SDK prefix: $SDK_PREFIX"

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
