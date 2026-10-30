#!/bin/bash
#
# This script is a prototype to create a self-contained macOS .app bundle for
# solvcon, package it into a DMG file, and verify both are hermetic. It employs
# macdeployqt, otool, and install_name_tool to process the binaires, and ad-hoc
# codesign to reseal them.
#
# !! WARNING !!
# The script assumes dependencies were installed by prepare-with-aqt.sh.
#
# Usage:
#   ./contrib/bundle/bundle-with-aqt.sh check
#   ./contrib/bundle/bundle-with-aqt.sh bundle [--skip-build]
#                                                   [--skip-check]
#                                                   [--output DIR]
#   ./contrib/bundle/bundle-with-aqt.sh verify path/to/pilot.dmg
#
# Legacy usage without a subcommand still runs the bundle step:
#   ./contrib/bundle/bundle-with-aqt.sh [--skip-build] [--skip-check]
#                                            [--output DIR]
#
#   --skip-build   Skip `make pilot` and use the existing build output.
#   --skip-check   Skip the bundle/DMG verification phase.
#   --output DIR   Write pilot.dmg into DIR (default: build/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$BUNDLE_REPO_ROOT"

MIN_LOADS=${MIN_LOADS:-50}
SDK_ROOT="$BUNDLE_REPO_ROOT/build/bundle-deps/sdk"
PREFIX=""
TOOLS_PREFIX="$BUNDLE_REPO_ROOT/build/bundle-deps/tools"
HOST_PREFIX_RE="^(${SDK_ROOT//./\.}|${TOOLS_PREFIX//./\.})"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  contrib/bundle/bundle-with-aqt.sh check
  contrib/bundle/bundle-with-aqt.sh bundle [--skip-build] [--skip-check] [--output DIR]
  contrib/bundle/bundle-with-aqt.sh verify path/to/pilot.dmg
  contrib/bundle/bundle-with-aqt.sh all [--skip-build] [--skip-check] [--output DIR]

Subcommands:
  check    Check macOS bundle release dependencies. Does not build or install.
  bundle   Check dependencies, then build/package with aqt dependencies.
  verify   Verify a generated release DMG artifact.
  all      Run check, bundle, and verify.
EOF
}

note() { printf '==> %s\n' "$*"; }
ok() { printf 'OK    %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*" >&2; return 1; }

require_macos() {
    [[ "$(uname -s)" == "Darwin" ]] || fail "macOS is required for macOS bundle release"
    [[ "$(uname -m)" == "arm64" ]] || fail "Apple Silicon is required for macOS bundle release"
}

setup_hint() {
    cat >&2 <<'EOF'

Prepare the bundle dependencies, then retry:
  ./contrib/bundle/prepare-with-aqt.sh
EOF
}

initialize_sdk_prefix() {
    local candidates
    candidates=("$SDK_ROOT"/*/macos)
    if [[ ${#candidates[@]} -ne 1 || ! -d "${candidates[0]}" ]]; then
        fail "expected exactly one prepared SDK under $SDK_ROOT"
        setup_hint
        return 1
    fi
    PREFIX="${candidates[0]}"
    export PATH="$PREFIX/bin:$TOOLS_PREFIX/bin:$PATH"
}

require_command() {
    local cmd="$1" hint="${2:-}"
    if command -v "$cmd" >/dev/null 2>&1; then
        ok "$cmd: $(command -v "$cmd")"
    else
        fail "$cmd: not found${hint:+ ($hint)}"
    fi
}

python_module_check() {
    local module="$1" hint="${2:-}" out
    out=$(mktemp -t solvcon-bundle-pycheck)
    if python3 - "$module" <<'PY' >"$out" 2>&1
import importlib
import sys
module = sys.argv[1]
try:
    mod = importlib.import_module(module)
except Exception as e:
    print(f"{type(e).__name__}: {e}")
    raise SystemExit(1)
else:
    print(getattr(mod, "__file__", "built-in"))
PY
    then
        ok "python module $module: $(cat "$out")"
        rm -f "$out"
    else
        cat "$out" >&2 || true
        rm -f "$out"
        fail "python module $module: missing${hint:+ ($hint)}"
    fi
}

python_dylib_check() {
    local path="$PREFIX/lib/libpython$(bundle_pyver).dylib"
    [[ -f "$path" ]] || return 1
    printf '%s\n' "$path"
}

check_pybind11_cmake() {
    local config_dir
    config_dir=$("$TOOLS_PREFIX/bin/python3" -m pybind11 --cmakedir 2>/dev/null || true)
    if [[ -n "$config_dir" && -f "$config_dir/pybind11Config.cmake" ]]; then
        ok "pybind11 CMake config: $config_dir/pybind11Config.cmake"
    else
        fail "pybind11 CMake config: not found under $TOOLS_PREFIX"
    fi
}

check_deps() {
    local dylib_path failed=0
    require_macos
    initialize_sdk_prefix

    ok "SDK prefix: $PREFIX"
    ok "tools prefix: $TOOLS_PREFIX"

    for cmd in cmake make python3 macdeployqt qtpaths otool lipo stat \
        install_name_tool codesign ditto hdiutil readlink rsync shasum; do
        require_command "$cmd" || failed=1
    done

    if dylib_path=$(python_dylib_check 2>&1); then
        ok "Python dylib: $dylib_path"
    else
        fail "Python dylib not found under $PREFIX/lib" || failed=1
    fi

    check_pybind11_cmake || failed=1
    python_module_check numpy || failed=1
    python_module_check PySide6.QtCore || failed=1
    python_module_check PySide6.QtGui || failed=1
    python_module_check PySide6.QtWidgets || failed=1
    python_module_check PySide6.QtSvg || failed=1
    python_module_check shiboken6 || failed=1
    python_module_check matplotlib || failed=1

    if [[ $failed -ne 0 ]]; then
        setup_hint
        exit 1
    fi
}

# SDK Python's "X.Y" version string.
bundle_pyver() {
    python3 -c "import sys; print('%d.%d' % sys.version_info[:2])"
}

# Print every Mach-O under $1, NUL-delimited: standalone dylibs, .so
# extensions, MacOS/ binaries, and the main binary of every nested framework.
# Framework binaries cannot be matched by file mode, so synthesise the
# canonical path (Versions/<v>/<name>) explicitly.
list_macho() {
    find "$1" -name '*.framework' -type d -print0 | \
        while IFS= read -r -d '' fw; do
            local nm ver bin
            nm=$(basename "$fw" .framework)
            for ver in "$fw"/Versions/*; do
                [[ -d "$ver" && ! -L "$ver" ]] || continue
                bin="$ver/$nm"
                [[ -f "$bin" ]] && printf '%s\0' "$bin"
            done
        done
    find "$1" -type f \( -name '*.dylib' -o -name '*.so' -o -path '*/MacOS/*' \) -print0
}

# Print every LC_RPATH entry of the Mach-O at $1, one per line.  `otool -L`
# does not list these, so we parse `otool -l`.  The entry `otool -l` output
# looks like:
# ```
# Load command 33
#           cmd LC_RPATH
#       cmdsize 96
#          path /path/to/PySide6 (offset 12)
# ```
# The parsed result is:
# ```
# /path/to/PySide6
# ```
binary_rpaths() {
    otool -l "$1" 2>/dev/null | awk '
        /^[[:space:]]*cmd / { c = 0 }
        /cmd LC_RPATH/      { c = 1; next }
        c && $1 == "path"   { print $2; c = 0 }
    '
}

# Verify the .app under $1 contains no surviving dependency prefix reference
# with both LC_LOAD_DYLIB-style commands (visible via otool -L) and LC_RPATH
# search paths. Print offenders. Return 0 if clean and 1 otherwise.
check_self_contained() {
    local total=0 bad=0 f load_refs rpath_refs
    while IFS= read -r -d '' f; do
        total=$((total+1))
        load_refs=$(otool -L "$f" 2>/dev/null | tail -n +2 | awk '{print $1}' \
            | grep -E "$HOST_PREFIX_RE" || true)
        rpath_refs=$(binary_rpaths "$f" | grep -E "$HOST_PREFIX_RE" || true)
        if [[ -n "$load_refs" || -n "$rpath_refs" ]]; then
            bad=$((bad+1))
            echo "    BAD: $f"
            [[ -n "$load_refs" ]] && echo "$load_refs" | sed 's/^/        load:  /'
            [[ -n "$rpath_refs" ]] && echo "$rpath_refs" | sed 's/^/        rpath: /'
        fi
    done < <(list_macho "$1")
    if [[ $bad -eq 0 ]]; then
        ok "$total Mach-O files, none reference the dependency prefixes"
        return 0
    fi
    fail "$bad of $total Mach-O files still reference dependency prefixes"
    return 1
}

verify_app_structure() {
    local app="$1" bin
    [[ -d "$app" ]] || fail "app not found: $app"
    [[ -f "$app/Contents/Info.plist" ]] || fail "Info.plist not found"
    bin="$app/Contents/MacOS/$(basename "$app" .app)"
    [[ -x "$bin" ]] || fail "main binary not executable: $bin"
    [[ -d "$app/Contents/Frameworks" ]] || fail "Contents/Frameworks not found"
    [[ -d "$app/Contents/Resources" ]] || fail "Contents/Resources not found"
    file "$bin"
}

runtime_import_check() {
    local bin="$1"
    (
        cd "$VERIFY_TMP"
        env -i HOME="$VERIFY_TMP/home" USER="${USER:-nobody}" \
            PATH=/usr/bin:/bin TERM="${TERM:-xterm}" PYTHONNOUSERSITE=1 \
            "$bin" --mode=python -c \
            "from PySide6 import QtCore, QtGui, QtWidgets, QtSvg; import matplotlib.pyplot; import solvcon.pilot.base._base_app; import solvcon.pilot.painter._icons; import solvcon.pilot.visual._movie as movie; assert not movie.HAS_VIDEO"
    )
}

smoke_launch_verify() {
    local bin="$1" trace="$2" pid elapsed=0 loaded
    : > "$trace"
    (
        cd "$VERIFY_TMP"
        exec env -i HOME="$VERIFY_TMP/home" USER="${USER:-nobody}" \
            PATH=/usr/bin:/bin TERM="${TERM:-xterm}" PYTHONNOUSERSITE=1 \
            DYLD_PRINT_LIBRARIES=1 "$bin"
    ) >/dev/null 2>"$trace" &
    pid=$!
    while [[ $elapsed -lt 15 ]]; do
        sleep 1
        elapsed=$((elapsed + 1))
        loaded=$(grep -c '^dyld\[' "$trace" 2>/dev/null || true)
        [[ ${loaded:-0} -ge $MIN_LOADS ]] && break
    done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true

    loaded=$(grep -c '^dyld\[' "$trace" 2>/dev/null || true)
    if [[ $loaded -lt $MIN_LOADS ]]; then
        fail "pilot loaded only $loaded libraries; did it crash early?" || {
            sed -n '1,80p' "$trace" >&2
            return 1
        }
    fi
    if grep -E '^dyld\[' "$trace" | grep -E "${HOST_PREFIX_RE#^}" >/dev/null; then
        fail "smoke launch loaded libraries from host package prefixes:" || {
            grep -E '^dyld\[' "$trace" | grep -E "${HOST_PREFIX_RE#^}" | head -20 >&2
            return 1
        }
    fi
    ok "$loaded libraries loaded, none from host package prefixes"
}

VERIFY_MNT=""
VERIFY_TMP=""
VERIFY_TRACE=""
VERIFY_MARKER=""
cleanup_verify() {
    [[ -n "${VERIFY_MNT:-}" ]] && hdiutil detach -quiet "$VERIFY_MNT" >/dev/null 2>&1 || true
    [[ -n "${VERIFY_TMP:-}" ]] && rm -rf "$VERIFY_TMP"
    [[ -n "${VERIFY_TRACE:-}" ]] && rm -f "$VERIFY_TRACE"
    [[ -n "${VERIFY_MARKER:-}" ]] && rm -f "$VERIFY_MARKER"
}

verify_dmg() {
    local dmg="$1" mnt tmp trace marker app bin copied_app
    require_macos
    [[ -f "$dmg" ]] || fail "release artifact not found: $dmg"

    note "Release artifact"
    ls -lh "$dmg"
    shasum -a 256 "$dmg"
    hdiutil imageinfo "$dmg" >/dev/null

    VERIFY_MNT=$(mktemp -d -t solvcon-release-dmg)
    VERIFY_TMP=$(mktemp -d -t solvcon-release-app)
    VERIFY_TRACE=$(mktemp -t solvcon-release-dyld)
    VERIFY_MARKER=$(mktemp -t solvcon-release-marker)
    mnt="$VERIFY_MNT"
    tmp="$VERIFY_TMP"
    trace="$VERIFY_TRACE"
    marker="$VERIFY_MARKER"
    mkdir -p "$tmp/home"

    note "Mounting DMG"
    hdiutil attach -nobrowse -readonly -mountpoint "$mnt" "$dmg" >/dev/null
    app=$(find "$mnt" -maxdepth 2 -name '*.app' -type d -print -quit)
    [[ -n "$app" ]] || fail "no .app found in $dmg"
    bin="$app/Contents/MacOS/$(basename "$app" .app)"

    note "Checking mounted app structure"
    verify_app_structure "$app"

    note "Checking code signature before launch"
    codesign --verify --deep --strict "$app"
    spctl --assess --type execute --verbose=4 "$app" 2>&1 || \
        echo "    warning: spctl rejected this app (expected for ad-hoc signed prototype builds)"

    note "Static scan for host package paths"
    check_self_contained "$app"

    note "Runtime import check from a writable app copy"
    cp -R "$app" "$tmp/"
    copied_app="$tmp/$(basename "$app")"
    codesign --verify --deep --strict "$copied_app"
    touch "$marker"
    runtime_import_check "$copied_app/Contents/MacOS/$(basename "$copied_app" .app)"
    if find "$copied_app" -name '*.pyc' -newer "$marker" -print -quit | grep -q .; then
        fail "runtime import wrote Python bytecode into the signed app bundle:" || {
            find "$copied_app" -name '*.pyc' -newer "$marker" -print | sed -n '1,20p' >&2
            return 1
        }
    fi
    codesign --verify --deep --strict "$copied_app"

    note "Smoke launch from mounted DMG"
    smoke_launch_verify "$bin" "$trace"

    note "Release artifact verification passed"
}

COMMAND=bundle
if [[ $# -gt 0 ]]; then
    case "$1" in
        check|bundle|verify|all)
            COMMAND="$1"
            shift
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
    esac
fi

VERIFY_ARTIFACT=""
case "$COMMAND" in
    check)
        [[ $# -eq 0 ]] || { usage >&2; exit 2; }
        ;;
    verify)
        [[ $# -eq 1 ]] || { usage >&2; exit 2; }
        VERIFY_ARTIFACT="$1"
        shift
        ;;
esac

SKIP_BUILD=0
SKIP_CHECK=0
OUTPUT_DIR="$BUNDLE_REPO_ROOT/build"

if [[ "$COMMAND" == "bundle" || "$COMMAND" == "all" ]]; then
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-build) SKIP_BUILD=1    ; shift ;;
            --skip-check) SKIP_CHECK=1    ; shift ;;
            --output)
                [[ $# -ge 2 ]] || { echo "Missing argument for --output" >&2; usage >&2; exit 2; }
                OUTPUT_DIR="$2"
                shift 2
                ;;
            *) echo "Unknown option: $1" >&2 ; usage >&2 ; exit 1 ;;
        esac
    done
elif [[ $# -gt 0 ]]; then
    usage >&2
    exit 2
fi

if [[ "$COMMAND" == "bundle" || "$COMMAND" == "all" ]]; then
    check_deps
fi

# Record the starting time of the script and the first step. SECONDS is bash's
# built-in elapsed-second counter.
T_SCRIPT=$SECONDS
T_STEP=$T_SCRIPT

# Single cleanup hook for every temp resource the script creates. Each
# variable is initialised empty here and populated lazily (in later steps).
# The trap is set once. Any later step can register a temp file or DMG mount
# without juggling its own trap.
CLOSURE_TMP=""
TRACE=""
DMG_MOUNT=""
cleanup() {
    [[ -n "$CLOSURE_TMP" ]] && rm -rf "$CLOSURE_TMP"
    [[ -n "$TRACE" ]] && rm -f "$TRACE"
    if [[ -n "$DMG_MOUNT" ]]; then
        hdiutil detach -quiet "$DMG_MOUNT" >/dev/null 2>&1 \
            || hdiutil detach -force -quiet "$DMG_MOUNT" >/dev/null 2>&1 \
            || true
        rmdir "$DMG_MOUNT" 2>/dev/null || true
    fi
}
trap cleanup EXIT

case "$COMMAND" in
    check)
        check_deps
        exit 0
        ;;
    verify)
        verify_dmg "$VERIFY_ARTIFACT"
        exit 0
        ;;
esac

# ---------------------------------------------------------------------------
# Derive paths
# ---------------------------------------------------------------------------

PY_VER=$(bundle_pyver)
BUILD_PATH=build/relbundle
APP="$BUNDLE_REPO_ROOT/$BUILD_PATH/cpp/binary/pilot/pilot.app"
BINARY="$APP/Contents/MacOS/pilot"
FW_DIR="$APP/Contents/Frameworks"

PY_DYLIB="$PREFIX/lib/libpython${PY_VER}.dylib"
BUNDLED_LIB="$APP/Contents/lib/python${PY_VER}"
BUNDLED_SITE="$BUNDLED_LIB/site-packages"

note "Build path : $BUILD_PATH"
note "App bundle : $APP"
note "Python     : $PY_DYLIB"

# ---------------------------------------------------------------------------
# Step 1: Build pilot
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
if [[ $SKIP_BUILD -eq 0 ]]; then
    # macdeployqt cannot redeploy on top of an existing .app: it
    # short-circuits but its later codesign pass still walks paths the
    # (skipped) deploy step would have populated, failing nested signing.
    # `make pilot` only re-links the binary; remove the .app so the
    # build produces a virgin one each run.
    rm -rf "$APP"
    note "make pilot CMAKE_BUILD_TYPE=Release BUILD_PATH=$BUILD_PATH"
    PYBIND11_CMAKE_DIR=$("$TOOLS_PREFIX/bin/python3" -m pybind11 --cmakedir)
    env -u CMAKE_PREFIX_PATH make -W CMakeLists.txt pilot \
        CMAKE_BUILD_TYPE=Release BUILD_PATH="$BUILD_PATH" \
        CMAKE_ARGS="${CMAKE_ARGS:-} --fresh -DCMAKE_PREFIX_PATH=\"$PREFIX\" -DQt6_DIR=\"$PREFIX/lib/cmake/Qt6\" -DPYTHON_EXECUTABLE=\"$PREFIX/bin/python3\" -Dpybind11_DIR=\"$PYBIND11_CMAKE_DIR\""
fi

[[ -f "$BINARY" ]] || fail "pilot binary not found at $BINARY"
echo "    [Step 1 (Build pilot): $((SECONDS - T_STEP))s]"

# ---------------------------------------------------------------------------
# Step 2: Deploy Qt
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
note "macdeployqt (bundles Qt frameworks, plugins, transitive dylibs)"
# Later steps modify the deployed files, so sign the finished bundle once.
macdeployqt "$APP" -verbose=1 -no-codesign -libpath="$PREFIX/lib"
for plugin_dir in "$APP"/Contents/PlugIns/*/; do
    [[ -d "$plugin_dir" ]] || continue
    plugin_name=$(basename "$plugin_dir")
    ln -sfn "../PlugIns/$plugin_name" "$APP/Contents/MacOS/$plugin_name"
done
echo "    [Step 2 (Deploy Qt): $((SECONDS - T_STEP))s]"

# ---------------------------------------------------------------------------
# Step 3: Bundle prepared SDK
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
note "Copying prepared SDK into the app bundle"
SDK_RSYNC_EXCLUDES=(
    # Generated caches and static typing metadata are rebuilt or unused.
    '--exclude=__pycache__'
    '--exclude=*.pyi'

    # CPython development tools and test suites are not used by Pilot.
    '--exclude=/lib/python*/ensurepip/'
    '--exclude=/lib/python*/idlelib/'
    '--exclude=/lib/python*/test/'
    '--exclude=/lib/python*/turtledemo/'
    '--exclude=/lib/python*/site-packages/**/test/'
    '--exclude=/lib/python*/site-packages/**/tests/'

    # PySide6 ships binding development files and standalone Qt tools.
    '--exclude=/lib/python*/site-packages/PySide6/*.app/'
    '--exclude=/lib/python*/site-packages/PySide6/glue/'
    '--exclude=/lib/python*/site-packages/PySide6/include/'
    '--exclude=/lib/python*/site-packages/PySide6/scripts/'
    '--exclude=/lib/python*/site-packages/PySide6/typesystems/'
    '--exclude=/lib/python*/site-packages/PySide6/lrelease'
    '--exclude=/lib/python*/site-packages/PySide6/lupdate'
    '--exclude=/lib/python*/site-packages/PySide6/qmlformat'
    '--exclude=/lib/python*/site-packages/PySide6/qmllint'
    '--exclude=/lib/python*/site-packages/PySide6/qmlls'
    '--exclude=/lib/python*/site-packages/PySide6/svgtoqml'

    # Headers, build metadata, documentation, and SDK tools are not runtime.
    '--exclude=/bin/'
    '--exclude=/config_qt*'
    '--exclude=/doc/'
    '--exclude=/examples/'
    '--exclude=/include/'
    '--exclude=/lib/*.a'
    '--exclude=/lib/cmake/'
    '--exclude=/lib/pkgconfig/'
    '--exclude=/libexec/'
    '--exclude=/metatypes/'
    '--exclude=/mkspecs/'
    '--exclude=/modules/'
    '--exclude=/phrasebooks/'
    '--exclude=/share/'

    # macdeployqt already places the required Qt runtime in the app bundle.
    '--exclude=/lib/Qt*.framework/'
    '--exclude=/plugins/'
    '--exclude=/qml/'
    '--exclude=/resources/'
    '--exclude=/translations/'
)
rsync -a "${SDK_RSYNC_EXCLUDES[@]}" "$PREFIX/" "$APP/Contents/"
chmod -R u+w "$APP/Contents/lib"
echo "    [Step 3 (Bundle prepared SDK): $((SECONDS - T_STEP))s]"

# ---------------------------------------------------------------------------
# Step 4: Replace the PySide6 Qt runtime
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
# Pilot's C++ code and the PySide6 bindings must use one Qt runtime. Point
# PySide6 to the frameworks deployed by macdeployqt instead of loading the Qt
# runtime originally supplied by its wheel.
PYSIDE_QT="$BUNDLED_SITE/PySide6/Qt"
rm -rf "$PYSIDE_QT"
mkdir -p "$PYSIDE_QT"
ln -s ../../../../../Frameworks "$PYSIDE_QT/lib"
echo "    [Step 4 (Replace the PySide6 Qt runtime): $((SECONDS - T_STEP))s]"

# ---------------------------------------------------------------------------
# Step 5: Bundle solvcon
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
note "Copying solvcon package into bundled site-packages"
# Copy the solvcon Python package. Skip _solvcon*.so because the binary code is
# statically linked into pilot via PYBIND11_EMBEDDED_MODULE.
rsync -a --exclude '__pycache__' --exclude '_solvcon*.so' \
    "$BUNDLE_REPO_ROOT/solvcon" "$BUNDLED_SITE/"
echo "    solvcon -> $BUNDLED_SITE/solvcon"
echo "    [Step 5 (Bundle solvcon): $((SECONDS - T_STEP))s]"

# ---------------------------------------------------------------------------
# Step 5.5: Pre-compile bundled bytecode
#
# Ship .pyc alongside .py so the embedded interpreter finds a valid cache on
# import and never writes back into the signed bundle (a runtime write would
# mutate the sealed contents and break the code signature). Use the bundled
# python so the .pyc magic matches the interpreter that will actually read
# them. unchecked-hash makes the cache trusted unconditionally, sidestepping
# mtime drift from rsync/DMG creation that would otherwise let the default
# timestamp mode falsely invalidate the cache at runtime.
#
# Target the whole lib/python<ver> tree so both the stdlib and the rsynced
# site-packages get a uniform unchecked-hash cache. The `-f` flag
# forces rewrite over the inherited stdlib .pyc.
#
# The SDK copy excludes test suites. Keep the regex for packages copied in a
# later step and for intentionally-uncompilable template files. Use `|| true`
# because neither kind of file is imported at runtime. The trailing sanity
# check still fails if compileall produced nothing at all, which would silently
# revert us to the writes-into-bundle problem.
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
note "Pre-compiling bundled Python modules"
PY_LIB="$BUNDLED_LIB"
python3 -m compileall -f --invalidation-mode unchecked-hash -q \
    -x '(^|/)(test|tests)/|\.tmpl\.py$' \
    "$PY_LIB" || true
PYC_COUNT=$(find "$PY_LIB" -name '*.pyc' | wc -l | tr -d ' ')
[[ $PYC_COUNT -gt 0 ]] || \
    fail "compileall produced no .pyc files under $PY_LIB"
echo "    compiled $PYC_COUNT .pyc files"
echo "    [Step 5.5 (Pre-compile bytecode): $((SECONDS - T_STEP))s]"

# ---------------------------------------------------------------------------
# Step 6: Vendor prefix deps
#
# macdeployqt only follows Qt's load commands, so other dependencies (Python's
# lib-dynload .so files, libpyside6, libshiboken6, etc.) still resolve through
# the SDK prefix at runtime. Copy them into Contents/Frameworks/ and rewrite
# their load commands and LC_RPATH entries.
# ---------------------------------------------------------------------------

# BFS-walk every Mach-O in the bundle: pop one, copy any missing prefix
# dependency into Contents/Frameworks/, enqueue what we just copied. Touching
# each file once keeps cost linear in bundle size. Sets the global SEEN_DYLIB /
# SEEN_FW indexes consumed by rewrite_load_commands below.
vendor_prefix_deps() {
    CLOSURE_TMP=$(mktemp -d)
    QUEUE="$CLOSURE_TMP/queue"
    SEEN_BIN="$CLOSURE_TMP/seen_bin"
    SEEN_DYLIB="$CLOSURE_TMP/seen_dylib"
    SEEN_FW="$CLOSURE_TMP/seen_fw"
    SEEN_MISSING="$CLOSURE_TMP/seen_missing"
    touch "$QUEUE" "$SEEN_BIN" "$SEEN_DYLIB" "$SEEN_FW" "$SEEN_MISSING"

    # Skip what's already in Contents/Frameworks/.
    local f
    for f in "$FW_DIR"/*.dylib; do
        [[ -f "$f" ]] && basename "$f" >> "$SEEN_DYLIB"
    done
    for f in "$FW_DIR"/*.framework; do
        [[ -d "$f" ]] && basename "$f" .framework >> "$SEEN_FW"
    done

    list_macho "$APP" | xargs -0 -n1 echo >> "$QUEUE"

    local BIN OLD fw SRC base cand
    while [[ -s "$QUEUE" ]]; do
        BIN=$(head -n 1 "$QUEUE")
        sed -i '' '1d' "$QUEUE"
        grep -qxF "$BIN" "$SEEN_BIN" && continue
        echo "$BIN" >> "$SEEN_BIN"
        while IFS= read -r OLD; do
            case "$OLD" in
            "$PREFIX"/*.framework/Versions/*/*)
                fw=${OLD##*/}
                grep -qxF "$fw" "$SEEN_FW" && continue
                SRC=${OLD%%/Versions/*}
                if [[ ! -d "$SRC" ]]; then
                    # Same handling as the dylib branch: warn once and let
                    # prune_unfixable_machos drop the dependant.
                    if ! grep -qxF "$OLD" "$SEEN_MISSING"; then
                        echo "$OLD" >> "$SEEN_MISSING"
                        echo "    WARN: missing source $OLD" >&2
                    fi
                    continue
                fi
                # Preserve internal symlinks but materialise Cellar- bound ones
                # (the binary itself), keeping the framework's on-disk shape
                # intact.
                mkdir -p "$FW_DIR/$fw.framework"
                if rsync -a --copy-unsafe-links \
                        "$SRC/" "$FW_DIR/$fw.framework/" 2>/dev/null; then
                    chmod -R u+w "$FW_DIR/$fw.framework"
                    echo "$fw" >> "$SEEN_FW"
                    list_macho "$FW_DIR/$fw.framework" \
                        | xargs -0 -n1 echo >> "$QUEUE"
                fi
                ;;
            "$PREFIX"/*)
                base=${OLD##*/}
                grep -qxF "$base" "$SEEN_DYLIB" && continue
                if [[ ! -f "$OLD" ]]; then
                    # Source not on this host; prune_unfixable_machos drops
                    # the dependant later. Warn once per unique path.
                    if ! grep -qxF "$OLD" "$SEEN_MISSING"; then
                        echo "$OLD" >> "$SEEN_MISSING"
                        echo "    WARN: missing source $OLD" >&2
                    fi
                    continue
                fi
                if cp -L "$OLD" "$FW_DIR/$base" 2>/dev/null; then
                    chmod u+w "$FW_DIR/$base"
                    echo "$base" >> "$SEEN_DYLIB"
                    echo "$FW_DIR/$base" >> "$QUEUE"
                fi
                ;;
            @rpath/lib*.dylib|@loader_path/lib*.dylib)
                # Sibling references such as @rpath/libpyside6.abi3.6.11.dylib.
                base=${OLD##*/}
                grep -qxF "$base" "$SEEN_DYLIB" && continue
                [[ "$OLD" == @loader_path/* && -f "$(dirname "$BIN")/${OLD#@loader_path/}" ]] && continue
                SRC=""
                for cand in \
                    "$PREFIX/lib/$base" \
                    "$PREFIX/lib/python${PY_VER}/site-packages/PySide6/$base" \
                    "$PREFIX/lib/python${PY_VER}/site-packages/shiboken6/$base"
                do
                    [[ -f "$cand" ]] && { SRC="$cand"; break; }
                done
                if [[ -z "$SRC" ]]; then
                    if ! grep -qxF "$OLD" "$SEEN_MISSING"; then
                        echo "$OLD" >> "$SEEN_MISSING"
                        echo "    WARN: could not resolve $OLD under $PREFIX" >&2
                    fi
                    continue
                fi
                if cp -L "$SRC" "$FW_DIR/$base" 2>/dev/null; then
                    chmod u+w "$FW_DIR/$base"
                    echo "$base" >> "$SEEN_DYLIB"
                    echo "$FW_DIR/$base" >> "$QUEUE"
                fi
                ;;
            esac
        done < <(otool -L "$BIN" 2>/dev/null | awk 'NR>1 {print $1}')
    done

    echo "    bundled $(wc -l < "$SEEN_DYLIB" | tr -d ' ') dylibs and \
$(wc -l < "$SEEN_FW" | tr -d ' ') frameworks"
}

# Rewrite every Mach-O reference whose target is now bundled (load commands,
# LC_ID_DYLIB, and the two Python re-export stubs that hard-code the original
# Python path). Reads SEEN_DYLIB / SEEN_FW populated by vendor_prefix_deps.
rewrite_load_commands() {
    # Surrounding spaces let bash do containment with the *" $tok "* glob.
    local INDEX FW_INDEX
    INDEX=" $(tr '\n' ' ' < "$SEEN_DYLIB")"
    FW_INDEX=" $(tr '\n' ' ' < "$SEEN_FW")"

    local BIN OLD fw ver base NEW
    while IFS= read -r -d '' BIN; do
        while IFS= read -r OLD; do
            case "$OLD" in
            "$PREFIX"/*.framework/Versions/*/*)
                fw=${OLD##*/}
                if [[ "$FW_INDEX" == *" $fw "* ]]; then
                    # Preserve the original version segment.
                    ver=${OLD#*.framework/Versions/}; ver=${ver%%/*}
                    NEW="@executable_path/../Frameworks/${fw}.framework/Versions/${ver}/${fw}"
                    install_name_tool -change "$OLD" "$NEW" "$BIN" 2>/dev/null || true
                fi
                ;;
            "$PREFIX"/*)
                base=${OLD##*/}
                if [[ "$INDEX" == *" $base "* ]]; then
                    NEW="@executable_path/../Frameworks/$base"
                    install_name_tool -change "$OLD" "$NEW" "$BIN" 2>/dev/null || true
                fi
                ;;
            @rpath/lib*.dylib)
                base=${OLD#@rpath/}
                if [[ "$INDEX" == *" $base "* ]]; then
                    NEW="@executable_path/../Frameworks/$base"
                    install_name_tool -change "$OLD" "$NEW" "$BIN" 2>/dev/null || true
                fi
                ;;
            esac
        done < <(otool -L "$BIN" 2>/dev/null | awk 'NR>1 {print $1}')
    done < <(list_macho "$APP")

    # Each bundled dylib/framework's own LC_ID_DYLIB still names its original
    # prefix path; rewrite so dyld dedupes by install name.
    local f fw_path name vname bin_path VER
    for f in "$FW_DIR"/*.dylib; do
        [[ -f "$f" ]] || continue
        install_name_tool -id \
            "@executable_path/../Frameworks/$(basename "$f")" "$f" \
            2>/dev/null || true
    done
    for fw_path in "$FW_DIR"/*.framework; do
        [[ -d "$fw_path" ]] || continue
        name=$(basename "$fw_path" .framework)
        for VER in "$fw_path"/Versions/*; do
            [[ -d "$VER" && ! -L "$VER" ]] || continue
            bin_path="$VER/$name"
            [[ -f "$bin_path" ]] || continue
            vname=$(basename "$VER")
            install_name_tool -id \
                "@executable_path/../Frameworks/${name}.framework/Versions/${vname}/${name}" \
                "$bin_path" 2>/dev/null || true
        done
    done

}

# LC_RPATH is invisible to otool -L so the rewrite above missed it. A surviving
# prefix rpath silently falls back at runtime on the build machine and fails
# everywhere else. For each affected Mach-O: add
# @executable_path/../Frameworks (so bundled siblings stay reachable via
# @rpath/), then delete every prefix entry one at a time
# (install_name_tool removes one per call).
strip_prefix_rpaths() {
    local EP_RPATH='@executable_path/../Frameworks'
    local BIN prefix_rps rp
    while IFS= read -r -d '' BIN; do
        prefix_rps=$(binary_rpaths "$BIN" | grep -E "$HOST_PREFIX_RE" || true)
        [[ -z "$prefix_rps" ]] && continue
        if ! binary_rpaths "$BIN" | grep -qxF "$EP_RPATH"; then
            install_name_tool -add_rpath "$EP_RPATH" "$BIN" 2>/dev/null || true
        fi
        while IFS= read -r rp; do
            [[ -z "$rp" ]] && continue
            install_name_tool -delete_rpath "$rp" "$BIN" 2>/dev/null || true
        done <<< "$prefix_rps"
    done < <(list_macho "$APP")
}

# Drop any Mach-O still referencing the SDK prefix after vendor + rewrite have
# run, typically Qt plugins pre-linked against libraries absent from the SDK.
# dyld would fail to load them
# anyway. Skips framework MAIN binaries since removing those breaks the
# framework structure; that case is better surfaced by the static check.
prune_unfixable_machos() {
    local count=0 BIN load_refs rpath_refs fname fwdir
    while IFS= read -r -d '' BIN; do
        # Skip path <NAME>.framework/Versions/<V>/<NAME>. Plugins and Python
        # C extensions nested deeper than that are not main binaries.
        case "$BIN" in
        *.framework/Versions/*)
            fname="${BIN##*/}"
            fwdir="${BIN%/Versions/*}"
            if [[ "${fwdir##*/}" == "${fname}.framework" ]]; then
                continue
            fi
            ;;
        esac
        load_refs=$(otool -L "$BIN" 2>/dev/null | tail -n +2 | awk '{print $1}' \
            | grep -E "$HOST_PREFIX_RE" || true)
        rpath_refs=$(binary_rpaths "$BIN" | grep -E "$HOST_PREFIX_RE" || true)
        if [[ -n "$load_refs" || -n "$rpath_refs" ]]; then
            rm -f "$BIN"
            echo "    WARN: pruned ${BIN#"$APP/"}" >&2
            count=$((count+1))
        fi
    done < <(list_macho "$APP")
    if [[ $count -gt 0 ]]; then
        # Stderr + WARN prefix: each pruned file is a feature pilot loses on
        # this host, even if startup smoke-launch still passes. User can
        # install the missing keg and rerun to keep it.
        echo "    WARN: pruned $count Mach-O file(s) with unresolvable prefix deps" >&2
    fi
}

thin_bundle_arm64() {
    local count=0 BIN archs mode thin
    while IFS= read -r -d '' BIN; do
        archs=$(lipo -archs "$BIN" 2>/dev/null || true)
        [[ "$archs" == *x86_64* ]] || continue
        [[ "$archs" == *arm64* ]] || {
            fail "x86_64-only Mach-O in Apple Silicon bundle: $BIN"
            return 1
        }

        mode=$(stat -f '%Lp' "$BIN")
        thin="${BIN}.arm64"
        rm -f "$thin"
        if ! lipo "$BIN" -thin arm64 -output "$thin"; then
            rm -f "$thin"
            return 1
        fi
        chmod "$mode" "$thin"
        mv -f "$thin" "$BIN"
        count=$((count+1))
    done < <(list_macho "$APP")

    while IFS= read -r -d '' BIN; do
        archs=$(lipo -archs "$BIN" 2>/dev/null || true)
        [[ "$archs" != *x86_64* ]] || {
            fail "x86_64 remains in $BIN"
            return 1
        }
    done < <(list_macho "$APP")
    echo "    thinned $count universal Mach-O file(s) to arm64"
}

# Set PYTHONNOUSERSITE=1 in LSEnvironment so a Finder/open launch of pilot
# never adds ~/Library/Python/<X.Y>/lib/python/site-packages to sys.path,
# which on dev machines may carry a second PySide6 and Qt runtime.
inject_pythonnousersite() {
    local plist="$APP/Contents/Info.plist"
    [[ -f "$plist" ]] || { echo "    SKIP: no Info.plist at $plist" >&2; return 0; }
    # macdeployqt's Step 2 codesign locks Info.plist read-only; Step 7
    # re-locks it.
    chmod u+w "$plist"
    if /usr/libexec/PlistBuddy -c "Print :LSEnvironment:PYTHONNOUSERSITE" \
            "$plist" >/dev/null 2>&1; then
        /usr/libexec/PlistBuddy -c \
            "Set :LSEnvironment:PYTHONNOUSERSITE 1" "$plist" || {
            fail "failed to set LSEnvironment.PYTHONNOUSERSITE"
            return 1
        }
    else
        # Dict may already exist; tolerate either way.
        /usr/libexec/PlistBuddy -c "Add :LSEnvironment dict" "$plist" \
            2>/dev/null || true
        /usr/libexec/PlistBuddy -c \
            "Add :LSEnvironment:PYTHONNOUSERSITE string 1" "$plist" || {
            fail "failed to add LSEnvironment.PYTHONNOUSERSITE"
            return 1
        }
    fi
    echo "    Info.plist LSEnvironment.PYTHONNOUSERSITE = 1"
}

T_STEP=$SECONDS
note "Vendoring prefix dependencies"
vendor_prefix_deps
note "Redirecting prefix load commands to bundled copies"
rewrite_load_commands
note "Stripping prefix rpaths"
strip_prefix_rpaths
note "Pruning Mach-O files with unresolvable prefix deps"
prune_unfixable_machos
note "Thinning universal Mach-O files to arm64"
thin_bundle_arm64
note "Setting PYTHONNOUSERSITE=1 in Info.plist"
inject_pythonnousersite
echo "    [Step 6 (Vendor prefix deps): $((SECONDS - T_STEP))s]"

# ---------------------------------------------------------------------------
# Step 7: Ad-hoc codesign
#
# Sign bottom-up. `codesign --deep` would skip files we added after the
# original signing so each nested artifact must be sealed first, in dependency
# order: dylibs/.so, then nested helper .apps, then frameworks, then the main
# binary and main .app.
# ---------------------------------------------------------------------------

# Sign every standalone dylib and .so under Contents/.
sign_inner_machos() {
    local BIN
    while IFS= read -r -d '' BIN; do
        codesign --force --sign - "$BIN" 2>/dev/null || true
    done < <(find "$APP/Contents" \( -name '*.dylib' -o -name '*.so' \) -print0)
}

# Sign nested helper apps (e.g. QtWebEngineCore's QtWebEngineProcess.app)
# before sealing their containing framework.  Reverse-sort by path
# length so the deepest .app signs first.
sign_helper_apps() {
    local INNER_APP
    find "$APP" -name '*.app' -type d -print | awk '{print length($0), $0}' | \
        sort -k1,1nr | cut -d' ' -f2- | while IFS= read -r INNER_APP; do
        [[ "$INNER_APP" == "$APP" ]] && continue
        rm -rf "$INNER_APP/Contents/_CodeSignature"
        codesign --force --sign - "$INNER_APP" 2>/dev/null || true
    done
}

# Re-seal each nested framework (Qt*.framework, ...).
# On macOS the framework version directory carries the signature.
sign_frameworks() {
    local FW VER
    for FW in "$FW_DIR"/*.framework; do
        [[ -d "$FW" ]] || continue
        for VER in "$FW"/Versions/*; do
            [[ -d "$VER" && ! -L "$VER" ]] || continue
            rm -rf "$VER/_CodeSignature"
            codesign --force --sign - "$VER" 2>/dev/null || true
        done
    done
}

# Sign the main binary, then the outer .app.
sign_main_app() {
    codesign --force --sign - "$BINARY"
    codesign --force --sign - "$APP"
}

T_STEP=$SECONDS
note "Ad-hoc re-signing all bundled Mach-O files"
sign_inner_machos
sign_helper_apps
sign_frameworks
note "Ad-hoc re-signing the app bundle"
sign_main_app
echo "    [Step 7 (Ad-hoc codesign): $((SECONDS - T_STEP))s]"

echo ""
ok "Bundle complete: $APP"

# ---------------------------------------------------------------------------
# Step 8: Package DMG
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
mkdir -p "$OUTPUT_DIR"
DMG="$OUTPUT_DIR/pilot.dmg"
note "Creating $DMG from $APP"
hdiutil create -volname "solvcon Pilot" \
    -srcfolder "$APP" \
    -ov -format UDZO \
    "$DMG"
SIZE=$(du -sh "$DMG" | cut -f1)
echo "    [Step 8 (Package DMG): $((SECONDS - T_STEP))s]"

echo ""
ok "DMG complete: $DMG ($SIZE)"

# ---------------------------------------------------------------------------
# Step 9: Verify hermeticity
# ---------------------------------------------------------------------------

launch_and_trace() {
    local bin="$1" pid elapsed=0 loaded
    : > "$TRACE"
    env -i HOME="${HOME:-/}" USER="${USER:-nobody}" \
        PATH=/usr/bin:/bin TERM="${TERM:-xterm}" \
        DYLD_PRINT_LIBRARIES=1 \
        "$bin" >/dev/null 2>"$TRACE" &
    pid=$!
    while [[ $elapsed -lt 15 ]]; do
        sleep 1
        elapsed=$((elapsed+1))
        loaded=$(grep -c '^dyld\[' "$TRACE" 2>/dev/null || true)
        [[ ${loaded:-0} -ge $MIN_LOADS ]] && break
    done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

smoke_check() {
    local bin="$1" label="${2:-}" total prefix_count
    [[ -n "$label" ]] && label=" ($label)"
    note "Smoke launch${label} under sandboxed env (DYLD_PRINT_LIBRARIES=1)"
    launch_and_trace "$bin"
    total=$(grep -c '^dyld\[' "$TRACE" || true)
    prefix_count=$(grep -E '^dyld\[' "$TRACE" \
        | grep -E -c "${HOST_PREFIX_RE#^}" || true)
    if [[ $total -lt $MIN_LOADS ]]; then
        fail "pilot loaded only $total libraries (expected >= $MIN_LOADS);" \
            "did it crash early?" || {
            sed -n '1,40p' "$TRACE" >&2
            exit 1
        }
    fi
    if [[ $prefix_count -gt 0 ]]; then
        fail "$prefix_count of $total dyld loads came from dependency prefixes:" || {
            grep -E '^dyld\[' "$TRACE" | grep -E "${HOST_PREFIX_RE#^}" \
                | head -10 | sed 's/^/    /' >&2
            exit 1
        }
    fi
    ok "$total libraries loaded, none from dependency prefixes"
}

if [[ $SKIP_CHECK -eq 0 ]]; then
    T_STEP=$SECONDS
    TRACE=$(mktemp -t check-bundle-trace)

    echo ""
    note "Checking $APP"
    note "Static scan for dependency path references"
    check_self_contained "$APP"
    smoke_check "$BINARY"
    echo ""
    ok "Bundle check passed: $APP"

    DMG_MOUNT=$(mktemp -d -t check-dmg)
    echo ""
    note "Mounting $DMG at $DMG_MOUNT"
    hdiutil attach -nobrowse -readonly -mountpoint "$DMG_MOUNT" "$DMG" \
        >/dev/null

    DMG_APP=$(find "$DMG_MOUNT" -maxdepth 2 -name '*.app' -type d -print -quit)
    [[ -n "$DMG_APP" ]] || fail "no .app found in $DMG"
    note "Found app: $DMG_APP"
    note "Static scan (DMG) for dependency path references"
    check_self_contained "$DMG_APP"
    DMG_BINARY="$DMG_APP/Contents/MacOS/$(basename "$DMG_APP" .app)"
    smoke_check "$DMG_BINARY" "DMG"
    echo ""
    ok "DMG check passed: $DMG"
    echo "    [Step 9 (Verify hermeticity): $((SECONDS - T_STEP))s]"
fi

echo ""
echo "Total elapsed: $((SECONDS - T_SCRIPT))s"

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
