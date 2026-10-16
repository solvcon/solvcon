#!/bin/bash
#
# This script is a prototype to create a self-contained macOS .app bundle for
# solvcon, package it into a DMG file, and verify both are hermetic. It employs
# macdeployqt, otool, and install_name_tool to process the binaires, and ad-hoc
# codesign to reseal them.
#
# !! WARNING !!
# The script is only tested in a standalone VM environment, and assumes the
# dependencies are from homebrew. Be very careful when running it in a
# development or production machine.
#
# Usage:
#   ./contrib/bundle/bundle-with-homebrew.sh check
#   ./contrib/bundle/bundle-with-homebrew.sh bundle [--skip-build]
#                                                   [--skip-check]
#                                                   [--output DIR]
#   ./contrib/bundle/bundle-with-homebrew.sh verify path/to/pilot.dmg
#
# Legacy usage without a subcommand still runs the bundle step:
#   ./contrib/bundle/bundle-with-homebrew.sh [--skip-build] [--skip-check]
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
HOST_PREFIX_RE='^(/opt/homebrew|/usr/local)'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  contrib/bundle/bundle-with-homebrew.sh check
  contrib/bundle/bundle-with-homebrew.sh bundle [--skip-build] [--skip-check] [--output DIR]
  contrib/bundle/bundle-with-homebrew.sh verify path/to/pilot.dmg
  contrib/bundle/bundle-with-homebrew.sh all [--skip-build] [--skip-check] [--output DIR]

Subcommands:
  check    Check macOS bundle release dependencies. Does not build or install.
  bundle   Check dependencies, then build/package pilot.app and pilot.dmg.
  verify   Verify a generated release DMG artifact.
  all      Alias for bundle. (No verify.)
EOF
}

note() { printf '==> %s\n' "$*"; }
ok() { printf 'OK    %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*" >&2; return 1; }

require_macos() {
    [[ "$(uname -s)" == "Darwin" ]] || fail "macOS is required for macOS bundle release"
}

setup_hint() {
    cat >&2 <<'EOF'

For a fresh macOS bundling environment, try:
  bash contrib/vm/macos/mac26_vmsetup.sh homebrew
  bash contrib/vm/macos/mac26_vmsetup.sh dependency
EOF
}

find_brew() {
    if command -v brew >/dev/null 2>&1; then
        command -v brew
    elif [[ -x /opt/homebrew/bin/brew ]]; then
        printf '%s\n' /opt/homebrew/bin/brew
    elif [[ -x /usr/local/bin/brew ]]; then
        printf '%s\n' /usr/local/bin/brew
    else
        return 1
    fi
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

python_framework_check() {
    python3 <<'PY'
import os
import sys
import sysconfig

fw = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX")
if not fw:
    raise SystemExit("PYTHONFRAMEWORKPREFIX is empty; Homebrew framework Python is required")
ver = f"{sys.version_info[0]}.{sys.version_info[1]}"
path = os.path.join(fw, "Python.framework", "Versions", ver, "Python")
if not os.path.isfile(path):
    raise SystemExit(f"Python dylib not found: {path}")
print(path)
PY
}

check_pybind11_cmake() {
    local config_dir prefix
    if command -v pybind11-config >/dev/null 2>&1; then
        config_dir=$(pybind11-config --cmakedir 2>/dev/null || true)
        if [[ -n "$config_dir" && -f "$config_dir/pybind11Config.cmake" ]]; then
            ok "pybind11 CMake config: $config_dir/pybind11Config.cmake"
            return 0
        fi
    fi
    prefix=$(brew --prefix pybind11 2>/dev/null || true)
    if [[ -n "$prefix" && -f "$prefix/share/cmake/pybind11/pybind11Config.cmake" ]]; then
        ok "pybind11 CMake config: $prefix/share/cmake/pybind11/pybind11Config.cmake"
    else
        fail "pybind11 CMake config: not found (brew install pybind11)"
    fi
}

check_deps() {
    local brew_exe brew_prefix python_path fw_path failed=0
    require_macos

    if brew_exe=$(find_brew); then
        ok "brew: $brew_exe"
        eval "$("$brew_exe" shellenv)"
        brew_prefix=$(brew --prefix)
        ok "Homebrew prefix: $brew_prefix"
    else
        fail "brew: not found (install Homebrew)" || true
        setup_hint
        exit 1
    fi

    for cmd in cmake make python3 macdeployqt qtpaths otool \
        install_name_tool codesign hdiutil rsync shasum; do
        require_command "$cmd" || failed=1
    done

    python_path=$(command -v python3 || true)
    if [[ -n "$python_path" && "$python_path" == "$brew_prefix"/* ]]; then
        ok "python3 is under Homebrew prefix"
    else
        fail "python3 is not from Homebrew prefix ($python_path; expected under $brew_prefix)" || failed=1
    fi

    if fw_path=$(python_framework_check 2>&1); then
        ok "Python framework dylib: $fw_path"
    else
        fail "$fw_path" || failed=1
    fi

    check_pybind11_cmake || failed=1
    python_module_check numpy "brew install numpy" || failed=1
    python_module_check PySide6 "brew install pyside" || failed=1
    python_module_check shiboken6 "brew install pyside" || failed=1
    python_module_check shiboken6_generator "brew install pyside" || failed=1
    python_module_check matplotlib "brew install python-matplotlib" || failed=1

    if [[ $failed -ne 0 ]]; then
        setup_hint
        exit 1
    fi
}

# Active Python's "X.Y" version string.
bundle_pyver() {
    python3 -c "import sys; print('%d.%d' % sys.version_info[:2])"
}

# Print every Mach-O under $1, NUL-delimited: standalone dylibs, .so
# extensions, MacOS/ binaries, and the main binary of every nested framework.
# Framework binaries cannot be matched by file mode -- Homebrew ships them as
# 0644 -- so synthesise the canonical path (Versions/<v>/<name>) explicitly.
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

# Verify the .app under $1 contains no surviving Homebrew prefix reference
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
        echo "    OK: $total Mach-O files, none reference /opt/homebrew or /usr/local"
        return 0
    fi
    echo "ERROR: $bad of $total Mach-O files still reference Homebrew prefixes" >&2
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
    env -i HOME="${HOME:-/}" USER="${USER:-nobody}" \
        PATH=/usr/bin:/bin TERM="${TERM:-xterm}" \
        "$bin" --mode=python -c \
        "import matplotlib; import solvcon.pilot._base_app"
}

smoke_launch_verify() {
    local bin="$1" trace="$2" pid elapsed=0 loaded
    : > "$trace"
    env -i HOME="${HOME:-/}" USER="${USER:-nobody}" \
        PATH=/usr/bin:/bin TERM="${TERM:-xterm}" \
        DYLD_PRINT_LIBRARIES=1 \
        "$bin" >/dev/null 2>"$trace" &
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
        echo "ERROR: pilot loaded only $loaded libraries; did it crash early?" >&2
        sed -n '1,80p' "$trace" >&2
        return 1
    fi
    if grep -E '^dyld\[' "$trace" | grep -E '/opt/homebrew|/usr/local' >/dev/null; then
        echo "ERROR: smoke launch loaded libraries from host package prefixes:" >&2
        grep -E '^dyld\[' "$trace" | grep -E '/opt/homebrew|/usr/local' | head -20 >&2
        return 1
    fi
    echo "    OK: $loaded libraries loaded, none from host package prefixes"
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
    trap cleanup_verify EXIT

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
        echo "ERROR: runtime import wrote Python bytecode into the signed app bundle:" >&2
        find "$copied_app" -name '*.pyc' -newer "$marker" -print | sed -n '1,20p' >&2
        return 1
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

# Single cleanup hook for every temp resource the script creates. EachS
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

# Locate the Python framework used at build time.
PY_FW=$(python3 -c "
import sys, os, sysconfig
fw = sysconfig.get_config_var('PYTHONFRAMEWORKPREFIX')
if fw:
    print(os.path.join(fw, 'Python.framework'))
else:
    base = os.path.dirname(os.path.dirname(sys.executable))
    print(os.path.join(base, 'Frameworks', 'Python.framework'))
")
PY_DYLIB="$PY_FW/Versions/$PY_VER/Python"
NEW_PY_PATH="@executable_path/../Frameworks/Python.framework/Versions/$PY_VER/Python"

# Every site-packages dir the embedded interpreter would search, expanded
# through .pth files (so packages installed under separate Homebrew prefixes
# such as the matplotlib keg are found). macOS bash 3.2 has no mapfile; read
# into an array with a loop.
SITE_DIRS=()
while IFS= read -r line; do
    SITE_DIRS+=("$line")
done < <(python3 -c "
import site, os
seen, out = set(), []
def add(p):
    p = os.path.realpath(p)
    if p in seen or not os.path.isdir(p):
        return
    seen.add(p); out.append(p)
    for f in sorted(os.listdir(p)):
        if not f.endswith('.pth'):
            continue
        try:
            with open(os.path.join(p, f)) as fp:
                for line in fp:
                    line = line.strip()
                    if line and not line.startswith(('#', 'import ')):
                        add(line if os.path.isabs(line) else os.path.join(p, line))
        except OSError:
            pass
for d in site.getsitepackages():
    add(d)
print('\n'.join(out))
")

# Resolve the closure of Python packages pilot actually imports, so Step 4
# copies only those, not the developer's full site-packages (which can
# contain unrelated packages: vtkmodules, openvino, PyQt6, ...). Override
# seeds via BUNDLE_SEED_MODULES; force-add via BUNDLE_EXTRA_PACKAGES (for
# lazy / runtime imports the probe cannot observe). A failed import aborts
# the script with the missing name(s); output goes through a file in
# CLOSURE_TMP, which the exit trap removes, since `done < <(...)` would
# swallow python3's exit status.
CLOSURE_TMP=$(mktemp -d -t solvcon-bundle-closure)
if ! python3 -c "
import importlib, os, sys
seeds = os.environ.get('BUNDLE_SEED_MODULES',
    'PySide6.QtCore,PySide6.QtWidgets,PySide6.QtGui,numpy,matplotlib,'
    'dateutil.relativedelta,dateutil.tz,dateutil.parser').split(',')
extras = os.environ.get('BUNDLE_EXTRA_PACKAGES', '').split(',')
missing = []
for m in seeds + extras:
    m = m.strip()
    if not m: continue
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(m)
        sys.stderr.write('ERROR: required package %s failed to import: %s\n' % (m, e))
if missing:
    sys.stderr.write('ERROR: missing package(s): %s\n' % ', '.join(missing))
    raise SystemExit(1)
out = set()
for name, mod in list(sys.modules.items()):
    if mod is None or '.' in name: continue
    p = getattr(mod, '__path__', None)
    f = getattr(mod, '__file__', None) or ''
    if p:
        for d in p:
            if 'site-packages' in d:
                out.add(os.path.realpath(d)); break
    elif 'site-packages' in f:
        out.add(os.path.realpath(f))
print('\n'.join(sorted(out)))
" > "$CLOSURE_TMP/probe"
then
    echo "ERROR: aborting before bundling (see missing package(s) above)" >&2
    exit 1
fi

WANTED_PATHS=()
while IFS= read -r line; do
    [[ -n "$line" ]] && WANTED_PATHS+=("$line")
done < "$CLOSURE_TMP/probe"
# Step 6 reassigns CLOSURE_TMP, so release this one now.
rm -rf "$CLOSURE_TMP"
CLOSURE_TMP=""

echo "==> Build path : $BUILD_PATH"
echo "==> App bundle : $APP"
echo "==> Python fw  : $PY_FW"
for d in "${SITE_DIRS[@]}"; do
    echo "==> site-pkgs  : $d"
done

[[ -f "$PY_DYLIB" ]] || { echo "ERROR: Python dylib not found at $PY_DYLIB" >&2; exit 1; }

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
    echo "==> make pilot BUILD_PATH=$BUILD_PATH"
    make pilot BUILD_PATH="$BUILD_PATH"
fi

[[ -f "$BINARY" ]] || { echo "ERROR: pilot binary not found at $BINARY" >&2; exit 1; }
echo "    [Step 1 (Build pilot): $((SECONDS - T_STEP))s]"

# ---------------------------------------------------------------------------
# Step 2: Deploy Qt
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
echo "==> macdeployqt (bundles Qt frameworks, plugins, transitive dylibs)"
# macdeployqt looks for sibling frameworks under <prefix>/lib/, but Homebrew
# puts Python.framework under <prefix>/Frameworks/. This triggers two cosmetic
# ERROR lines (exit 0; Qt deployment unaffected; Step 3 bundles Python
# ourselves). Filter just those lines; pipefail keeps real failures visible.
macdeployqt "$APP" -verbose=1 2>&1 | sed -E \
    -e '/^ERROR: .*otool-classic: can.t open file: .*lib\/Python\.framework.*No such file or directory/d' \
    -e '/^ERROR: no file at ".*lib\/Python\.framework/d'
echo "    [Step 2 (Deploy Qt): $((SECONDS - T_STEP))s]"

# ---------------------------------------------------------------------------
# Step 3: Bundle Python framework
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
echo "==> Bundling Python.framework"

DEST_FW="$FW_DIR/Python.framework"
rm -rf "$DEST_FW"
cp -R "$PY_FW" "$DEST_FW"

install_name_tool -change "$PY_DYLIB" "$NEW_PY_PATH" "$BINARY"
echo "    Python -> $NEW_PY_PATH"

# site-packages inside the framework is a symlink into Cellar; replace with a
# real directory so subsequent steps can drop packages in.
BUNDLED_SITE="$DEST_FW/Versions/$PY_VER/lib/python${PY_VER}/site-packages"
rm -rf "$BUNDLED_SITE"
mkdir -p "$BUNDLED_SITE"
echo "    [Step 3 (Bundle Python framework): $((SECONDS - T_STEP))s]"

# ---------------------------------------------------------------------------
# Step 4: Bundle site-packages
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
echo "==> Copying Python packages into bundled site-packages"
# Copy the closure resolved above (dir => rsync tree, file => cp).
# -L dereferences symlinks. Falls back to every SITE_DIR if the probe
# failed, so the script still runs when import discovery breaks.
if [[ ${#WANTED_PATHS[@]} -gt 0 ]]; then
    echo "    closure: ${#WANTED_PATHS[@]} entries (override via BUNDLE_SEED_MODULES / BUNDLE_EXTRA_PACKAGES)"
    for path in "${WANTED_PATHS[@]}"; do
        if [[ -d "$path" ]]; then
            rsync -aL --exclude '__pycache__' --exclude '*.pth' \
                "$path" "$BUNDLED_SITE/"
        elif [[ -f "$path" ]]; then
            cp -L "$path" "$BUNDLED_SITE/"
        fi
    done
else
    echo "    WARNING: import-closure discovery failed; copying full SITE_DIRS"
    for SITE in "${SITE_DIRS[@]}"; do
        rsync -aL --exclude '__pycache__' --exclude '*.pth' \
            "$SITE/" "$BUNDLED_SITE/"
    done
fi
echo "    [Step 4 (Bundle site-packages): $((SECONDS - T_STEP))s]"

# ---------------------------------------------------------------------------
# Step 5: Bundle solvcon
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
echo "==> Copying solvcon package into bundled site-packages"
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
# Target the whole lib/python<ver> tree so both the stdlib (which Step 3
# copied in via `cp -R`, including Homebrew's timestamp-mode .pyc) and the
# rsynced site-packages get a uniform unchecked-hash cache. The `-f` flag
# forces rewrite over the inherited stdlib .pyc.
#
# Drop lib/python<ver>/test first: CPython's own test suite is not used at
# runtime, accounts for ~100 MB, and ships intentionally-bad-syntax fixtures
# that would make compileall exit nonzero. Use `|| true` so any stray
# uncompilable file in a package's own tests/ does not kill the build --
# those files are not imported at runtime, so missing .pyc is harmless. The
# trailing sanity check fails loudly if compileall produced nothing at all,
# which would silently revert us to the writes-into-bundle problem.
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
echo "==> Pre-compiling bundled Python modules"
PY_LIB="$DEST_FW/Versions/$PY_VER/lib/python$PY_VER"
rm -rf "$PY_LIB/test"
"$DEST_FW/Versions/$PY_VER/bin/python$PY_VER" -m compileall -f \
    --invalidation-mode unchecked-hash -q \
    "$PY_LIB" || true
PYC_COUNT=$(find "$PY_LIB" -name '*.pyc' | wc -l | tr -d ' ')
[[ $PYC_COUNT -gt 0 ]] || \
    { echo "ERROR: compileall produced no .pyc files under $PY_LIB" >&2; exit 1; }
echo "    compiled $PYC_COUNT .pyc files"
echo "    [Step 5.5 (Pre-compile bytecode): $((SECONDS - T_STEP))s]"

# ---------------------------------------------------------------------------
# Step 6: Vendor Homebrew deps
#
# macdeployqt only follows Qt's load commands, so other dependencies (Python's
# lib-dynload .so files, libpyside6, libshiboken6, etc.) still resolve through
# /opt/homebrew at runtime. Walk every Mach-O in the bundle, copy missing
# /opt/homebrew dylibs/frameworks into Contents/Frameworks/, rewrite load
# commands and LC_ID_DYLIB to the bundled copy, and strip /opt/homebrew from
# LC_RPATH.
# ---------------------------------------------------------------------------

# BFS-walk every Mach-O in the bundle: pop one, copy any missing /opt/homebrew
# dependency into Contents/Frameworks/, enqueue what we just copied. Touching
# each file once keeps cost linear in bundle size. Sets the global SEEN_DYLIB /
# SEEN_FW indexes consumed by rewrite_load_commands below.
vendor_homebrew_deps() {
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
            /opt/homebrew/*.framework/Versions/*/*)
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
            /opt/homebrew/*)
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
                # Sibling references like libgfortran -> @rpath/
                # libgcc_s.1.1.dylib, where Homebrew ships the sibling next to
                # the loader under one of its keg paths.
                base=${OLD##*/}
                grep -qxF "$base" "$SEEN_DYLIB" && continue
                SRC=""
                for cand in \
                    "/opt/homebrew/lib/$base" \
                    /opt/homebrew/opt/*/lib/"$base" \
                    /opt/homebrew/opt/*/lib/gcc/current/"$base"
                do
                    [[ -f "$cand" ]] && { SRC="$cand"; break; }
                done
                if [[ -z "$SRC" ]]; then
                    if ! grep -qxF "$OLD" "$SEEN_MISSING"; then
                        echo "$OLD" >> "$SEEN_MISSING"
                        echo "    WARN: could not resolve $OLD under /opt/homebrew" >&2
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
# Python path). Reads SEEN_DYLIB / SEEN_FW populated by vendor_homebrew_deps.
rewrite_load_commands() {
    # Surrounding spaces let bash do containment with the *" $tok "* glob.
    local INDEX FW_INDEX
    INDEX=" $(tr '\n' ' ' < "$SEEN_DYLIB")"
    FW_INDEX=" $(tr '\n' ' ' < "$SEEN_FW")"

    local BIN OLD fw ver base NEW
    while IFS= read -r -d '' BIN; do
        while IFS= read -r OLD; do
            case "$OLD" in
            /opt/homebrew/*.framework/Versions/*/*)
                fw=${OLD##*/}
                if [[ "$FW_INDEX" == *" $fw "* ]]; then
                    # Preserve the original version segment.
                    ver=${OLD#*.framework/Versions/}; ver=${ver%%/*}
                    NEW="@executable_path/../Frameworks/${fw}.framework/Versions/${ver}/${fw}"
                    install_name_tool -change "$OLD" "$NEW" "$BIN" 2>/dev/null || true
                fi
                ;;
            /opt/homebrew/*)
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
    # Homebrew path; rewrite so dyld dedupes by install name.
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

    # Two re-export stubs inside Python.framework still hard-code the original
    # Homebrew Python path. Redirect them to the bundled copy.
    for f in \
        "$DEST_FW/Versions/$PY_VER/lib/libpython${PY_VER}.dylib" \
        "$DEST_FW/Versions/$PY_VER/lib/python${PY_VER}/config-${PY_VER}-darwin/libpython${PY_VER}.dylib"
    do
        [[ -f "$f" ]] || continue
        install_name_tool -change "$PY_DYLIB" "$NEW_PY_PATH" "$f" 2>/dev/null || true
    done
}

# LC_RPATH is invisible to otool -L so the rewrite above missed it. A surviving
# /opt/homebrew rpath silently falls back at runtime on Homebrew machines and
# fails everywhere else. For each affected Mach-O: add
# @executable_path/../Frameworks (so bundled siblings stay reachable via
# @rpath/), then delete every /opt/homebrew entry one at a time
# (install_name_tool removes one per call).
strip_homebrew_rpaths() {
    local EP_RPATH='@executable_path/../Frameworks'
    local BIN homebrew_rps rp
    while IFS= read -r -d '' BIN; do
        homebrew_rps=$(binary_rpaths "$BIN" | grep -E '^/opt/homebrew' || true)
        [[ -z "$homebrew_rps" ]] && continue
        if ! binary_rpaths "$BIN" | grep -qxF "$EP_RPATH"; then
            install_name_tool -add_rpath "$EP_RPATH" "$BIN" 2>/dev/null || true
        fi
        while IFS= read -r rp; do
            [[ -z "$rp" ]] && continue
            install_name_tool -delete_rpath "$rp" "$BIN" 2>/dev/null || true
        done <<< "$homebrew_rps"
    done < <(list_macho "$APP")
}

# Drop any Mach-O still referencing /opt/homebrew after vendor + rewrite have
# run -- typically Qt plugins (e.g. PySide6's sqldrivers) pre-linked against
# Homebrew kegs not installed on this host. dyld would fail to load them
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
            | grep -E '^/opt/homebrew' || true)
        rpath_refs=$(binary_rpaths "$BIN" | grep -E '^/opt/homebrew' || true)
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
        echo "    WARN: pruned $count Mach-O file(s) with unresolvable /opt/homebrew deps" >&2
    fi
}

# PySide6 wheels ship a private copy of Qt under PySide6/Qt/lib. Its .abi3.so
# modules load that copy via @rpath; pilot's C++ binary loads Qt from
# Contents/Frameworks/. Two Qt instances in one process => duplicate Obj-C
# classes, separate global state, "QWidget: Must construct a QApplication"
# abort. Replace each duplicated nested framework with a relative symlink to
# the main bundle copy. Frameworks unique to PySide6 (QtCharts, ...) are left
# in place; their @rpath/QtCore... refs resolve through these symlinks too.
dedupe_pyside_qt() {
    local pyqt_lib count=0 nested name main rel
    pyqt_lib="$DEST_FW/Versions/$PY_VER/lib/python${PY_VER}/site-packages/PySide6/Qt/lib"
    [[ -d "$pyqt_lib" ]] || return 0
    # 9 levels: PySide6/Qt/lib up to Contents/Frameworks/.
    rel='../../../../../../../../..'
    for nested in "$pyqt_lib"/Qt*.framework; do
        [[ -d "$nested" && ! -L "$nested" ]] || continue
        name=$(basename "$nested" .framework)
        main="$FW_DIR/${name}.framework"
        [[ -d "$main" && ! -L "$main" ]] || continue
        rm -rf "$nested"
        ln -s "$rel/${name}.framework" "$nested"
        count=$((count+1))
    done
    if [[ $count -gt 0 ]]; then
        echo "    deduped $count PySide6/Qt/lib framework(s) into main Qt"
    fi
}

# Set PYTHONNOUSERSITE=1 in LSEnvironment so a Finder/open launch of pilot
# never adds ~/Library/Python/<X.Y>/lib/python/site-packages to sys.path,
# which on dev machines may carry a second PySide6 (and its Qt). Same
# dual-instance crash as dedupe_pyside_qt fixes for the in-bundle case.
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
            echo "    ERROR: failed to set LSEnvironment.PYTHONNOUSERSITE" >&2
            return 1
        }
    else
        # Dict may already exist; tolerate either way.
        /usr/libexec/PlistBuddy -c "Add :LSEnvironment dict" "$plist" \
            2>/dev/null || true
        /usr/libexec/PlistBuddy -c \
            "Add :LSEnvironment:PYTHONNOUSERSITE string 1" "$plist" || {
            echo "    ERROR: failed to add LSEnvironment.PYTHONNOUSERSITE" >&2
            return 1
        }
    fi
    echo "    Info.plist LSEnvironment.PYTHONNOUSERSITE = 1"
}

T_STEP=$SECONDS
echo "==> Vendoring Homebrew dependencies"
vendor_homebrew_deps
echo "==> Redirecting Homebrew load commands to bundled copies"
rewrite_load_commands
echo "==> Stripping /opt/homebrew rpaths"
strip_homebrew_rpaths
echo "==> Pruning Mach-O files with unresolvable /opt/homebrew deps"
prune_unfixable_machos
echo "==> Deduplicating PySide6's bundled Qt against the main bundle Qt"
dedupe_pyside_qt
echo "==> Setting PYTHONNOUSERSITE=1 in Info.plist"
inject_pythonnousersite
echo "    [Step 6 (Vendor Homebrew deps): $((SECONDS - T_STEP))s]"

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

# Re-seal each nested framework (Python.framework, Qt*.framework, ...).
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
echo "==> Ad-hoc re-signing all bundled Mach-O files"
sign_inner_machos
sign_helper_apps
sign_frameworks
echo "==> Ad-hoc re-signing the app bundle"
sign_main_app
echo "    [Step 7 (Ad-hoc codesign): $((SECONDS - T_STEP))s]"

echo ""
echo "Bundle complete: $APP"

# ---------------------------------------------------------------------------
# Step 8: Package DMG
# ---------------------------------------------------------------------------

T_STEP=$SECONDS
mkdir -p "$OUTPUT_DIR"
DMG="$OUTPUT_DIR/pilot.dmg"
echo "==> Creating $DMG from $APP"
hdiutil create -volname "solvcon Pilot" \
    -srcfolder "$APP" \
    -ov -format UDZO \
    "$DMG"
SIZE=$(du -sh "$DMG" | cut -f1)
echo "    [Step 8 (Package DMG): $((SECONDS - T_STEP))s]"

echo ""
echo "DMG complete: $DMG ($SIZE)"

# ---------------------------------------------------------------------------
# Step 9: Verify hermeticity
#
# Two complementary checks per artifact:
#   - Static scan of every Mach-O load command and LC_RPATH for any
#     surviving Homebrew prefix reference.
#   - Smoke launch under env -i with DYLD_PRINT_LIBRARIES=1 to catch
#     anything that resolves at runtime via PATH or dyld fallbacks.
# Repeated for both the .app and the .app inside the mounted DMG.
# ---------------------------------------------------------------------------

# A real launch loads ~900-1500 libraries (Qt + Python + dependents, yeah it's
# a lot, and why this script is still a prototype). Anything under this floor
# suggests the process was killed before dyld finished its initial pass and the
# trace tells us nothing.
# env -i drops every variable; we set only the minimum to let Cocoa initialise.
# pilot is a GUI app and never exits on its own, so we poll the trace at 1 Hz
# and SIGTERM as soon as dyld has emitted >= MIN_LOADS lines. Polling absorbs
# the multi-second kernel signature validation stall on a fresh codesign or DMG
# mount without paying a fixed wait on subsequent launches.
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

# Run a sandboxed launch and verify the dyld trace: enough loads happened, none
# from host Homebrew prefixes.
smoke_check() {
    local bin="$1" label="${2:-}" total brew_count
    [[ -n "$label" ]] && label=" ($label)"
    echo "==> Smoke launch${label} under sandboxed env (DYLD_PRINT_LIBRARIES=1)"
    launch_and_trace "$bin"
    total=$(grep -c '^dyld\[' "$TRACE" || true)
    brew_count=$(grep -E '^dyld\[' "$TRACE" | grep -E -c '/opt/homebrew|/usr/local' || true)
    if [[ $total -lt $MIN_LOADS ]]; then
        echo "ERROR: pilot loaded only $total libraries (expected >= $MIN_LOADS);" \
             "did it crash early?" >&2
        sed -n '1,40p' "$TRACE" >&2
        exit 1
    fi
    if [[ $brew_count -gt 0 ]]; then
        echo "ERROR: $brew_count of $total dyld loads came from host package prefixes:" >&2
        grep -E '^dyld\[' "$TRACE" | grep -E '/opt/homebrew|/usr/local' | head -10 \
            | sed 's/^/    /' >&2
        exit 1
    fi
    echo "    OK: $total libraries loaded, none from host package prefixes"
}

# Run Step 9 against both the .app and the .app inside the mounted DMG. TRACE
# and DMG_MOUNT are set on globals so the cleanup trap can tear them down on
# any exit path.
if [[ $SKIP_CHECK -eq 0 ]]; then
    T_STEP=$SECONDS
    TRACE=$(mktemp -t check-bundle-trace)

    echo ""
    echo "==> Checking $APP"
    echo "==> Static scan for host package path references"
    check_self_contained "$APP"
    smoke_check "$BINARY"
    echo ""
    echo "Bundle check passed: $APP"

    DMG_MOUNT=$(mktemp -d -t check-dmg)
    echo ""
    echo "==> Mounting $DMG at $DMG_MOUNT"
    hdiutil attach -nobrowse -readonly -mountpoint "$DMG_MOUNT" "$DMG" \
        >/dev/null

    DMG_APP=$(find "$DMG_MOUNT" -maxdepth 2 -name '*.app' -type d -print -quit)
    [[ -n "$DMG_APP" ]] || { echo "ERROR: no .app found in $DMG" >&2; exit 1; }
    echo "==> Found app: $DMG_APP"
    echo "==> Static scan (DMG) for host package path references"
    check_self_contained "$DMG_APP"
    DMG_BINARY="$DMG_APP/Contents/MacOS/$(basename "$DMG_APP" .app)"
    smoke_check "$DMG_BINARY" "DMG"
    echo ""
    echo "DMG check passed: $DMG"
    echo "    [Step 9 (Verify hermeticity): $((SECONDS - T_STEP))s]"
fi

echo ""
echo "Total elapsed: $((SECONDS - T_SCRIPT))s"

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
