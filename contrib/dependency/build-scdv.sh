#!/bin/bash
#
# Build solvcon's runtime dependencies from source into a user-space prefix,
# with no dependency on the devenv tool. This is a single self-contained
# cross-platform script: it holds the shared build logic and the
# platform-specific parts for every supported OS inline, so nothing else in the
# repository is sourced or called. The target platform is auto-detected from
# `uname -s` (Linux -> ubuntu, Darwin -> macos) and can be forced with
# SCDV_OS=ubuntu|macos.
#
# The platform-specific parts live in the per-OS `case` blocks below (search for
# 'case "${SCDV_OS}"'). Each block sets SCDV_OS_TAG and the plat_* hook
# functions this shared logic calls (see "Platform block contract" below), and
# carries its own notes (apt/brew prerequisites, libclang, etc.). The
# per-package build recipes are inlined translations of the scripts under
# devenv/scripts/build.d/ (zlib, openssl, sqlite, python, pybind11, cython,
# numpy, scipy, qt, pyside6).
#
# 4 sections (BASE, PYTHON, NUMPY, QT) are guarded by the corresponding
# SCDVBUILD_* environment variables.
#
# Before running this script, install the system prerequisites. The script
# never invokes the OS package manager itself; run it with --print-deps to see
# the exact commands, review them, and run them yourself.
#
# Usage:
#   ./build-scdv.sh
#       Builds everything (BASE + PYTHON + NUMPY + QT).  The default is "build
#       all" when no SCDVBUILD_* env var is set.  To limit to a single section,
#       set one of SCDVBUILD_BASE/PYTHON/NUMPY/QT=1 and leave the others unset.
#       In addition, SCDVBUILD_ALL=1 works the same as the default invocation.
#       Before any package is built the user is prompted with "Press Enter to
#       start the build, Ctrl-C to abort"; pass --no-confirm to skip.
#   ./build-scdv.sh --write-activate-only
#       Write only ${SCDV_BASE}/activate and exit.  Useful for refreshing the
#       activation script for an already-built scdv without triggering any
#       build section.
#   ./build-scdv.sh --skip PKG [--skip PKG ...]
#       Skip a package within whatever sections are otherwise selected.  PKG
#       can be one of: zlib openssl sqlite python pybind11 cython numpy scipy
#       qt pyside6.  The flag accepts a single name, a comma-separated list
#       ("--skip openssl,sqlite"), and may be repeated.  The --skip=PKG form
#       also works.
#   ./build-scdv.sh --no-confirm
#       Skip the "Press Enter to start the build" prompt that fires after the
#       startup echo block.  Use in non-interactive runs (CI, scripts).
#   ./build-scdv.sh --print-prefix
#       Print SCDV_PREFIX (the path prefix that SCDV_BASE is derived from)
#       to stdout and exit.  Nothing else is printed and no directories are
#       created, so this is safe to capture: PREFIX=$(./script --print-prefix).
#   ./build-scdv.sh --print-deps
#       Print the system prerequisite commands for the detected platform (apt
#       on Ubuntu, brew on macOS) to stdout and exit.  Nothing is built and no
#       directories are created.  The script never runs the package manager
#       itself; copy the output, review it, and run it.  --print-apt is a
#       backward-compatible alias.
#
# Overridable variables (search below for defaults):
#   Platform:
#     SCDV_OS: Target platform, "ubuntu" or "macos".  Auto-detected from
#       `uname -s`; set explicitly to override.
#
#   Package versions:
#     PYTHON_VERSION: CPython release tag.
#     QT_MAJOR_VER: Qt major.minor version.
#     QT_SUB_VER: Qt patch version.
#     PYSIDE_VERSION: Qt for Python (pyside-setup) source release version.
#
#   Build settings:
#     SCDV_NP: Parallel build jobs.
#     SCDV_PREFIX: Path prefix to SCDV_BASE.
#     SCDV_BASE: Base directory holding the install prefix, including Python
#       and Qt version.
#     SCDV_DLDIR: Directory for downloaded tarballs (real dir or symlink,
#       depending on SCDV_SHARED_DLDIR).
#     SCDV_SHARED_DLDIR: Shared cache for downloaded tarballs across solvcon
#       development environments (scdvs).
#
# Platform block contract. Each per-OS `case` block below sets the variable
# SCDV_OS_TAG (the token baked into the default prefix, e.g. ubuntu2404) and
# defines these hook functions, which the shared logic calls:
#   plat_init                 One-time platform setup (e.g. detect BREW_PREFIX).
#   plat_nproc                Echo the default parallel-job count.
#   plat_print_deps           Print the OS package-manager prerequisite lines.
#   plat_startup_echo         Echo any extra platform lines in the startup block.
#   plat_write_activate TGT   Write the activation script to path TGT.
#   plat_md5 FILE             Echo the md5 hash of FILE.
#   plat_python_env           Export CPPFLAGS/LDFLAGS for the CPython build.
#   plat_numpy_install        Run the numpy `pip install .` (BLAS choice).
#   plat_scipy_install        Run the scipy `pip install .` (BLAS choice).
#   plat_numpy_run            Env-prep + `scdv_time build_numpy numpy`.
#   plat_qt_env_strip         Strip a stray Qt from the loader/PATH env.
#   plat_qt_extra_cfg         Set array PLAT_QT_CFG with extra Qt cmake args.
#   plat_qt_libclang_setup    Set LLVM_INSTALL_DIR for shiboken.
#   plat_pyside6_cmake_opt    Set array PLAT_PYSIDE_CMAKE_OPT for setup.py.

set -e
# Without pipefail, exit codes are lost through the `tee` pipe in with_log,
# so a failed build step would otherwise be silently ignored.
set -o pipefail

# Detect the target platform.
SCDV_OS=${SCDV_OS:-}
if [ -z "${SCDV_OS}" ] ; then
  case "$(uname -s)" in
    Linux)
      SCDV_OS=ubuntu
      # `uname -s` reports only "Linux", not the distribution, so every Linux
      # host defaults to the Ubuntu 24.04 block (apt package names,
      # /usr/lib/llvm-22, x86_64 openblas paths). Warn when /etc/os-release
      # does not look like Ubuntu 24.04 so the assumption is visible; set
      # SCDV_OS explicitly to silence this. The block is still used (as the
      # per-OS Ubuntu script always did), just no longer silently.
      if [ -r /etc/os-release ] ; then
        _scdv_osrel=$(set +e ; . /etc/os-release 2>/dev/null ; \
                      printf '%s:%s' "${ID:-}" "${VERSION_ID:-}")
        if [ "${_scdv_osrel}" != "ubuntu:24.04" ] ; then
          echo "warning: assuming the Ubuntu 24.04 build block on" \
               "$(uname -sr) (/etc/os-release '${_scdv_osrel}'); set" \
               "SCDV_OS=ubuntu|macos to override." >&2
        fi
        unset _scdv_osrel
      fi
      ;;
    Darwin) SCDV_OS=macos ;;
    *) echo "unsupported OS '$(uname -s)'; set SCDV_OS=ubuntu|macos" >&2
       exit 1 ;;
  esac
fi

# Define the platform-specific parts inline (nothing is sourced).  Each branch
# sets SCDV_OS_TAG and the plat_* hook functions; only the selected platform's
# block is defined.
case "${SCDV_OS}" in
  ubuntu)

#
# Ubuntu 24.04 platform block: sets SCDV_OS_TAG and the plat_* hook functions
# the shared build logic dispatches to.  Defined inline (only when
# SCDV_OS=ubuntu); see the "Platform block contract" in the script header.
#
# Before running the build, install the apt prerequisites. plat_print_deps
# prints the exact commands; copy and run them yourself (the build never
# invokes apt). LLVM_INSTALL_DIR defaults to /usr/lib/llvm-22 (the apt
# llvm-22-dev path); override if needed.
#
# LLVM 22 is the libclang shiboken parses Qt headers with on Ubuntu 24.04 with
# GCC 16: clang-18 is too old (its parser chokes on libstdc++-16).  Override
# LLVM_INSTALL_DIR for another.  The patchelf apt package is the same binary
# the patchelf PyPI wheel ships; either works, the build installs the wheel
# anyway as a safety net.  libreadline is intentionally not listed: Python is
# built with --with-readline=editline, which uses libedit (libedit-dev).
# libmpdec is not packaged on Ubuntu 24.04; Python is built with
# --with-system-libmpdec=no to use its bundled copy.

# Token baked into the default prefix path (kept per-OS so existing installs
# keep resolving).
SCDV_OS_TAG=ubuntu2404

plat_init() {
  : # No Ubuntu-specific one-time setup.
}

plat_nproc() {
  nproc
}

plat_startup_echo() {
  : # No extra Ubuntu startup lines.
}

# md5 hash of a file. Linux md5sum prints "hash  filename"; cut the hash off.
plat_md5() {
  md5sum "$1" | cut -d ' ' -f 1
}

scdv_apt_base_cmd() {
  # Print the apt command for the BASE/PYTHON/NUMPY sections. The build never
  # runs apt itself; copy the output, review it, and run it. doxygen is the
  # system half of the documentation C++ API path (see doc/README.md).
  cat <<'EOF'
sudo apt install -y \
  build-essential gcc g++ make cmake ninja-build pkg-config \
  git curl xz-utils gfortran libopenblas-dev \
  libffi-dev libbz2-dev liblzma-dev libgdbm-dev \
  libncurses-dev uuid-dev tk-dev libedit-dev libexpat1-dev \
  doxygen
EOF
}

scdv_apt_latex_cmd() {
  # Print the apt command for the LaTeX toolchain the documentation needs.
  # This is required by the ordinary HTML build (make html in doc/), not just
  # a PDF: math is rendered client-side by MathJax (no TeX), but the pstake
  # Sphinx extension (doc/ext/pstake.py) turns the ".. pstake::" PSTricks
  # figures into PNG at build time by shelling out to latex -> dvips (EPS) ->
  # ImageMagick convert (EPS -> PNG; Ghostscript is the fallback and also
  # convert's EPS delegate).  pstake's TeX template needs pst-all/pst-3dplot/
  # pst-eps/pst-coil/pst-bar/multido (texlive-pstricks), fancyvrb
  # (texlive-latex-recommended), and optionally cmbright
  # (texlive-fonts-extra); latex/dvips come from texlive-latex-base.
  #
  # The package set below is deliberately broad: texlive-latex-extra and
  # texlive-pictures (TikZ/PGF and the pst-node family) catch any package a
  # figure pulls in beyond the core pst-* styles, texlive-plain-generic
  # covers generic (non-LaTeX) macros, and texlive-extra-utils plus
  # texlive-font-utils supply helper binaries (epstopdf, dvips/dvipdf
  # wrappers, font tools) used along the dvips/EPS path.
  #
  # Gotcha: Ubuntu's ImageMagick ships /etc/ImageMagick-6/policy.xml with the
  # EPS/PS coders disabled by default, so convert fails on the generated .eps.
  # The pstake input is locally built and trusted, so comment out the
  # rights="none" lines for "EPS" and "PS" there (or rely on the Ghostscript
  # fallback by not installing imagemagick).
  cat <<'EOF'
sudo apt install -y \
  texlive-latex-base texlive-latex-recommended texlive-latex-extra \
  texlive-pstricks texlive-pictures texlive-plain-generic \
  texlive-fonts-recommended texlive-fonts-extra \
  texlive-extra-utils texlive-font-utils \
  ghostscript imagemagick
EOF
}

scdv_apt_qt_cmd() {
  # Print the apt command for the QT section (Qt + pyside6 build deps).
  # The X11/XCB -dev packages are the full set Qt's xcb platform plugin
  # needs at configure time. Without all of them Qt silently builds without
  # the xcb QPA plugin, leaving only offscreen/minimal, so a real (or xvfb)
  # X display cannot be used. plat_qt_extra_cfg force-enables FEATURE_xcb so a
  # missing dependency fails the configure loudly instead of dropping the
  # plugin. CI installs the runtime counterparts of these -dev packages in
  # .github/actions/setup_linux/action.yml; keep the two sets in sync.
  cat <<'EOF'
sudo apt install -y \
  llvm-22-dev clang-22 libclang-22-dev patchelf \
  libxkbcommon-dev libxkbcommon-x11-dev libfontconfig1-dev \
  libfreetype-dev libdbus-1-dev libgl1-mesa-dev libglu1-mesa-dev \
  libx11-dev libx11-xcb-dev libxext-dev libxfixes-dev libxi-dev \
  libxrender-dev libxcb1-dev libxcb-glx0-dev libxcb-cursor-dev \
  libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev \
  libxcb-randr0-dev libxcb-render0-dev libxcb-render-util0-dev \
  libxcb-shape0-dev libxcb-shm0-dev libxcb-sync-dev libxcb-util-dev \
  libxcb-xfixes0-dev libxcb-xinerama0-dev libxcb-xkb-dev
EOF
}

plat_print_deps() {
  # Ubuntu prints the apt base/QT/LaTeX sets.
  scdv_apt_base_cmd
  echo
  scdv_apt_qt_cmd
  echo
  scdv_apt_latex_cmd
}

plat_python_env() {
  export CPPFLAGS="-I${SCDV_USRDIR}/include ${CPPFLAGS:-}"
  local ldflags="-Wl,--no-as-needed"
  ldflags="${ldflags} -Wl,-rpath,${SCDV_USRDIR}/lib"
  ldflags="${ldflags} -Wl,-rpath,${SCDV_USRDIR}/lib64"
  ldflags="${ldflags} -L${SCDV_USRDIR}/lib"
  ldflags="${ldflags} -L${SCDV_USRDIR}/lib64"
  export LDFLAGS="${ldflags} ${LDFLAGS:-}"
}

plat_numpy_install() {
  # Runs in the unpacked numpy source directory (the caller has pushd'd).
  rm -f site.cfg
  # Point numpy at apt's libopenblas-dev. NPY_BLAS_ORDER is honored by the
  # meson build path; site.cfg is kept for older fallback paths.
  export NPY_BLAS_ORDER=openblas
  cat > site.cfg <<EOF
[openblas]
libraries = openblas
library_dirs = /usr/lib/x86_64-linux-gnu:${SCDV_USRDIR}/lib
include_dirs = /usr/include/x86_64-linux-gnu/openblas-pthread:/usr/include/openblas:${SCDV_USRDIR}/include
runtime_library_dirs = /usr/lib/x86_64-linux-gnu
EOF
  # GCC 16 trunk on Ubuntu 24.04 rejects the AVX512 `evex512` target attribute
  # that numpy 2.2.x uses, so cap cpu-dispatch at AVX2 instead of MAX.  Pass
  # via --config-settings so the pip-driven meson rebuild honors it (not just
  # an out-of-tree `spin build`).
  with_log install.log "${PY}" -m pip install . --no-build-isolation \
    --config-settings="setup-args=-Dcpu-dispatch=SSE3 SSSE3 SSE41 POPCNT SSE42 AVX F16C FMA3 AVX2"
}

plat_scipy_install() {
  # Runs in the unpacked scipy source directory (the caller has pushd'd).
  # Ubuntu's pybind11-dev (2.11) at /usr/include shadows our 2.13.6 and
  # rejects scipy's 3-arg PYBIND11_MODULE. -isystem orders our headers ahead
  # of /usr/include so scipy compiles against the right pybind11.
  local saved_cxxflags=${CXXFLAGS:-}
  export CXXFLAGS="-isystem ${SCDV_USRDIR}/include ${saved_cxxflags}"
  with_log install.log "${PY}" -m pip install . --no-build-isolation
  export CXXFLAGS="${saved_cxxflags}"
}

plat_numpy_run() {
  # NOFORTRAN mirrors the macOS reference; with apt's gfortran installed, scipy
  # will still pick gfortran up via meson's compiler detection.  Scope
  # NOFORTRAN to this one call.
  NOFORTRAN=1 scdv_time build_numpy numpy
}

plat_qt_env_strip() {
  # A pre-existing Qt (e.g. /home/$USER/var/Qt/6.x) leaking through
  # LD_LIBRARY_PATH or CMAKE_PREFIX_PATH would be loaded ahead of the freshly
  # built Qt tools.  Strip them, plus any /var/Qt/ entries on PATH.
  unset LD_LIBRARY_PATH CMAKE_PREFIX_PATH QT_ROOT QT_VER
  PATH=$(printf '%s' "${PATH}" | tr ':' '\n' | grep -v '/var/Qt/' | paste -sd:)
}

plat_qt_extra_cfg() {
  # Force the xcb (X11) platform plugin on. Qt otherwise silently drops it when
  # an XCB dev dependency is missing, leaving only the offscreen and minimal
  # QPA plugins and making a real (or xvfb) X display unusable. Force-enabling
  # turns a missing dependency into a configure-time failure here instead of a
  # runtime "Could not find the Qt platform plugin xcb". xcb_xlib is the
  # Xlib/XCB interop the plugin relies on. The required -dev packages are
  # listed in scdv_apt_qt_cmd above.
  PLAT_QT_CFG=("-DFEATURE_xcb=ON" "-DFEATURE_xcb_xlib=ON")
}

plat_qt_libclang_setup() {
  # LLVM_INSTALL_DIR points at the libclang shiboken should use.  Defaults to
  # the apt path for llvm-22, the newest stable and supported by PySide 6.11.1
  # on Ubuntu 24.04 (see the header comment); override if your llvm-22-dev
  # install lives elsewhere.
  LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR:-/usr/lib/llvm-22}
  if [ ! -d "${LLVM_INSTALL_DIR}" ] ; then
    echo "LLVM_INSTALL_DIR='${LLVM_INSTALL_DIR}' does not exist;" \
         "install llvm-22-dev or set LLVM_INSTALL_DIR explicitly" >&2
    exit 1
  fi
}

plat_pyside6_cmake_opt() {
  # Ubuntu needs no extra setup.py cmake options.
  PLAT_PYSIDE_CMAKE_OPT=()
}

plat_write_activate() {
  # Write the activation script to $1. Self-locating (reads its own path at
  # source time) so the scdv directory can be moved without rewriting the file.
  # Sourcing it prepends our bin to PATH and our lib to LD_LIBRARY_PATH, strips
  # any leaked /var/Qt/ entries from both, and defines scdv_deactivate to
  # restore the original environment.
  local target=$1
  cat > "${target}" <<'ACTIVATE_EOF'
# shellcheck shell=bash
# Activate this SCDV: source this file (do not execute).
#
#   $ source <path-to-this-file>/activate
#   $ scdv_deactivate   # to restore the original environment
#
# Compatible with bash and zsh.

if [ -n "${SCDV_BASE:-}" ] ; then
  echo "SCDV '${SCDV_BASE##*/}' is already active." \
       "Run scdv_deactivate first." >&2
  return 1 2>/dev/null || exit 1
fi

# Resolve the directory this file lives in SCDV_BASE.
if [ -n "${BASH_VERSION:-}" ] ; then
  _scdv_self=${BASH_SOURCE[0]}
elif [ -n "${ZSH_VERSION:-}" ] ; then
  _scdv_self=${(%):-%x}
else
  _scdv_self=$0
fi
SCDV_BASE=$(cd "$(dirname "${_scdv_self}")" && pwd)
unset _scdv_self
export SCDV_BASE
export SCDV_USRDIR=${SCDV_BASE}/usr

# Snapshot the current environment so scdv_deactivate can restore it.
# These leading-underscore vars are private to the activation script.
export _SCDV_OLD_PATH=${PATH}
export _SCDV_OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}
export _SCDV_OLD_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH-}
export _SCDV_OLD_PS1=${PS1-}
export _SCDV_OLD_QT_QPA_PLATFORM=${QT_QPA_PLATFORM-}
export _SCDV_HAD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH+1}
export _SCDV_HAD_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH+1}
export _SCDV_HAD_QT_QPA_PLATFORM=${QT_QPA_PLATFORM+1}

# A pre-existing Qt (e.g. /home/$USER/var/Qt/6.x) leaking through PATH or
# LD_LIBRARY_PATH will be loaded ahead of this scdv's freshly built Qt and
# break with "version `Qt_6.x' not found". Strip those.
PATH=$(printf '%s' "${PATH}" | tr ':' '\n' | grep -v '/var/Qt/' | paste -sd: -)
export PATH=${SCDV_USRDIR}/bin:${PATH}

LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH-}" \
  | tr ':' '\n' | grep -v '/var/Qt/' | paste -sd: -)
if [ -n "${LD_LIBRARY_PATH}" ] ; then
  export LD_LIBRARY_PATH=${SCDV_USRDIR}/lib:${LD_LIBRARY_PATH}
else
  export LD_LIBRARY_PATH=${SCDV_USRDIR}/lib
fi

if [ -n "${CMAKE_PREFIX_PATH-}" ] ; then
  export CMAKE_PREFIX_PATH=${SCDV_USRDIR}:${CMAKE_PREFIX_PATH}
else
  export CMAKE_PREFIX_PATH=${SCDV_USRDIR}
fi

# solvcon's _solvcon module constructs a QApplication during PyInit__modmesh;
# without a display or QT_QPA_PLATFORM Qt aborts in
# QGuiApplicationPrivate::createPlatformIntegration. Default to the offscreen
# platform when nothing is set and no display is available, so plain `import
# solvcon` and `make pytest` work in headless shells.  If a real display
# becomes available later the user can `export QT_QPA_PLATFORM=xcb` (or unset
# it) without re-sourcing.
if [ -z "${QT_QPA_PLATFORM-}" ] \
   && [ -z "${DISPLAY:-}" ] \
   && [ -z "${WAYLAND_DISPLAY:-}" ] ; then
  export QT_QPA_PLATFORM=offscreen
fi

# Mark the prompt so the active scdv is visible.
if [ -n "${BASH_VERSION:-}" ] || [ -n "${ZSH_VERSION:-}" ] ; then
  PS1="(${SCDV_BASE##*/}) ${PS1-}"
  export PS1
fi

scdv_deactivate() {
  if [ -z "${SCDV_BASE:-}" ] ; then
    echo "No active SCDV." >&2
    return 1
  fi
  export PATH=${_SCDV_OLD_PATH}
  if [ -n "${_SCDV_HAD_LD_LIBRARY_PATH:-}" ] ; then
    export LD_LIBRARY_PATH=${_SCDV_OLD_LD_LIBRARY_PATH}
  else
    unset LD_LIBRARY_PATH
  fi
  if [ -n "${_SCDV_HAD_CMAKE_PREFIX_PATH:-}" ] ; then
    export CMAKE_PREFIX_PATH=${_SCDV_OLD_CMAKE_PREFIX_PATH}
  else
    unset CMAKE_PREFIX_PATH
  fi
  if [ -n "${_SCDV_HAD_QT_QPA_PLATFORM:-}" ] ; then
    export QT_QPA_PLATFORM=${_SCDV_OLD_QT_QPA_PLATFORM}
  else
    unset QT_QPA_PLATFORM
  fi
  PS1=${_SCDV_OLD_PS1}
  export PS1
  unset _SCDV_OLD_PATH _SCDV_OLD_LD_LIBRARY_PATH \
        _SCDV_OLD_CMAKE_PREFIX_PATH _SCDV_OLD_PS1 \
        _SCDV_OLD_QT_QPA_PLATFORM \
        _SCDV_HAD_LD_LIBRARY_PATH _SCDV_HAD_CMAKE_PREFIX_PATH \
        _SCDV_HAD_QT_QPA_PLATFORM
  unset SCDV_BASE SCDV_USRDIR
  unset -f scdv_deactivate
}
ACTIVATE_EOF
  # Do not set the execution bit of "${target}"; it should be sourced.
  echo "wrote activation script: ${target}"
}

    ;;
  macos)

#
# macOS 26 platform block: sets SCDV_OS_TAG and the plat_* hook functions the
# shared build logic dispatches to.  Defined inline (only when SCDV_OS=macos);
# see the "Platform block contract" in the script header.
#
# Before running the build, install the (Home)brew prerequisites.
# plat_print_deps prints the exact command; copy and run it yourself (the build
# never invokes brew).
#
# shiboken (the PySide6 binding generator) needs libclang, which Apple's
# Command Line Tools clang does not ship.  Rather than depend on a Homebrew
# llvm, the QT section downloads Qt's own prebuilt libclang into the scdv tree
# (user-space, no system install) -- see fetch_libclang.  LIBCLANG_VERSION is
# pinned to 21.x: unlike the Ubuntu block (which uses LLVM 22 against glibc's
# headers), Qt's prebuilt libclang 22.x segfaults shiboken while parsing the Qt
# headers on macOS, so 21.x stays the macOS sweet spot.  Set LLVM_INSTALL_DIR
# to use an existing libclang (e.g. a brew llvm) instead of fetching.
#
# Apple Silicon (arm64) is the assumed default.  The build also runs on Intel
# Macs, but solvcon does not plan to support Intel Macs.
#
# Python is built --enable-shared (matching the Ubuntu block) into
# ${SCDV_USRDIR}, rather than --enable-framework.  Python.framework support
# will be added in the future for bundling
# (contrib/bundle/bundle-with-homebrew.sh).
#
# libreadline is intentionally not listed: Python is built with
# --with-readline=editline, which uses macOS's built-in libedit.  libmpdec is
# not packaged via brew by default; Python is built with
# --with-system-libmpdec=no to use its bundled copy.  tk-dev / gdbm are not
# included in the brew prereqs: tkinter and dbm.gnu are not load-bearing for
# solvcon, and including them would force every user to install tcl-tk and gdbm
# even when they will never import them.  xcb / X11 prereqs do not apply: Qt on
# macOS uses the cocoa platform plugin, not xcb.

# Token baked into the default prefix path (kept per-OS so existing installs
# keep resolving).
SCDV_OS_TAG=macos26

# Qt prebuilt libclang version for shiboken (see fetch_libclang).  Pinned to
# 21.x on macOS: Qt's prebuilt libclang 22.1.2 segfaults shiboken while parsing
# the Qt headers, while libclang <=18 is too old for the macOS 26 SDK's libc++
# headers.
LIBCLANG_VERSION=${LIBCLANG_VERSION:-21.1.2}

plat_init() {
  # Detect the Homebrew prefix once so later hooks can reuse it.  brew is
  # keg-only for several packages we depend on (openblas, llvm, gcc), so we
  # build the absolute keg paths from BREW_PREFIX rather than relying on PATH.
  if command -v brew >/dev/null 2>&1 ; then
    BREW_PREFIX=$(brew --prefix 2>/dev/null || true)
  fi
  if [ -z "${BREW_PREFIX:-}" ] ; then
    if [ -d /opt/homebrew ] ; then
      BREW_PREFIX=/opt/homebrew
    elif [ -d /usr/local/Homebrew ] || [ -x /usr/local/bin/brew ] ; then
      BREW_PREFIX=/usr/local
    else
      BREW_PREFIX=
    fi
  fi
  export BREW_PREFIX
}

plat_nproc() {
  sysctl -n hw.ncpu
}

plat_startup_echo() {
  echo "BREW_PREFIX=${BREW_PREFIX:-(none)}"
}

# md5 hash of a file. macOS md5 -q prints just the hash (no filename column).
plat_md5() {
  md5 -q "$1"
}

scdv_brew_base_cmd() {
  # Print the brew command for the BASE/PYTHON/NUMPY sections.  The build never
  # runs brew itself; copy the output, review it, and run it.  gcc is included
  # for gfortran (scipy's Fortran sources); openblas is included as an optional
  # fallback BLAS even though numpy/scipy default to Apple's Accelerate
  # framework on macOS.  xz is included for Python's lzma module (Apple's
  # Command Line Tools do not ship liblzma).  doxygen is the system half of the
  # documentation C++ API path (see doc/README.md).  The QT section needs no
  # brew packages beyond this base set: Qt is built with Apple clang, and
  # shiboken's libclang is fetched from Qt's prebuilt packages (see
  # fetch_libclang), not Homebrew.
  cat <<'EOF'
brew install \
  cmake ninja pkg-config xz gcc openblas doxygen
EOF
}

plat_print_deps() {
  # macOS prints the brew base set (the QT and LaTeX toolchains need no extra
  # brew packages).
  scdv_brew_base_cmd
}

plat_python_env() {
  # Pick up our own zlib/openssl/sqlite headers; pull in xz from brew so the
  # lzma extension builds (Apple CLT has no liblzma).
  local cppflags="-I${SCDV_USRDIR}/include"
  if [ -n "${BREW_PREFIX}" ] && [ -d "${BREW_PREFIX}/opt/xz/include" ] ; then
    cppflags="${cppflags} -I${BREW_PREFIX}/opt/xz/include"
  fi
  export CPPFLAGS="${cppflags} ${CPPFLAGS:-}"
  # macOS ld64 / Apple linker: drop GNU-only --no-as-needed; use
  # -headerpad_max_install_names so install_name_tool has room to rewrite paths
  # later (a no-op when we don't, harmless when we do).
  local ldflags="-Wl,-headerpad_max_install_names"
  ldflags="${ldflags} -Wl,-rpath,${SCDV_USRDIR}/lib"
  ldflags="${ldflags} -L${SCDV_USRDIR}/lib"
  if [ -n "${BREW_PREFIX}" ] && [ -d "${BREW_PREFIX}/opt/xz/lib" ] ; then
    ldflags="${ldflags} -L${BREW_PREFIX}/opt/xz/lib"
  fi
  export LDFLAGS="${ldflags} ${LDFLAGS:-}"
}

plat_numpy_install() {
  # Runs in the unpacked numpy source directory (the caller has pushd'd).
  # Prefer macOS's built-in Accelerate framework over openblas; meson's
  # blas-order/lapack-order accept a comma-separated fallback list, so the
  # build still succeeds on Macs without Accelerate (very rare) by falling back
  # to brew openblas.  The AVX2 cpu-dispatch cap that the Ubuntu block needs
  # is GCC-16-specific; Apple clang on macOS handles the default
  # cpu-dispatch=MAX fine.
  with_log install.log "${PY}" -m pip install . --no-build-isolation \
    --config-settings="setup-args=-Dblas-order=accelerate,openblas" \
    --config-settings="setup-args=-Dlapack-order=accelerate,openblas"
}

plat_scipy_install() {
  # Runs in the unpacked scipy source directory (the caller has pushd'd).
  # No -isystem CXXFLAGS workaround here: the Ubuntu block needs it to outrank
  # apt's stale pybind11-dev at /usr/include, but macOS does not ship a system
  # pybind11, so our header at ${SCDV_USRDIR}/include wins automatically. scipy
  # autodetects Accelerate via meson on macOS, same as numpy above.
  with_log install.log "${PY}" -m pip install . --no-build-isolation \
    --config-settings="setup-args=-Dblas=accelerate" \
    --config-settings="setup-args=-Dlapack=accelerate"
}

plat_numpy_run() {
  # Surface brew's gfortran (installed by `brew install gcc`) so meson picks it
  # up for scipy's Fortran sources.  On Apple Silicon the binary is
  # /opt/homebrew/opt/gcc/bin/gfortran; on Intel /usr/local/opt/gcc/bin.  The
  # PATH export persists into build_scipy, which is intended.
  if [ -n "${BREW_PREFIX}" ] && [ -d "${BREW_PREFIX}/opt/gcc/bin" ] ; then
    export PATH="${BREW_PREFIX}/opt/gcc/bin:${PATH}"
  fi
  scdv_time build_numpy numpy
}

plat_qt_env_strip() {
  # A pre-existing brew Qt may leak through DYLD_LIBRARY_PATH /
  # DYLD_FRAMEWORK_PATH / CMAKE_PREFIX_PATH and cause the freshly built Qt tools
  # (rcc/moc/...) to load the older libQt6Core at runtime.  Strip them out,
  # plus any /opt/homebrew/Cellar/qt or /opt/homebrew/opt/qt entries on PATH.
  unset DYLD_LIBRARY_PATH DYLD_FRAMEWORK_PATH CMAKE_PREFIX_PATH QT_ROOT QT_VER
  PATH=$(printf '%s' "${PATH}" | tr ':' '\n' \
    | grep -vE '/(Cellar|opt)/qt(@|/|$)' | paste -sd: -)
}

plat_qt_extra_cfg() {
  # Skip Qt's build-time Xcode-version check: it runs `xcrun xcodebuild
  # -version` and aborts ("Can't determine Xcode version.  Is Xcode
  # installed?") on a CLT-only machine -- the common solvcon dev setup -- where
  # xcodebuild (shipped only with the full Xcode.app) is absent.  Qt builds
  # fine with Apple clang from the Command Line Tools; only the version probe
  # needs disabling.  Qt 6.11 also supports the macOS 26 SDK
  # (QT_SUPPORTED_MAX_MACOS_SDK_VERSION=26), so no SDK-max-version override is
  # needed any more either.
  PLAT_QT_CFG=("-DQT_NO_XCODE_MIN_VERSION_CHECK=ON")
}

fetch_libclang() {
  # Download Qt's prebuilt libclang and lay out a minimal LLVM_INSTALL_DIR for
  # shiboken under ${SCDV_SRCDIR}/libclang.  The full archive is ~4 GB because
  # it bundles the entire LLVM/clang command-line toolset under bin/ (clangd,
  # llc, clang-tidy, ...) that shiboken never invokes; it loads libclang.dylib
  # directly.  So extract only lib/ (the dylib, the LLVM/Clang CMake package
  # config, static libs, and clang resource headers) and include/ -- ~1 GB.
  # macOS bsdtar (libarchive + liblzma) reads 7-Zip natively, so no 7z/brew
  # tool is needed.
  #
  # LLVMExports.cmake / ClangTargets.cmake assert that every exported target's
  # file exists, including the bin/ tools we skipped, so find_package(Clang)
  # would FATAL_ERROR.  Drop empty placeholders for exactly those referenced
  # paths; shiboken never executes them, it only needs the existence check to
  # pass and the real libclang.dylib in lib/.
  local fn=libclang-release_${LIBCLANG_VERSION}-based-macos-universal.7z
  download_md5 "${fn}" \
    "https://download.qt.io/development_releases/prebuilt/libclang/${fn}" \
    2a332ef2f3e6f87a68a19d3d2698e7ae
  local dest=${SCDV_SRCDIR}/libclang
  rm -rf "${dest}"
  pushd "${SCDV_SRCDIR}" > /dev/null
    tar xf "${SCDV_DLDIR}/${fn}" libclang/lib libclang/include
  popd > /dev/null
  pushd "${dest}" > /dev/null
    grep -rhoE '[$][{]_IMPORT_PREFIX[}]/[A-Za-z0-9._/-]+' lib/cmake 2>/dev/null \
      | sed 's#[$][{]_IMPORT_PREFIX[}]/##' | sort -u | while read -r rel ; do
        [ -e "${rel}" ] && continue
        mkdir -p "$(dirname "${rel}")"
        : > "${rel}"
        chmod +x "${rel}"
      done
  popd > /dev/null
  if [ ! -f "${dest}/lib/libclang.dylib" ] ; then
    echo "fetch_libclang: ${dest}/lib/libclang.dylib missing after extract" >&2
    exit 1
  fi
}

plat_qt_libclang_setup() {
  # By default fetch Qt's prebuilt libclang into the scdv tree (user-space, no
  # Homebrew) -- see fetch_libclang and the header comment for the version
  # rationale.  Set LLVM_INSTALL_DIR to point at an existing libclang (e.g. a
  # brew llvm) to skip the download.
  if [ -z "${LLVM_INSTALL_DIR:-}" ] ; then
    scdv_time fetch_libclang libclang
    LLVM_INSTALL_DIR=${SCDV_SRCDIR}/libclang
  fi
  if [ ! -d "${LLVM_INSTALL_DIR}" ] ; then
    echo "LLVM_INSTALL_DIR='${LLVM_INSTALL_DIR}' does not exist;" \
         "unset it to fetch Qt's prebuilt libclang, or point it at a libclang" \
         "install." >&2
    exit 1
  fi
}

plat_pyside6_cmake_opt() {
  # CMake 3.27+ auto-adds the Homebrew prefix to its system search path on
  # macOS.  With brew's `qt` and `pyside` formulae installed (the solvcon macOS
  # VM setup does `brew install ... qt pyside`), pyside-setup's find_package
  # calls resolve Qt6 / Shiboken6 / PySide6 to brew's copies instead of the
  # ones this scdv just built.  That mixes incompatible Qt versions (brew's
  # Qt6UiTools pulls brew Qt6CoreTools, whose newer machinery aborts with
  # Unknown CMake command "_qt_internal_should_include_targets") and trips over
  # brew's broken ${BREW_PREFIX}/typesystems.  The clean global fix is
  # -DCMAKE_IGNORE_PREFIX_PATH=${BREW_PREFIX}; scdv provides Qt, shiboken, and
  # Python directly and libclang via LLVM_INSTALL_DIR, so nothing the pyside6
  # build needs lives under the ignored brew prefix.  setup.py exposes no way
  # to forward extra cmake -D args (and its --cmake-toolchain-file flag forces
  # cross-compile mode), so point it at a wrapper cmake via --cmake= that
  # injects the flag on configure invocations only -- build/install and
  # cache-inspection (-L/-N) calls are passed through untouched.  Skipped
  # entirely when no Homebrew is present (nothing to hide).
  PLAT_PYSIDE_CMAKE_OPT=()
  if [ -n "${BREW_PREFIX}" ] ; then
    local realcmake cmwrap
    realcmake=$(command -v cmake)
    cmwrap=${SCDV_SRCDIR}/cmake-no-brew
    mkdir -p "${cmwrap}"
    cat > "${cmwrap}/cmake" <<WRAP
#!/bin/sh
# Wrapper that hides the Homebrew prefix from CMake package discovery during
# the pyside6 configure, so brew's Qt/PySide do not shadow this scdv's. Only
# configure runs get the flag; --build / --install / -L / -N pass through.
# Generated by build-scdv.sh.
for a in "\$@" ; do
  case "\$a" in
    --build|--install|--open|-E|-P|-N|-L|--version|--find-package)
      exec "${realcmake}" "\$@" ;;
  esac
done
exec "${realcmake}" -DCMAKE_IGNORE_PREFIX_PATH="${BREW_PREFIX}" "\$@"
WRAP
    chmod +x "${cmwrap}/cmake"
    PLAT_PYSIDE_CMAKE_OPT=("--cmake=${cmwrap}/cmake")
  fi
}

plat_write_activate() {
  # Write the activation script to $1. Self-locating (reads its own path at
  # source time) so the scdv directory can be moved without rewriting the file.
  # Sourcing it prepends our bin to PATH and our lib to DYLD_LIBRARY_PATH,
  # strips any leaked Homebrew Qt entries from both, and defines scdv_deactivate
  # to restore the original environment.
  local target=$1
  cat > "${target}" <<'ACTIVATE_EOF'
# shellcheck shell=bash
# Activate this SCDV: source this file (do not execute).
#
#   $ source <path-to-this-file>/activate
#   $ scdv_deactivate   # to restore the original environment
#
# Compatible with bash and zsh.

if [ -n "${SCDV_BASE:-}" ] ; then
  echo "SCDV '${SCDV_BASE##*/}' is already active." \
       "Run scdv_deactivate first." >&2
  return 1 2>/dev/null || exit 1
fi

# Resolve the directory this file lives in SCDV_BASE.
if [ -n "${BASH_VERSION:-}" ] ; then
  _scdv_self=${BASH_SOURCE[0]}
elif [ -n "${ZSH_VERSION:-}" ] ; then
  _scdv_self=${(%):-%x}
else
  _scdv_self=$0
fi
SCDV_BASE=$(cd "$(dirname "${_scdv_self}")" && pwd)
unset _scdv_self
export SCDV_BASE
export SCDV_USRDIR=${SCDV_BASE}/usr

# Snapshot the current environment so scdv_deactivate can restore it.
# These leading-underscore vars are private to the activation script.
export _SCDV_OLD_PATH=${PATH}
export _SCDV_OLD_DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH-}
export _SCDV_OLD_DYLD_FRAMEWORK_PATH=${DYLD_FRAMEWORK_PATH-}
export _SCDV_OLD_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH-}
export _SCDV_OLD_PS1=${PS1-}
export _SCDV_HAD_DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH+1}
export _SCDV_HAD_DYLD_FRAMEWORK_PATH=${DYLD_FRAMEWORK_PATH+1}
export _SCDV_HAD_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH+1}

# A pre-existing Homebrew Qt (kegs under /opt/homebrew/Cellar/qt or
# /opt/homebrew/opt/qt, or the Intel-Mac equivalents under /usr/local) can
# leak through PATH, DYLD_LIBRARY_PATH, or CMAKE_PREFIX_PATH and load ahead
# of this scdv's freshly built Qt. Strip those.
PATH=$(printf '%s' "${PATH}" | tr ':' '\n' \
  | grep -vE '/(Cellar|opt)/qt(@|/|$)' | paste -sd: -)
export PATH=${SCDV_USRDIR}/bin:${PATH}

DYLD_LIBRARY_PATH=$(printf '%s' "${DYLD_LIBRARY_PATH-}" \
  | tr ':' '\n' | grep -vE '/(Cellar|opt)/qt(@|/|$)' | paste -sd: -)
if [ -n "${DYLD_LIBRARY_PATH}" ] ; then
  export DYLD_LIBRARY_PATH=${SCDV_USRDIR}/lib:${DYLD_LIBRARY_PATH}
else
  export DYLD_LIBRARY_PATH=${SCDV_USRDIR}/lib
fi

# DYLD_FRAMEWORK_PATH covers any Qt/Python pieces that resolve as macOS
# frameworks rather than plain dylibs (Qt itself uses framework layout when
# built with the macOS generator, regardless of how we configured Python).
DYLD_FRAMEWORK_PATH=$(printf '%s' "${DYLD_FRAMEWORK_PATH-}" \
  | tr ':' '\n' | grep -vE '/(Cellar|opt)/qt(@|/|$)' | paste -sd: -)
if [ -n "${DYLD_FRAMEWORK_PATH}" ] ; then
  export DYLD_FRAMEWORK_PATH=${SCDV_USRDIR}/lib:${DYLD_FRAMEWORK_PATH}
else
  export DYLD_FRAMEWORK_PATH=${SCDV_USRDIR}/lib
fi

if [ -n "${CMAKE_PREFIX_PATH-}" ] ; then
  export CMAKE_PREFIX_PATH=${SCDV_USRDIR}:${CMAKE_PREFIX_PATH}
else
  export CMAKE_PREFIX_PATH=${SCDV_USRDIR}
fi

# Note on macOS and DYLD_*: SIP strips DYLD_* when a shell spawns a system
# binary (anything under /usr/bin, /bin, /sbin).  The scdv binaries we build
# embed an rpath to ${SCDV_USRDIR}/lib at link time, so they keep loading our
# dylibs even when DYLD_LIBRARY_PATH is dropped.  The exports here are
# belt-and-suspenders for direct-invoke from this shell.

# Mark the prompt so the active scdv is visible.
if [ -n "${BASH_VERSION:-}" ] || [ -n "${ZSH_VERSION:-}" ] ; then
  PS1="(${SCDV_BASE##*/}) ${PS1-}"
  export PS1
fi

scdv_deactivate() {
  if [ -z "${SCDV_BASE:-}" ] ; then
    echo "No active SCDV." >&2
    return 1
  fi
  export PATH=${_SCDV_OLD_PATH}
  if [ -n "${_SCDV_HAD_DYLD_LIBRARY_PATH:-}" ] ; then
    export DYLD_LIBRARY_PATH=${_SCDV_OLD_DYLD_LIBRARY_PATH}
  else
    unset DYLD_LIBRARY_PATH
  fi
  if [ -n "${_SCDV_HAD_DYLD_FRAMEWORK_PATH:-}" ] ; then
    export DYLD_FRAMEWORK_PATH=${_SCDV_OLD_DYLD_FRAMEWORK_PATH}
  else
    unset DYLD_FRAMEWORK_PATH
  fi
  if [ -n "${_SCDV_HAD_CMAKE_PREFIX_PATH:-}" ] ; then
    export CMAKE_PREFIX_PATH=${_SCDV_OLD_CMAKE_PREFIX_PATH}
  else
    unset CMAKE_PREFIX_PATH
  fi
  PS1=${_SCDV_OLD_PS1}
  export PS1
  unset _SCDV_OLD_PATH _SCDV_OLD_DYLD_LIBRARY_PATH \
        _SCDV_OLD_DYLD_FRAMEWORK_PATH \
        _SCDV_OLD_CMAKE_PREFIX_PATH _SCDV_OLD_PS1 \
        _SCDV_HAD_DYLD_LIBRARY_PATH _SCDV_HAD_DYLD_FRAMEWORK_PATH \
        _SCDV_HAD_CMAKE_PREFIX_PATH
  unset SCDV_BASE SCDV_USRDIR
  unset -f scdv_deactivate
}
ACTIVATE_EOF
  # Do not set the execution bit of "${target}"; it should be sourced.
  echo "wrote activation script: ${target}"
}

    ;;
  *) echo "unknown SCDV_OS='${SCDV_OS}'; expected ubuntu or macos" >&2
     exit 1 ;;
esac

SCDV_WRITE_ACTIVATE_ONLY=0
SCDV_PRINT_PREFIX_ONLY=0
SCDV_PRINT_DEPS_ONLY=0
SCDV_NO_CONFIRM=0
SCDV_SKIP_LIST=""
SCDV_KNOWN_PKGS="zlib openssl sqlite python pybind11 cython numpy scipy qt pyside6"

scdv_add_skip() {
  # Accept "pkg" or "pkg1,pkg2,..."; warn on unknown names but accept.
  local raw=$1 name
  for name in $(echo "${raw}" | tr ',' ' ') ; do
    [ -z "${name}" ] && continue
    case " ${SCDV_KNOWN_PKGS} " in
      *" ${name} "*) ;;
      *) echo "warning: --skip '${name}' is not a known package (known: ${SCDV_KNOWN_PKGS})" >&2 ;;
    esac
    SCDV_SKIP_LIST="${SCDV_SKIP_LIST} ${name}"
  done
}

scdv_skip_p() {
  case " ${SCDV_SKIP_LIST} " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

while [ $# -gt 0 ] ; do
  case "$1" in
    --write-activate-only)
      SCDV_WRITE_ACTIVATE_ONLY=1
      ;;
    --print-prefix)
      SCDV_PRINT_PREFIX_ONLY=1
      ;;
    --print-deps|--print-apt)
      SCDV_PRINT_DEPS_ONLY=1
      ;;
    --no-confirm)
      SCDV_NO_CONFIRM=1
      ;;
    --skip)
      shift
      if [ $# -eq 0 ] ; then
        echo "--skip requires a package name" >&2
        exit 2
      fi
      scdv_add_skip "$1"
      ;;
    --skip=*)
      scdv_add_skip "${1#--skip=}"
      ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^#//' >&2
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

# --print-deps needs only plat_print_deps. Exit before
# plat_init / plat_nproc so SCDV_OS=<other> dry-runs work on this host (e.g.
# SCDV_OS=macos --print-deps on Linux must not call macOS sysctl).
if [ "${SCDV_PRINT_DEPS_ONLY}" = "1" ] ; then
  plat_print_deps
  exit 0
fi

# One-time platform setup (e.g. macOS BREW_PREFIX detection).
plat_init

# CPython release tag.
PYTHON_VERSION=${PYTHON_VERSION:-3.14.5}
# Qt major.minor version.
QT_MAJOR_VER=${QT_MAJOR_VER:-6.11}
# Qt patch version.
QT_SUB_VER=${QT_SUB_VER:-1}
# Qt for Python (pyside-setup) source release version.
PYSIDE_VERSION=${PYSIDE_VERSION:-${QT_MAJOR_VER}.${QT_SUB_VER}}

# Default to a full BASE+PYTHON+NUMPY+QT build when the caller has not selected
# a specific section. Setting any one of SCDVBUILD_* to "1" disables this
# default and runs only the explicitly selected sections.
if [ -z "${SCDVBUILD_ALL:-}${SCDVBUILD_BASE:-}${SCDVBUILD_PYTHON:-}${SCDVBUILD_NUMPY:-}${SCDVBUILD_QT:-}" ] ; then
  SCDVBUILD_ALL=1
fi

# Path prefix to SCDV_BASE.
SCDV_PREFIX=${SCDV_PREFIX:-${HOME}/var/scdv/${SCDV_OS_TAG}}

# --print-prefix is honored as soon as SCDV_PREFIX is resolved so the caller
# can capture the value without triggering any side effects (mkdir,
# activate-write, build). plat_nproc is deferred until after this exit for the
# same cross-OS dry-run reason as --print-deps above.
if [ "${SCDV_PRINT_PREFIX_ONLY}" = "1" ] ; then
  echo "SCDV_PREFIX=${SCDV_PREFIX}"
  exit 0
fi

# Parallel build jobs (platform-specific; only needed once we build).
SCDV_NP=${SCDV_NP:-$(plat_nproc)}

# Base directory holding the install prefix, including Python and Qt version.
SCDV_BASE=${SCDV_BASE:-${SCDV_PREFIX}-py${PYTHON_VERSION}-qt${QT_MAJOR_VER}.${QT_SUB_VER}}
# Directory for downloaded tarballs.
SCDV_DLDIR=${SCDV_DLDIR:-${SCDV_BASE}/downloaded}
# Shared cache for downloaded tarballs across solvcon development environments
# (scdvs).  When non-empty, ${SCDV_DLDIR} is created as a symlink that points
# here.  The "-" (not ":-") form lets the caller explicitly set
# SCDV_SHARED_DLDIR= to opt out and keep a per-scdv directory instead.
SCDV_SHARED_DLDIR=${SCDV_SHARED_DLDIR-${HOME}/var/scdv/downloaded}

# Directory for building from source.
SCDV_SRCDIR=${SCDV_BASE}/src
# Install root (user-space).
SCDV_USRDIR=${SCDV_BASE}/usr

# Directories under ${SCDV_BASE} and the activation script are created below,
# after the confirmation prompt (or immediately for the --write-activate-only
# path). We deliberately do not touch the filesystem here so that aborting the
# prompt leaves no trace.

PY=${SCDV_USRDIR}/bin/python3
# Put our prefix on PATH so console scripts (cython, etc.) installed into
# ${SCDV_USRDIR}/bin are found by downstream builders such as meson.
export PATH="${SCDV_USRDIR}/bin:${PATH}"

scdv_write_activate() {
  # Delegate to the platform block's plat_write_activate (the dynamic-linker
  # env vars and stray-Qt handling are fundamentally per-OS).
  plat_write_activate "${SCDV_BASE}/activate"
}

echo "SCDV_OS=${SCDV_OS}"
echo "SCDV_NP=${SCDV_NP}"
echo "SCDV_PREFIX=${SCDV_PREFIX}"
echo "SCDV_BASE=${SCDV_BASE}"
echo "SCDV_DLDIR=${SCDV_DLDIR}"
echo "SCDV_SHARED_DLDIR=${SCDV_SHARED_DLDIR}"
echo "SCDV_SRCDIR=${SCDV_SRCDIR}"
echo "SCDV_USRDIR=${SCDV_USRDIR}"
plat_startup_echo
echo "PYTHON_VERSION=${PYTHON_VERSION}"
if [ -n "${SCDV_SKIP_LIST# }" ] ; then
  echo "SCDV_SKIP_LIST=${SCDV_SKIP_LIST# }"
fi
echo "ready to build"

# --write-activate-only is its own explicit confirmation: create the scdv
# directory just enough to drop the activate file in, then exit.
if [ "${SCDV_WRITE_ACTIVATE_ONLY}" = "1" ] ; then
  mkdir -p "${SCDV_BASE}"
  scdv_write_activate
  exit 0
fi

# If SCDV_SHARED_DLDIR is configured, it must exist already.  The script never
# auto-creates it.  Check before any filesystem change (mkdir, symlink, prompt)
# so a typo or unmounted cache fails immediately instead of half-creating the
# per-scdv tree.
if [ -n "${SCDV_SHARED_DLDIR}" ] && [ ! -d "${SCDV_SHARED_DLDIR}" ] ; then
  echo "SCDV_SHARED_DLDIR=${SCDV_SHARED_DLDIR} does not exist;" \
       "create it (e.g. \`mkdir -p ${SCDV_SHARED_DLDIR}\`)" \
       "or unset SCDV_SHARED_DLDIR." >&2
  exit 1
fi

# Final confirmation before any package is built. Skip with --no-confirm in
# non-interactive contexts (CI, automated runs). No directory or activation
# file is created until the user has confirmed, so aborting at the prompt
# leaves the filesystem untouched.
if [ "${SCDV_NO_CONFIRM}" != "1" ] ; then
  # `test -r /dev/tty` returns true even with no controlling terminal, so try
  # to actually open it; that fails with ENXIO when there is no controlling tty
  # (e.g. under setsid or `< /dev/null` in CI).
  if ! { : </dev/tty ; } 2>/dev/null ; then
    echo "no controlling tty for confirmation prompt; rerun with --no-confirm" >&2
    exit 2
  fi
  read -r -p "Press Enter to start the build, Ctrl-C to abort: " _ </dev/tty
fi

mkdir -p "${SCDV_USRDIR}" "${SCDV_SRCDIR}"
if [ -n "${SCDV_SHARED_DLDIR}" ] ; then
  # SCDV_SHARED_DLDIR is set: make SCDV_DLDIR a symlink into the shared cache.
  # The shared directory's existence was verified above; we do not auto-create
  # it.  Refuse to clobber a non-symlink directory at SCDV_DLDIR to avoid
  # losing data; the caller has to move it (or unset SCDV_SHARED_DLDIR) first.
  if [ -L "${SCDV_DLDIR}" ] ; then
    if [ "$(readlink "${SCDV_DLDIR}")" != "${SCDV_SHARED_DLDIR}" ] ; then
      rm "${SCDV_DLDIR}"
      ln -s "${SCDV_SHARED_DLDIR}" "${SCDV_DLDIR}"
    fi
  elif [ -d "${SCDV_DLDIR}" ] ; then
    echo "SCDV_DLDIR=${SCDV_DLDIR} exists as a real directory but" \
         "SCDV_SHARED_DLDIR is set; move its contents to" \
         "${SCDV_SHARED_DLDIR} and remove the directory," \
         "or unset SCDV_SHARED_DLDIR." >&2
    exit 1
  else
    ln -s "${SCDV_SHARED_DLDIR}" "${SCDV_DLDIR}"
  fi
else
  # No shared cache: SCDV_DLDIR is a real per-scdv directory.
  mkdir -p "${SCDV_DLDIR}"
fi
scdv_write_activate

####
# Helpers (translated from devenv/scripts/func.d/build_utils)
####

download_md5() {
  local fn=$1 url=$2 md5hash=${3:-}
  local loc=${SCDV_DLDIR}/${fn}
  local calc=""
  [ -e "${loc}" ] && calc=$(plat_md5 "${loc}")
  if [ ! -e "${loc}" ] || { [ -n "${md5hash}" ] && [ "${md5hash}" != "${calc}" ]; } ; then
    echo "Downloading ${url}"
    curl -fsSL -o "${loc}" "${url}"
    calc=$(plat_md5 "${loc}")
  fi
  if [ -n "${md5hash}" ] && [ "${md5hash}" != "${calc}" ] ; then
    echo "${fn} md5 mismatch: expected ${md5hash} got ${calc} (continuing)"
  fi
}

with_log() {
  local log=$1 ; shift
  echo "run: $*" | tee "${log}"
  { time "$@" ; } 2>&1 | tee -a "${log}"
}

unpack() {
  local fn=$1 destdir=$2
  pushd "${SCDV_SRCDIR}" > /dev/null
  rm -rf "${destdir}"
  tar xf "${SCDV_DLDIR}/${fn}"
  popd > /dev/null
}

####
# Timing: record wall-clock time spent in each build_<pkg> call and print a
# summary table on EXIT. If a build_* aborts the script mid-way (set -e), the
# trap still records that package as FAIL with the partial elapsed time.
####

SCDV_OVERALL_START=$(date +%s)
SCDV_TIMINGS=""
SCDV_CURRENT_PKG=""
SCDV_CURRENT_START=0

scdv_time() {
  local fn=$1 label=$2 end elapsed status
  SCDV_CURRENT_PKG=${label}
  SCDV_CURRENT_START=$(date +%s)
  "${fn}"
  end=$(date +%s)
  elapsed=$((end - SCDV_CURRENT_START))
  if scdv_skip_p "${label}" ; then
    status="skipped"
  else
    status="ok"
  fi
  SCDV_TIMINGS+="${label}|${status}|${elapsed}"$'\n'
  SCDV_CURRENT_PKG=""
}

scdv_fmt_time() {
  local s=$1
  if [ "${s}" -ge 3600 ] ; then
    printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
  elif [ "${s}" -ge 60 ] ; then
    printf '%dm%02ds' $((s/60)) $((s%60))
  else
    printf '%ds' "${s}"
  fi
}

scdv_print_timings() {
  local exit_code=$?
  set +e
  local end_ts overall part label sec status pkg
  end_ts=$(date +%s)
  if [ -n "${SCDV_CURRENT_PKG}" ] ; then
    part=$((end_ts - SCDV_CURRENT_START))
    if [ "${exit_code}" -eq 0 ] ; then
      label="RUNNING"
    else
      label="FAIL"
    fi
    SCDV_TIMINGS+="${SCDV_CURRENT_PKG}|${label}|${part}"$'\n'
  fi
  overall=$((end_ts - SCDV_OVERALL_START))
  # Print only if at least one package was timed or the script ran a
  # non-trivial time (avoids noise on --help / --write-activate-only).
  if [ -z "${SCDV_TIMINGS}" ] && [ "${overall}" -lt 2 ] ; then
    return 0
  fi
  printf '\n=== build timings ===\n'
  printf '%-10s %-8s %10s\n' "package" "status" "time"
  printf '%-10s %-8s %10s\n' "----------" "--------" "----------"
  printf '%s' "${SCDV_TIMINGS}" | while IFS='|' read -r pkg status sec ; do
    [ -z "${pkg}" ] && continue
    printf '%-10s %-8s %10s\n' "${pkg}" "${status}" "$(scdv_fmt_time "${sec}")"
  done
  printf '%-10s %-8s %10s\n' "----------" "--------" "----------"
  printf '%-10s %-8s %10s (exit %d)\n' \
    "TOTAL" "-" "$(scdv_fmt_time "${overall}")" "${exit_code}"
}
trap scdv_print_timings EXIT

####
# Build functions (translated from devenv/scripts/build.d/<pkg>). The recipes
# that are identical across platforms live here; the parts that diverge are
# delegated to the plat_* hooks defined in the per-OS case block above.
####

build_zlib() {
  scdv_skip_p zlib && { echo "skip: zlib" ; return 0 ; }
  local ver=1.3.1 full fn
  full=zlib-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://github.com/madler/zlib/archive/refs/tags/v${ver}.tar.gz" \
    ddb17dbbf2178807384e57ba0d81e6a1
  unpack "${fn}" "${full}"
  pushd "${SCDV_SRCDIR}/${full}" > /dev/null
    with_log configure.log ./configure --prefix="${SCDV_USRDIR}"
    with_log make.log make -j "${SCDV_NP}"
    with_log install.log make install
  popd > /dev/null
}

build_openssl() {
  scdv_skip_p openssl && { echo "skip: openssl" ; return 0 ; }
  local ver=1.1.1m full fn
  full=openssl-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://www.openssl.org/source/${fn}" \
    8ec70f665c145c3103f6e330f538a9db
  unpack "${fn}" "${full}"
  pushd "${SCDV_SRCDIR}/${full}" > /dev/null
    with_log configure.log ./config \
      --prefix="${SCDV_USRDIR}" \
      --openssldir="${SCDV_USRDIR}/share/ssl"
    with_log make.log make -j "${SCDV_NP}"
    with_log install.log make -j "${SCDV_NP}" install
  popd > /dev/null
}

build_sqlite() {
  scdv_skip_p sqlite && { echo "skip: sqlite" ; return 0 ; }
  local ver=3360000 full fn
  full=sqlite-autoconf-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://www.sqlite.org/2021/${fn}" \
    f5752052fc5b8e1b539af86a3671eac7
  unpack "${fn}" "${full}"
  pushd "${SCDV_SRCDIR}/${full}" > /dev/null
    with_log configure.log ./configure --prefix="${SCDV_USRDIR}"
    with_log make.log make
    with_log install.log make install
  popd > /dev/null
}

build_python() {
  scdv_skip_p python && { echo "skip: python" ; return 0 ; }
  local ver=${PYTHON_VERSION} full fn
  full=Python-${ver} ; fn=${full}.tar.xz
  # MD5 left empty; checksum varies per release and we don't pin it here.
  download_md5 "${fn}" \
    "https://www.python.org/ftp/python/${ver}/${fn}" \
    ""
  unpack "${fn}" "${full}"
  pushd "${SCDV_SRCDIR}/${full}" > /dev/null
    plat_python_env
    with_log configure.log ./configure \
      --prefix="${SCDV_USRDIR}" \
      --enable-shared \
      --enable-ipv6 \
      --enable-optimizations \
      --without-ensurepip \
      --with-system-expat \
      --with-system-libmpdec=no \
      --with-readline=editline \
      --with-lto \
      --with-openssl="${SCDV_USRDIR}" \
      --with-system-ffi \
      --enable-loadable-sqlite-extensions
    with_log profile-opt.log make profile-opt -j "${SCDV_NP}"
    with_log install.log make install -j "${SCDV_NP}"
  popd > /dev/null

  # Bootstrap pip (Python was configured --without-ensurepip). get-pip only
  # installs pip; install setuptools+wheel so subsequent pip installs that
  # disable build isolation (or use setup.py) can find them.
  curl -fsSL https://bootstrap.pypa.io/get-pip.py | "${PY}"
  rm -f "${SCDV_USRDIR}/bin/pip"
  "${PY}" -m pip install -U setuptools wheel
  # Update pip.
  "${PY}" -m pip install -U pip
}

build_pybind11() {
  scdv_skip_p pybind11 && { echo "skip: pybind11" ; return 0 ; }
  local ver=2.13.6 full fn
  full=pybind11-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://github.com/pybind/pybind11/archive/refs/tags/v${ver}.tar.gz" \
    a04dead9c83edae6d84e2e343da7feeb
  unpack "${fn}" "${full}"
  mkdir -p "${SCDV_SRCDIR}/${full}/build"
  pushd "${SCDV_SRCDIR}/${full}/build" > /dev/null
    with_log cmake.log cmake \
      -DPYTHON_EXECUTABLE:FILEPATH="${PY}" \
      -DCMAKE_INSTALL_PREFIX="${SCDV_USRDIR}" \
      -DPYBIND11_TEST=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      ..
    with_log make.log make -j "${SCDV_NP}"
    with_log install.log make install
  popd > /dev/null
  pushd "${SCDV_SRCDIR}/${full}" > /dev/null
    with_log setup.log "${PY}" -m pip install .
  popd > /dev/null
}

build_cython() {
  scdv_skip_p cython && { echo "skip: cython" ; return 0 ; }
  local ver=3.0.12 full fn
  full=cython-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://github.com/cython/cython/archive/refs/tags/${ver}.tar.gz" \
    194658f8ae1ae8804f864d4e147fddf6
  unpack "${fn}" "${full}"
  pushd "${SCDV_SRCDIR}/${full}" > /dev/null
    with_log install.log "${PY}" -m pip install .
  popd > /dev/null
}

build_numpy() {
  scdv_skip_p numpy && { echo "skip: numpy" ; return 0 ; }
  local ver=2.2.4 full fn
  full=numpy-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://github.com/numpy/numpy/releases/download/v${ver}/${fn}" \
    56232f4a69b03dd7a87a55fffc5f2ebc
  unpack "${fn}" "${full}"
  pushd "${SCDV_SRCDIR}/${full}" > /dev/null
    with_log dependency.log "${PY}" -m pip install -r requirements/build_requirements.txt
    plat_numpy_install
  popd > /dev/null
  "${PY}" -c "import numpy as np ; np.show_config()"
}

build_scipy() {
  scdv_skip_p scipy && { echo "skip: scipy" ; return 0 ; }
  local ver=1.15.2 full fn
  full=scipy-${ver} ; fn=${full}.tar.gz
  download_md5 "${fn}" \
    "https://github.com/scipy/scipy/releases/download/v${ver}/${fn}" \
    515fc1544d7617b38fe5a9328538047b
  unpack "${fn}" "${full}"
  pushd "${SCDV_SRCDIR}/${full}" > /dev/null
    # dev.py 1.15.2 breaks on click/rich-click >=8.2 with "Sentinel object is
    # not subscriptable" when loading doit tasks, so bypass dev.py and let pip
    # drive the meson build directly.
    with_log dependency.log "${PY}" -m pip install -r requirements/build.txt
    plat_scipy_install
  popd > /dev/null
}

build_qt() {
  scdv_skip_p qt && { echo "skip: qt" ; return 0 ; }
  local major=${QT_MAJOR_VER}
  local sub=${QT_SUB_VER}
  local ver=${major}.${sub}
  local full=qt-${ver}
  local pkgfolder=qt-everywhere-src-${ver}
  local fn=${full}.tar.xz
  local url="https://download.qt.io/official_releases/qt/${major}"
  url="${url}/${ver}/single/qt-everywhere-src-${ver}.tar.xz"
  download_md5 "${fn}" "${url}" 25d4d1dd74c92b978f164e8f20805985
  # A pre-existing Qt may leak through the dynamic-linker / CMake search paths
  # and cause the freshly built Qt tools (rcc/moc/...) to load the older
  # libQt6Core at runtime and fail with "version `Qt_6.x' not found".  The
  # platform block strips those from the environment (the vars and match
  # pattern are per-OS).
  plat_qt_env_strip
  if [ -d "${SCDV_SRCDIR}/${full}" ] ; then
    echo "Qt source already at ${SCDV_SRCDIR}/${full}; skipping extract"
  else
    pushd "${SCDV_SRCDIR}" > /dev/null
      echo "Extracting Qt source (large; takes a few minutes) ..."
      tar xf "${SCDV_DLDIR}/${fn}"
      mv "${pkgfolder}" "${full}"
    popd > /dev/null
  fi

  # Clean previous build dir by default; SCDV_QTNOCLEAN=1 to reuse it.
  if [ "${SCDV_QTNOCLEAN:-}" != "1" ] ; then
    rm -rf "${SCDV_SRCDIR}/${full}/build"
  fi
  mkdir -p "${SCDV_SRCDIR}/${full}/build"
  pushd "${SCDV_SRCDIR}/${full}/build" > /dev/null
    local cfgcmd=(cmake)
    cfgcmd+=("-DCMAKE_INSTALL_PREFIX=${SCDV_USRDIR}")
    cfgcmd+=("-DCMAKE_BUILD_TYPE=Release")
    # Disable Qt modules we do not need (mirrors devenv defaults).
    local m
    for m in qtquicktimeline qtgraphs qt5compat qtactiveqt \
             qtcharts qtcoap qtconnectivity qtdatavis3d qtwebsockets \
             qthttpserver qttools qtdoc qtlottie qtmqtt qtnetworkauth \
             qtopcua qtserialport qtlocation qtpositioning \
             qtquick3dphysics qtremoteobjects qtscxml qtsensors \
             qtserialbus qtspeech qttranslations qtvirtualkeyboard \
             qtwayland qtwebchannel qtwebengine qtwebview \
             qtquickeffectmaker qtgrpc qtmultimedia ; do
      cfgcmd+=("-DBUILD_${m}=OFF")
    done
    # Extra platform Qt cmake args (xcb on Ubuntu, Xcode-check skip on macOS).
    plat_qt_extra_cfg
    cfgcmd+=("${PLAT_QT_CFG[@]}")
    cfgcmd+=("-DQT_ALLOW_SYMLINK_IN_PATHS=ON")
    cfgcmd+=("-DCMAKE_PREFIX_PATH=${SCDV_USRDIR}")
    cfgcmd+=("-G" "Ninja")
    cfgcmd+=("..")

    if [ "${DVQT_NOCONFIG:-}" != "1" ] ; then
      with_log configure.log "${cfgcmd[@]}"
    fi
    if [ "${DVQT_NOBUILD:-}" != "1" ] ; then
      with_log build.log cmake --build . --parallel "${SCDV_NP}"
    fi
    if [ "${DVQT_NOINSTALL:-}" != "1" ] ; then
      with_log install.log cmake --install .
    fi
  popd > /dev/null
}

build_pyside6() {
  scdv_skip_p pyside6 && { echo "skip: pyside6" ; return 0 ; }
  local ver=${PYSIDE_VERSION} full fn url
  full=pyside-setup-everywhere-src-${ver} ; fn=${full}.tar.xz
  # Official Qt for Python source release, verified against the published
  # .md5 sidecar (download.qt.io also serves a matching .sha256:
  # 6ffd9835bb0dd2c56f061d62f1616bb1707cfc0202b80e3165d6be087f3965e2).
  url="https://download.qt.io/official_releases/QtForPython/pyside6"
  url="${url}/PySide6-${ver}-src/${fn}"
  download_md5 "${fn}" "${url}" a6fe3db5855d3cd09a381d0aca7d7f5e
  unpack "${fn}" "${full}"
  # Extra setup.py options (macOS injects a cmake wrapper to hide brew's Qt).
  plat_pyside6_cmake_opt
  pushd "${SCDV_SRCDIR}/${full}" > /dev/null
    # pyside-setup 6.11.1 has a regression in _get_make: the "make" branch
    # returns a str while others return Path, so a later .is_absolute() call
    # crashes.  Use ninja, which returns a Path.
    with_log install.log "${PY}" setup.py install \
      "${PLAT_PYSIDE_CMAKE_OPT[@]}" \
      --qtpaths="${QTPATHS}" \
      --verbose-build \
      --ignore-git \
      --no-qt-tools \
      --enable-numpy-support \
      --parallel="${SCDV_NP}" \
      --make-spec=ninja
  popd > /dev/null
}

####
# Base section
####
if [[ "${SCDVBUILD_ALL:-}" == "1" || "${SCDVBUILD_BASE:-}" == "1" ]] ; then

scdv_time build_zlib zlib
scdv_time build_openssl openssl
scdv_time build_sqlite sqlite

else

echo "Set \${SCDVBUILD_ALL} or \${SCDVBUILD_BASE} to build BASE section"

fi

####
# Python section
####
if [[ "${SCDVBUILD_ALL:-}" == "1" || "${SCDVBUILD_PYTHON:-}" == "1" ]] ; then

echo "Python build uses PGO; expect ~20 min"
scdv_time build_python python
"${PY}" -m pip install -U flake8 autopep8 black pytest jsonschema certifi
# Documentation toolchain (Sphinx-based; see doc/README.md). doxygen, the
# system half of the C++ API path, is in the base prerequisites above.
"${PY}" -m pip install -U sphinx myst-parser pydata-sphinx-theme \
  breathe sphinxcontrib-bibtex sphinxcontrib-mermaid
scdv_time build_pybind11 pybind11

else

echo "Set \${SCDVBUILD_ALL} or \${SCDVBUILD_PYTHON} to build PYTHON section"

fi

####
# Numpy section
####
if [[ "${SCDVBUILD_ALL:-}" == "1" || "${SCDVBUILD_NUMPY:-}" == "1" ]] ; then

scdv_time build_cython cython
# The numpy build differs in Fortran env prep per platform (NOFORTRAN on
# Ubuntu, surfacing brew's gfortran on macOS), so the platform hook drives the
# call.
plat_numpy_run
scdv_time build_scipy scipy

CERT_PATH=$("${PY}" -m certifi)
export SSL_CERT_FILE=${CERT_PATH}
export REQUESTS_CA_BUNDLE=${CERT_PATH}
"${PY}" -m pip install -U matplotlib

else

echo "Set \${SCDVBUILD_ALL} or \${SCDVBUILD_NUMPY} to build NUMPY section"

fi

####
# Qt section
####
if [[ "${SCDVBUILD_ALL:-}" == "1" || "${SCDVBUILD_QT:-}" == "1" ]] ; then

# Point LLVM_INSTALL_DIR at the libclang shiboken should use (apt llvm-22 on
# Ubuntu; Qt's prebuilt libclang fetched into the scdv tree on macOS).
plat_qt_libclang_setup

scdv_time build_qt qt

QTPATHS=${QTPATHS:-${SCDV_USRDIR}/bin/qtpaths6}
if [ ! -x "${QTPATHS}" ] ; then
  echo "qtpaths6 not found at ${QTPATHS}; check the Qt build"
  exit 1
fi
export LLVM_INSTALL_DIR QTPATHS PYSIDE_BUILD=1

scdv_time build_pyside6 pyside6

else

echo "Set \${SCDVBUILD_ALL} or \${SCDVBUILD_QT} to build QT section"

fi
