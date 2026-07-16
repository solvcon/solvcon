#!/bin/bash
#
# Ubuntu 24.04 platform module for build-scdv.sh. Sourced by the entrance, not
# run directly: it sets SCDV_OS_TAG and defines the plat_* hook functions the
# entrance dispatches to. See ../build-scdv.sh for the hook contract and the
# shared build logic.
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

# Guard against running this module directly; it must be sourced.
if [ "${BASH_SOURCE[0]}" = "${0}" ] ; then
  echo "this is a platform module for build-scdv.sh; run build-scdv.sh" >&2
  exit 1
fi

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
