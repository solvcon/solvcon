#!/bin/bash
#
# macOS 26 platform module for build-scdv.sh. Sourced by the entrance, not run
# directly: it sets SCDV_OS_TAG and defines the plat_* hook functions the
# entrance dispatches to. See ../build-scdv.sh for the hook contract and the
# shared build logic.
#
# Before running the build, install the (Home)brew prerequisites.
# plat_print_deps prints the exact command; copy and run it yourself (the build
# never invokes brew).
#
# shiboken (the PySide6 binding generator) needs libclang, which Apple's
# Command Line Tools clang does not ship.  Rather than depend on a Homebrew
# llvm, the QT section downloads Qt's own prebuilt libclang into the scdv tree
# (user-space, no system install) -- see fetch_libclang.  LIBCLANG_VERSION is
# pinned to 21.x: unlike the Ubuntu module (which uses LLVM 22 against glibc's
# headers), Qt's prebuilt libclang 22.x segfaults shiboken while parsing the Qt
# headers on macOS, so 21.x stays the macOS sweet spot.  Set LLVM_INSTALL_DIR
# to use an existing libclang (e.g. a brew llvm) instead of fetching.
#
# Apple Silicon (arm64) is the assumed default.  The build also runs on Intel
# Macs, but solvcon does not plan to support Intel Macs.
#
# Python is built --enable-shared (matching the Ubuntu module) into
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

# Guard against running this module directly; it must be sourced.
if [ "${BASH_SOURCE[0]}" = "${0}" ] ; then
  echo "this is a platform module for build-scdv.sh; run build-scdv.sh" >&2
  exit 1
fi

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
  # to brew openblas.  The AVX2 cpu-dispatch cap that the Ubuntu module needs
  # is GCC-16-specific; Apple clang on macOS handles the default
  # cpu-dispatch=MAX fine.
  with_log install.log "${PY}" -m pip install . --no-build-isolation \
    --config-settings="setup-args=-Dblas-order=accelerate,openblas" \
    --config-settings="setup-args=-Dlapack-order=accelerate,openblas"
}

plat_scipy_install() {
  # Runs in the unpacked scipy source directory (the caller has pushd'd).
  # No -isystem CXXFLAGS workaround here: the Ubuntu module needs it to outrank
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
