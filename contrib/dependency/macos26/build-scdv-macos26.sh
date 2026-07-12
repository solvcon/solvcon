#!/bin/bash
#
# Build solvcon's runtime dependencies from source on Ubuntu 24.04, with no
# dependency on the devenv tool.
#
# 4 sections (BASE, PYTHON, NUMPY, QT) are guarded by the corresponding
# SCDVBUILD_* environment variables.
#
# Before running this script, install the (Home)brew prerequisites.  The helper
# function below prints the exact command; copy and run it yourself (the script
# never invokes brew).
#
# shiboken (the PySide6 binding generator) needs libclang, which Apple's
# Command Line Tools clang does not ship.  Rather than depend on a Homebrew
# llvm, the QT section downloads Qt's own prebuilt libclang into the scdv tree
# (user-space, no system install) -- see fetch_libclang.  LIBCLANG_VERSION is
# pinned to 21.x: unlike the Ubuntu scdv (which uses LLVM 22 against glibc's
# headers), Qt's prebuilt libclang 22.x segfaults shiboken while parsing the Qt
# headers on macOS, so 21.x stays the macOS sweet spot.  Set LLVM_INSTALL_DIR
# to use an existing libclang (e.g. a brew llvm) instead of fetching.
#
# Apple Silicon (arm64) is the assumed default.  The script also runs on Intel
# Macs, but solvcon does not plan to support Intel Macs.
#
# Python is built --enable-shared (matching the Ubuntu script) into
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
#
# Usage:
#   ./build-scdv-macos26.sh
#       Builds everything (BASE + PYTHON + NUMPY + QT).  The default is "build
#       all" when no SCDVBUILD_* env var is set.  To limit to a single section,
#       set one of SCDVBUILD_BASE/PYTHON/NUMPY/QT=1 and leave the others unset.
#       In addition, SCDVBUILD_ALL=1 works the same as the default invocation.
#       Before any package is built the user is prompted with "Press Enter to
#       start the build, Ctrl-C to abort"; pass --no-confirm to skip.
#   ./build-scdv-macos26.sh --write-activate-only
#       Write only ${SCDV_BASE}/activate and exit.  Useful for refreshing the
#       activation script for an already-built scdv without triggering any
#       build section.
#   ./build-scdv-macos26.sh --skip PKG [--skip PKG ...]
#       Skip a package within whatever sections are otherwise selected.  PKG
#       can be one of: zlib openssl sqlite python pybind11 cython numpy scipy
#       qt pyside6.  The flag accepts a single name, a comma-separated list
#       ("--skip openssl,sqlite"), and may be repeated. The --skip=PKG form
#       also works.
#   ./build-scdv-macos26.sh --no-confirm
#       Skip the "Press Enter to start the build" prompt that fires after the
#       startup echo block.  Use in non-interactive runs (CI, scripts).
#   ./build-scdv-macos26.sh --print-prefix
#       Print SCDV_PREFIX (the path prefix that SCDV_BASE is derived from) to
#       stdout and exit.  Nothing else is printed and no directories are
#       created, so this is safe to capture: PREFIX=$(./script --print-prefix).
#
# Overridable variables (search in the script for their defaults and
# descriptions; not repeating here):
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
#     BREW_PREFIX: Homebrew install prefix (auto-detected from `brew
#       --prefix`; falls back to /opt/homebrew then /usr/local).

set -e
# Without pipefail, exit codes are lost through the `tee` pipe in with_log,
# so a failed build step would otherwise be silently ignored.
set -o pipefail

SCDV_WRITE_ACTIVATE_ONLY=0
SCDV_PRINT_PREFIX_ONLY=0
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

# CPython release tag.
PYTHON_VERSION=${PYTHON_VERSION:-3.14.5}
# Qt major.minor version.
QT_MAJOR_VER=${QT_MAJOR_VER:-6.11}
# Qt patch version.
QT_SUB_VER=${QT_SUB_VER:-1}
# Qt for Python (pyside-setup) source release version.
PYSIDE_VERSION=${PYSIDE_VERSION:-${QT_MAJOR_VER}.${QT_SUB_VER}}
# Qt prebuilt libclang version for shiboken (see fetch_libclang).  Pinned to
# 21.x on macOS: Qt's prebuilt libclang 22.1.2 segfaults shiboken while parsing
# the Qt headers, while libclang <=18 is too old for the macOS 26 SDK's libc++
# headers.
LIBCLANG_VERSION=${LIBCLANG_VERSION:-21.1.2}

# Detect the Homebrew prefix once so later sections can reuse it.  brew is
# keg-only for several packages we depend on (openblas, llvm, gcc), so we build
# the absolute keg paths from BREW_PREFIX rather than relying on PATH.
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

# Default to a full BASE+PYTHON+NUMPY+QT build when the caller has not selected
# a specific section. Setting any one of SCDVBUILD_* to "1" disables this
# default and runs only the explicitly selected sections.
if [ -z "${SCDVBUILD_ALL:-}${SCDVBUILD_BASE:-}${SCDVBUILD_PYTHON:-}${SCDVBUILD_NUMPY:-}${SCDVBUILD_QT:-}" ] ; then
  SCDVBUILD_ALL=1
fi

# Parallel build jobs.
SCDV_NP=${SCDV_NP:-$(sysctl -n hw.ncpu)}
# Path prefix to SCDV_BASE.
SCDV_PREFIX=${SCDV_PREFIX:-${HOME}/var/scdv/macos26}

# --print-prefix is honored as soon as SCDV_PREFIX is resolved so the caller
# can capture the value without triggering any side effects (mkdir,
# activate-write, build).
if [ "${SCDV_PRINT_PREFIX_ONLY}" = "1" ] ; then
  echo "SCDV_PREFIX=${SCDV_PREFIX}"
  exit 0
fi

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

scdv_brew_base_cmd() {
  # Print the brew command for the BASE/PYTHON/NUMPY sections.  The script
  # never runs brew itself; copy the output, review it, and run it.  gcc is
  # included for gfortran (scipy's Fortran sources); openblas is included as an
  # optional fallback BLAS even though numpy/scipy default to Apple's
  # Accelerate framework on macOS.  xz is included for Python's lzma module
  # (Apple's Command Line Tools do not ship liblzma).  doxygen is the system
  # half of the documentation C++ API path (see doc/README.md).
  cat <<'EOF'
brew install \
  cmake ninja pkg-config xz gcc openblas doxygen
EOF
}

# Note: the QT section needs no brew packages beyond the base set above.  Qt is
# built with Apple clang, and shiboken's libclang is fetched from Qt's prebuilt
# packages (see fetch_libclang), not Homebrew.  patchelf is not used on macOS
# (install_name_tool / otool replace it); Qt's platform-plugin dependencies are
# part of the system (cocoa, CoreText, Metal), so no xcb/X11 stack is required.

scdv_write_activate() {
  # Write ${SCDV_BASE}/activate. The activation script is self-locating (reads
  # its own path at source time) so the scdv directory can be moved without
  # rewriting the file. Sourcing it prepends our bin to PATH and our lib to
  # DYLD_LIBRARY_PATH, strips any leaked Homebrew Qt entries from both, and
  # defines scdv_deactivate to restore the original environment.
  local target=${SCDV_BASE}/activate
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

echo "SCDV_NP=${SCDV_NP}"
echo "SCDV_PREFIX=${SCDV_PREFIX}"
echo "SCDV_BASE=${SCDV_BASE}"
echo "SCDV_DLDIR=${SCDV_DLDIR}"
echo "SCDV_SHARED_DLDIR=${SCDV_SHARED_DLDIR}"
echo "SCDV_SRCDIR=${SCDV_SRCDIR}"
echo "SCDV_USRDIR=${SCDV_USRDIR}"
echo "BREW_PREFIX=${BREW_PREFIX:-(none)}"
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

# macOS md5(1) prints just the hash with -q, so unlike Linux md5sum we do not
# need to cut the filename off the second column.
download_md5() {
  local fn=$1 url=$2 md5hash=${3:-}
  local loc=${SCDV_DLDIR}/${fn}
  local calc=""
  [ -e "${loc}" ] && calc=$(md5 -q "${loc}")
  if [ ! -e "${loc}" ] || { [ -n "${md5hash}" ] && [ "${md5hash}" != "${calc}" ]; } ; then
    echo "Downloading ${url}"
    curl -fsSL -o "${loc}" "${url}"
    calc=$(md5 -q "${loc}")
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
# Build functions (translated from devenv/scripts/build.d/<pkg>)
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
    # Pick up our own zlib/openssl/sqlite headers; pull in xz from brew so the
    # lzma extension builds (Apple CLT has no liblzma).
    local cppflags="-I${SCDV_USRDIR}/include"
    if [ -n "${BREW_PREFIX}" ] && [ -d "${BREW_PREFIX}/opt/xz/include" ] ; then
      cppflags="${cppflags} -I${BREW_PREFIX}/opt/xz/include"
    fi
    export CPPFLAGS="${cppflags} ${CPPFLAGS:-}"
    # macOS ld64 / Apple linker: drop GNU-only --no-as-needed; use
    # -headerpad_max_install_names so install_name_tool has room to rewrite
    # paths later (a no-op when we don't, harmless when we do).
    local ldflags="-Wl,-headerpad_max_install_names"
    ldflags="${ldflags} -Wl,-rpath,${SCDV_USRDIR}/lib"
    ldflags="${ldflags} -L${SCDV_USRDIR}/lib"
    if [ -n "${BREW_PREFIX}" ] && [ -d "${BREW_PREFIX}/opt/xz/lib" ] ; then
      ldflags="${ldflags} -L${BREW_PREFIX}/opt/xz/lib"
    fi
    export LDFLAGS="${ldflags} ${LDFLAGS:-}"
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
    rm -f site.cfg
    # Prefer macOS's built-in Accelerate framework over openblas; meson's
    # blas-order/lapack-order accept a comma-separated fallback list, so the
    # build still succeeds on Macs without Accelerate (very rare) by falling
    # back to brew openblas.  The AVX2 cpu-dispatch cap that the Ubuntu script
    # needs is GCC-16-specific; Apple clang on macOS handles the default
    # cpu-dispatch=MAX fine.
    with_log dependency.log "${PY}" -m pip install -r requirements/build_requirements.txt
    with_log install.log "${PY}" -m pip install . --no-build-isolation \
      --config-settings="setup-args=-Dblas-order=accelerate,openblas" \
      --config-settings="setup-args=-Dlapack-order=accelerate,openblas"
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
    # No -isystem CXXFLAGS workaround here: the Ubuntu version needs it to
    # outrank apt's stale pybind11-dev at /usr/include, but macOS does not ship
    # a system pybind11, so our header at ${SCDV_USRDIR}/include wins
    # automatically. scipy autodetects Accelerate via meson on macOS, same as
    # numpy above.
    with_log install.log "${PY}" -m pip install . --no-build-isolation \
      --config-settings="setup-args=-Dblas=accelerate" \
      --config-settings="setup-args=-Dlapack=accelerate"
  popd > /dev/null
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
  # A pre-existing brew Qt may leak through DYLD_LIBRARY_PATH /
  # DYLD_FRAMEWORK_PATH / CMAKE_PREFIX_PATH and cause the freshly built Qt
  # tools (rcc/moc/...) to load the older libQt6Core at runtime.  Strip them
  # out, plus any /opt/homebrew/Cellar/qt or /opt/homebrew/opt/qt entries on
  # PATH.
  unset DYLD_LIBRARY_PATH DYLD_FRAMEWORK_PATH CMAKE_PREFIX_PATH QT_ROOT QT_VER
  PATH=$(printf '%s' "${PATH}" | tr ':' '\n' \
    | grep -vE '/(Cellar|opt)/qt(@|/|$)' | paste -sd: -)
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
    cfgcmd+=("-DQT_ALLOW_SYMLINK_IN_PATHS=ON")
    cfgcmd+=("-DCMAKE_PREFIX_PATH=${SCDV_USRDIR}")
    # Skip Qt's build-time Xcode-version check: it runs `xcrun xcodebuild
    # -version` and aborts ("Can't determine Xcode version.  Is Xcode
    # installed?") on a CLT-only machine -- the common solvcon dev setup --
    # where xcodebuild (shipped only with the full Xcode.app) is absent.  Qt
    # builds fine with Apple clang from the Command Line Tools; only the
    # version probe needs disabling.  Qt 6.11 also supports the macOS 26 SDK
    # (QT_SUPPORTED_MAX_MACOS_SDK_VERSION=26), so no SDK-max-version override
    # is needed any more either.
    cfgcmd+=("-DQT_NO_XCODE_MIN_VERSION_CHECK=ON")
    cfgcmd+=("-G" "Ninja")
    cfgcmd+=("..")

    if [ "${DVQT_NOCONFIG:-}" != "1" ] ; then
      with_log configure.log "${cfgcmd[@]}"
    fi
    if [ "${DVQT_NOBUILD:-}" != "1" ] ; then
      with_log build.log cmake --build . --parallel
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
  local cmake_opt=()
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
# Generated by build-scdv-macos26.sh.
for a in "\$@" ; do
  case "\$a" in
    --build|--install|--open|-E|-P|-N|-L|--version|--find-package)
      exec "${realcmake}" "\$@" ;;
  esac
done
exec "${realcmake}" -DCMAKE_IGNORE_PREFIX_PATH="${BREW_PREFIX}" "\$@"
WRAP
    chmod +x "${cmwrap}/cmake"
    cmake_opt=("--cmake=${cmwrap}/cmake")
  fi
  pushd "${SCDV_SRCDIR}/${full}" > /dev/null
    # pyside-setup 6.11.1 has a regression in _get_make: the "make" branch
    # returns a str while others return Path, so a later .is_absolute() call
    # crashes.  Use ninja, which returns a Path.
    with_log install.log "${PY}" setup.py install \
      "${cmake_opt[@]}" \
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
"${PY}" -m pip install -U nose boto paramiko
# Documentation toolchain (Sphinx-based; see doc/README.md). doxygen, the
# system half of the C++ API path, is in scdv_brew_base_cmd above.
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
# Surface brew's gfortran (installed by `brew install gcc`) so meson picks it
# up for scipy's Fortran sources.  On Apple Silicon the binary is
# /opt/homebrew/opt/gcc/bin/gfortran; on Intel /usr/local/opt/gcc/bin.
if [ -n "${BREW_PREFIX}" ] && [ -d "${BREW_PREFIX}/opt/gcc/bin" ] ; then
  export PATH="${BREW_PREFIX}/opt/gcc/bin:${PATH}"
fi
scdv_time build_numpy numpy
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

# libclang for shiboken.  By default fetch Qt's prebuilt libclang into the scdv
# tree (user-space, no Homebrew) -- see fetch_libclang and the header comment
# for the version rationale.  Set LLVM_INSTALL_DIR to point at an existing
# libclang (e.g. a brew llvm) to skip the download.
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
