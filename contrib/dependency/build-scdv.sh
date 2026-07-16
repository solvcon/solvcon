#!/bin/bash
#
# Build solvcon's runtime dependencies from source into a user-space prefix,
# with no dependency on the devenv tool. This is the cross-platform entrance:
# it holds the shared build logic and dispatches the platform-specific parts to
# a per-OS module it sources. The target platform is auto-detected from
# `uname -s` (Linux -> ubuntu, Darwin -> macos) and can be forced with
# SCDV_OS=ubuntu|macos.
#
#   contrib/dependency/build-scdv.sh            entrance (this file)
#   contrib/dependency/ubuntu2404/platform.sh   Ubuntu 24.04 module
#   contrib/dependency/macos26/platform.sh      macOS 26 module
#   contrib/dependency/*/build-scdv-*.sh        thin wrappers that exec this
#                                               entrance with SCDV_OS set
#
# The per-OS module defines SCDV_OS_TAG and the plat_* hook functions this
# entrance calls (see "Platform module contract" below); it is sourced, not run
# directly. Platform-specific notes (apt/brew prerequisites, libclang, etc.)
# live in the module headers. The per-package build recipes are inlined
# translations of the scripts under devenv/scripts/build.d/ (zlib, openssl,
# sqlite, python, pybind11, cython, numpy, scipy, qt, pyside6).
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
# Overridable variables (search here and in the platform module for defaults):
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
# Platform module contract. The sourced per-OS module must set the variable
# SCDV_OS_TAG (the token baked into the default prefix, e.g. ubuntu2404) and
# define these hook functions, which this entrance calls:
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

# Resolve this script's directory so the per-OS module can be sourced no matter
# what the caller's working directory is.
SCDV_SELFDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Detect the target platform.
SCDV_OS=${SCDV_OS:-}
if [ -z "${SCDV_OS}" ] ; then
  case "$(uname -s)" in
    Linux)
      SCDV_OS=ubuntu
      # `uname -s` reports only "Linux", not the distribution, so every Linux
      # host defaults to the Ubuntu 24.04 module (apt package names,
      # /usr/lib/llvm-22, x86_64 openblas paths). Warn when /etc/os-release
      # does not look like Ubuntu 24.04 so the assumption is visible; set
      # SCDV_OS explicitly to silence this. The module is still used (as the
      # per-OS Ubuntu script always did), just no longer silently.
      if [ -r /etc/os-release ] ; then
        _scdv_osrel=$(set +e ; . /etc/os-release 2>/dev/null ; \
                      printf '%s:%s' "${ID:-}" "${VERSION_ID:-}")
        if [ "${_scdv_osrel}" != "ubuntu:24.04" ] ; then
          echo "warning: assuming the Ubuntu 24.04 build module on" \
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

# Source the per-OS module: it sets SCDV_OS_TAG and defines the plat_* hooks.
case "${SCDV_OS}" in
  ubuntu) SCDV_MODULE=${SCDV_SELFDIR}/ubuntu2404/platform.sh ;;
  macos)  SCDV_MODULE=${SCDV_SELFDIR}/macos26/platform.sh ;;
  *) echo "unknown SCDV_OS='${SCDV_OS}'; expected ubuntu or macos" >&2
     exit 1 ;;
esac
if [ ! -r "${SCDV_MODULE}" ] ; then
  echo "platform module not found or not readable: ${SCDV_MODULE}" >&2
  exit 1
fi
# shellcheck source=/dev/null
. "${SCDV_MODULE}"

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

# --print-deps needs only the sourced module (plat_print_deps). Exit before
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
  # Delegate to the platform module, which writes the activation script (the
  # dynamic-linker env vars and stray-Qt handling are fundamentally per-OS).
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
# delegated to the plat_* hooks from the sourced module.
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
  # module strips those from the environment (the vars and match pattern are
  # per-OS).
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
# Ubuntu, surfacing brew's gfortran on macOS), so the module drives the call.
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
