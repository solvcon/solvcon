# Build Dependencies

To build the dependencies from source and install them into user space rather
than system-wide, use the standalone scdv build scripts in
`contrib/dependency/` described below.

For a complete, self-contained environment, the single cross-platform script
`build-scdv.sh` builds solvcon's full runtime stack from source -- zlib,
OpenSSL, SQLite, CPython, pybind11, Cython, NumPy, SciPy, Qt, and PySide6 --
into a versioned prefix under your home directory (by default
`${HOME}/var/scdv/<platform>-py<pyver>-qt<qtver>`). The target platform is
auto-detected from `uname -s` (Ubuntu 22.04, 24.04 or 26.04, or macOS 26);
set `SCDV_OS` to force it. Windows uses the separate
`windows/build-scdv-windows.ps1`.

The build is organized into four sections: `BASE`, `PYTHON`, `NUMPY`, and `QT`
with the corresponding environment variables `SCDVBUILD_BASE`,
`SCDVBUILD_PYTHON`, `SCDVBUILD_NUMPY`, and `SCDVBUILD_QT`. If none is set, the
script builds everything.

The script never runs `apt` or Homebrew itself. Print the prerequisite
commands, review them, and run them yourself:

```sh
cd contrib/dependency
./build-scdv.sh --print-deps   # review, then run the printed commands
./build-scdv.sh                # build the whole stack into the prefix
```

Useful flags: `--print-prefix` reports the install prefix and exits;
`--no-confirm` skips the pre-build prompt for non-interactive runs; `--skip
PKG` omits a package (repeatable or comma-separated); and
`--write-activate-only` (re)writes just the activation script.

## Core-Only Build

A solvcon built with `BUILD_QT=OFF` needs no Qt or PySide6, and `--core`
builds exactly that much: the `BASE`, `PYTHON`, and `NUMPY` sections, into
the same prefix a full build uses. The Qt section can then be added later
without redoing the core, so starting with `--core` costs nothing if the GUI
turns out to be needed. `--print-deps --core` prints only the prerequisites
the three core sections need.

```sh
cd contrib/dependency
./build-scdv.sh --print-deps --core   # core prerequisites only
./build-scdv.sh --core                # BASE, PYTHON, and NUMPY; no Qt
SCDVBUILD_QT=1 ./build-scdv.sh        # add Qt and PySide6 to that prefix
```

`--core` always selects the three core sections and refuses `SCDVBUILD_QT=1`
or `SCDVBUILD_ALL=1` on the same command line. The prefix keeps the
`-qt<qtver>` suffix even before the Qt section is built, so the two builds
land in one directory. A solvcon build tree configured with `BUILD_QT=OFF`
remembers that choice; after the Qt section is added, run `make cmakeclean`
(or delete the build directory) before `make pilot`.

The core build takes 10 to 15 minutes on a fast machine, most of it in
CPython (profile-guided optimization) and CMake. Ubuntu 26.04 skips the CMake
build because its own `cmake` is new enough.

## Platform Notes

- Ubuntu 22.04 (`ubuntu2204` prefix) follows the 24.04 recipe: GCC 16 comes
  from the toolchain PPA and LLVM 22 from the `jammy` suite of apt.llvm.org.
  Its `libexpat` is too old for CPython 3.14's test suite, which the
  profile-guided build runs, so CPython is built with its bundled expat
  there. Only the core sections are verified on 22.04; the Qt section has
  not been exercised.
- Ubuntu 24.04 (`ubuntu2404`) is the CI platform: GCC 16 from the toolchain
  PPA, LLVM 22 from apt.llvm.org, and CMake built by the `BASE` section
  because apt's 3.28 is below solvcon's minimum.
- Ubuntu 26.04 (`ubuntu2604`) carries GCC 16, LLVM 22, and a new enough
  CMake in its own archive, so there is no extra repository and the `BASE`
  section uses the system `cmake`. Its desktop is Wayland only, so Qt also
  builds the `qtwayland` plugin there.
- macOS 26 (`macos26`) builds with Apple clang from the Command Line Tools;
  Homebrew supplies `gfortran` (via `gcc`), `openblas`, and `xz`, and the Qt
  section downloads Qt's prebuilt libclang for shiboken.

When the build finishes it writes an `activate` script in the prefix.  Source
it to put the freshly built Python and Qt on your `PATH`, and run
`scdv_deactivate` to restore the original environment:

```sh
source ${HOME}/var/scdv/<platform>-py<pyver>-qt<qtver>/activate
```

The activation exports `SCDV_USRDIR`, the prefix that
`contrib/cmake/CMakeUserPresets.scdv.json` is installed with; see
{doc}`/devguide/cmake` for what that file is for.  It also points
`SOLVCON_DEPS_CACHE` at the prefix's `downloaded` directory, so a solvcon
build caches the archives it fetches beside the ones this script downloaded.
A value you set yourself is left alone.

Two toolchains sit outside the build sections, so the full `--print-deps`
ends with them (`--print-deps --core` stops after the core prerequisites).
The first is LaTeX. Building the documentation needs it even for plain
HTML, because the `pstake` extension renders the PSTricks figures through
`latex`, `dvips`, and ImageMagick, with Ghostscript behind the EPS step. On
Ubuntu that is the `texlive-*` set led by `texlive-pstricks`, plus
`ghostscript` and `imagemagick`; note that Ubuntu's ImageMagick disables the
EPS and PS coders in `policy.xml`, so either allow them or drop `imagemagick`
and let the Ghostscript fallback do the work. On macOS there is no system TeX
at all, so it is the `mactex-no-gui` cask (a several-gigabyte download) plus
`ghostscript` and `imagemagick`. The cask installs into `/Library/TeX/texbin`
and reaches `PATH` through `/etc/paths.d`, so run
`eval "$(/usr/libexec/path_helper)"` or open a new terminal before building
the documentation.

The second is the C++ lint tools `make lint` needs. Both `clang-format` and
`clang-tidy` come from the same LLVM 22 (`clang-format-22` and `clang-tidy-22`
from apt.llvm.org on Ubuntu, `llvm@22` from Homebrew on macOS), symlinked under
their bare names because that is how the `Makefile` and CMake look them up.
Note that LLVM 22 means `clang-format` is newer than the
`CLANG_FORMAT_CI_VERSION` the `Makefile` pins, so `make cformat` prints a
version-drift warning.

Once the dependencies are in place, build solvcon as described in
{doc}`build_solvcon`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
