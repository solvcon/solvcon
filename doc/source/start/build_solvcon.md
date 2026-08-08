# Build solvcon

All workflows are driven through `make` from the repository root.  The Makefile
wraps CMake and sets `PYTHONPATH` so the in-tree `_solvcon` extension is picked
up without installation.

- `make`: build the `_solvcon` Python extension (the default target).
- `make pilot`: build the Qt pilot GUI binary.
- `make clean` / `make cmakeclean`: remove build artifacts.

Release builds (the default) land in `build/rel<pyvminor>` and debug builds in
`build/dbg<pyvminor>`, where `<pyvminor>` is the active Python major and minor
version, e.g. `314`.

## Build Options

The Makefile configures through `cmake --preset`, so the defaults are the ones
in `CMakePresets.json`: `dev-rel` for a release build and `dev-dbg` for a debug
one.  It adds only the ABI-tagged build directory, the generator, and whatever
options are set below, each of which becomes a `-D` flag layered on the preset.
Set `CMAKE_PRESET` to build a different preset, for instance one of your own
from `CMakeUserPresets.json`.

Key options can be set on the command line, in `setup.mk` (which is read by
`Makefile`), or as environment variables:

| Variable             | Default   | Purpose                          |
|:---------------------|:----------|:---------------------------------|
| `CMAKE_BUILD_TYPE`   | `Release` | `Release` or `Debug`             |
| `BUILD_QT`           | `ON`      | build the Qt GUI components      |
| `BUILD_METAL`        | `OFF`     | build Metal GPU support (macOS)  |
| `SOLVCON_PROFILE`    | `OFF`     | enable the runtime profiler      |
| `USE_CLANG_TIDY`     | `OFF`     | run clang-tidy during the build  |
| `USE_CCACHE`         | `ON`      | use ccache when it is installed  |

Install `ccache` (`brew install ccache`, `apt install ccache`) to make a
rebuild after `make cmakeclean` mostly cache hits; a host without it builds
the same way as before, and an MSVC build skips the cache either way.  The
configure log says `use ccache` or `not use ccache`.  An existing build tree
picks up an install or an uninstall on its next configure, which
`make cmakeclean` forces.

CMake itself is configured through `CMakePresets.json`; see
{doc}`/devguide/cmake` for the preset a Windows build or an IDE selects, and
for where the paths of a local dependency prefix belong.

## Sharing ccache Between Checkouts

A second checkout, such as a `git worktree` tree, reuses nothing from the
first.  The compile line carries absolute paths, and ccache also hashes the
working directory when the build carries debug information.  Two settings
remove both:

```sh
ccache --set-config base_dir=$HOME
printf 'hash_dir = false\n' > build/rel314/ccache.conf
```

The second file needs ccache 4.13 or newer, and `make cmakeclean` removes it
along with the build tree.  ccache reads it from the compile directory and
its parents, so write one into each tree meant to share, and leave it out of
a debug tree: an object reused elsewhere records the source paths of the
checkout that compiled it first.

After building, run the tests as described in {doc}`/devguide/testing`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
