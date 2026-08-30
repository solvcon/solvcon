# Build solvcon

All workflows are driven through `make` from the repository root.  The Makefile
wraps CMake and sets `PYTHONPATH` so the in-tree `_solvcon` extension is picked
up without installation.

- `make`: build the `_solvcon` Python extension (the default target).
- `make pilot`: build the Qt pilot GUI binary.
- `make clean` / `make cmakeclean`: remove build artifacts.

## Build Types

`CMAKE_BUILD_TYPE` selects one of CMake's three standard build types.  Each
gets its own build tree, so switching between them never invalidates another
one's cache.  `<pyvminor>` is the active Python major and minor version, e.g.
`314`.

| `CMAKE_BUILD_TYPE` | Build tree               | Flags             |
|:-------------------|:-------------------------|:------------------|
| `RelWithDebInfo`   | `build/reldbg<pyvminor>` | `-O3 -g -DNDEBUG` |
| `Release`          | `build/rel<pyvminor>`    | `-O3 -DNDEBUG`    |
| `Debug`            | `build/dbg<pyvminor>`    | `-g`, no `-O`     |

The default is `RelWithDebInfo`: optimized, and carrying the debug symbols
that let a debugger name a function and a line.  It is what you want unless
you have a reason to want something else.

`Release` drops the symbols, which compiles noticeably faster and leaves a
much smaller build tree, so it suits a session that is only going to run the
test suite.  It is not merely the default minus `-g`: pybind11 gives the
extension module a set of extras for every build type except `Debug` and
`RelWithDebInfo`, so under `Release` the module is also link-time optimized
and stripped.  Stripping is why a debugger can name a frame inside
`_solvcon` under the default and not under `Release`.

`Debug` turns the optimizer off.  Reach for it when you need to step through
code and inspect variables, which `RelWithDebInfo` does poorly because `-O3`
inlines functions and discards variables.

The trees are separate but they all write the extension module to `solvcon/`,
so whichever built last is the one Python imports.  Building a tree whose
objects are already current does not relink, which means switching build types
alone does not replace the installed module.  Build the type you want last, or
touch a source to force the relink.

## Build Options

The Makefile configures through `cmake --preset`, so the defaults are the ones
in `CMakePresets.json`: `dev-reldbg`, `dev-rel`, and `dev-dbg` for the three
build types above.  It adds only the ABI-tagged build directory, the generator,
and whatever options are set below, each of which becomes a `-D` flag layered
on the preset.  Set `CMAKE_PRESET` to build a different preset, for instance
one of your own from `CMakeUserPresets.json`.

Key options can be set on the command line, in `setup.mk` (which is read by
`Makefile`), or as environment variables:

| Variable           | Default          | Purpose                         |
|:-------------------|:-----------------|:--------------------------------|
| `CMAKE_BUILD_TYPE` | `RelWithDebInfo` | see the table above             |
| `BUILD_QT`         | `ON`             | build the Qt GUI components     |
| `BUILD_METAL`      | `OFF`            | build Metal GPU support (macOS) |
| `SOLVCON_PROFILE`  | `OFF`            | enable the runtime profiler     |
| `USE_CLANG_TIDY`   | `OFF`            | run clang-tidy during the build |
| `USE_CCACHE`       | `ON`             | use ccache when it is installed |

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
