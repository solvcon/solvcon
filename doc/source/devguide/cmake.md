# CMake Presets

`CMakePresets.json` at the repository root holds solvcon's build
configuration: the generator, the build type, the cache variables, and the run
environment.  It is checked in, so every contributor and every continuous
integration job configures the same way, and both supported IDEs read it
directly.

List what the file offers on this host, then configure and build:

```bash
cmake --list-presets
cmake --preset <name>
cmake --build --preset <name>
```

A preset that names a toolchain the current host cannot run carries a
`condition`, so it does not appear in the listing there.  On Linux and macOS
the make targets described in {doc}`/start/build_solvcon` remain the entry
point every contributor uses, and they configure through these presets: the
Makefile adds the ABI-tagged build directory, the generator, and whatever
`make VAR=value` sets, and nothing else.

| Configure preset | Host          | What it configures                    |
|:-----------------|:--------------|:--------------------------------------|
| `dev-rel`        | Linux, macOS  | optimized module and pilot            |
| `dev-dbg`        | Linux, macOS  | debuggable module and pilot           |
| `dev-noqt`       | Linux, macOS  | optimized module, no pilot            |
| `win-rel`        | Windows       | optimized module and pilot, MSVC      |
| `win-dbg`        | Windows       | debuggable module and pilot, MSVC     |

Each configure preset comes with three build presets, one per thing that is
actually built:

| Build preset          | Targets                            |
|:----------------------|:-----------------------------------|
| `<preset>`            | the extension module and the pilot |
| `<preset>-module`     | the extension module alone         |
| `<preset>-gtest`      | the C++ gtest binary               |

`cmake --build --preset <name>` therefore needs no `--target` argument.  The
`dev-noqt` preset has no pilot to build, so it has only the plain and `-gtest`
entries.

The build tree is `build/<configure preset>`, named through the
`${presetName}` macro so that a preset inheriting another gets its own tree
rather than writing into its parent's.  It is one tree per preset, which is
what both IDEs assume, and it is not the Makefile's `build/rel<pyvminor>`, so
a preset build and a make build never share a cache.  They do share where the
extension module lands, `solvcon/` and the repository root, so whichever built
last is the one Python imports.

Test presets run the suites CTest has registered.  There is one per configure
preset, plus label-filtered ones on `dev-rel`:

```sh
ctest --preset dev-rel            # every suite
ctest --preset dev-rel-cpp        # the C++ cases alone
ctest --preset dev-rel-python     # the Python suite alone
ctest --preset dev-rel-pilot      # the Python suite inside the pilot binary
```

The C++ cases come from `gtest_discover_tests`, which enumerates them by
running the built binary, and the plain build preset does not build it.  Build
`<preset>-gtest` before a run that includes them, or CTest reports the single
placeholder case `test_nopython_NOT_BUILT` and fails:

```sh
cmake --preset dev-rel
cmake --build --preset dev-rel
cmake --build --preset dev-rel-gtest
ctest --preset dev-rel
```

CLion reads no test presets, so a CLion user reaches the same suites through
its CTest integration against the configured build tree.  That works because
the suites are registered with `add_test()`, not because a preset names them.

Workflow presets chain configure, build, and test into one command:

```bash
cmake --workflow --preset ci-win-rel
```

They exist for CI.  The `ci-` presets are what the Windows jobs name instead
of spelling out a `cmake` command line, and the two values a runner owns, the
pybind11 package directory and the MKL paths, reach them through `$env{}`
rather than being written into the file.  A preset that a command line names
cannot be `hidden`, so the `ci-` presets do appear in the preset pickers; the
prefix and their descriptions are what say to pick something else.  CLion does
not show workflow presets at all.

One consequence is worth knowing.  `_solvcon` is an ABI-tagged extension and
`PYTHON_EXECUTABLE` is a cache variable, so pointing the same preset at a
different interpreter reuses a stale cache.  Reconfigure with `--fresh` after
switching interpreters:

```bash
cmake --preset dev-rel --fresh
```

## Machine paths belong in `CMakeUserPresets.json`

Three cache variables name directories that exist on one machine only:

| Variable            | What it names                                         |
|:--------------------|:------------------------------------------------------|
| `PYTHON_EXECUTABLE` | the interpreter the extension is built against        |
| `pybind11_path`     | the pybind11 CMake package directory for that Python  |
| `CMAKE_PREFIX_PATH` | the dependency prefix holding Qt6, PySide6, and so on |

They must not be checked in, which is what `CMakeUserPresets.json` is for.
CMake reads it automatically, it implicitly includes `CMakePresets.json`, and
it is gitignored.  Both VS Code and CLion pick it up with no IDE settings at
all, which is the only way a dependency prefix reaches an IDE build: there is
no wrapper script to fall back on there.

There is one template per host family, because the checked-in presets a user
preset builds on are themselves split by host.  Copy the one for your host and
edit the paths:

```bash
# Linux and macOS
cp contrib/cmake/CMakeUserPresets.example.json CMakeUserPresets.json
# Windows
cp contrib/cmake/CMakeUserPresets.win-example.json CMakeUserPresets.json
```

Each template defines a configure preset named `local` that inherits a
checked-in preset and adds the three variables, plus the three build presets
that go with it:

```json
{
  "version": 10,
  "configurePresets": [
    {
      "name": "local",
      "inherits": "dev-rel",
      "displayName": "Local dependency prefix",
      "cacheVariables": {
        "PYTHON_EXECUTABLE": {
          "type": "FILEPATH",
          "value": "/path/to/prefix/bin/python3"
        },
        "pybind11_path": "/path/to/prefix/lib/python3.14/site-packages/pybind11/share/cmake/pybind11",
        "CMAKE_PREFIX_PATH": "/path/to/prefix"
      }
    }
  ],
  "buildPresets": [
    { "name": "local", "inherits": "dev-rel", "configurePreset": "local" }
  ]
}
```

Every `inherits` in the file names a preset of one host family, and they have
to stay a matched set.  A `dev-` build preset carries a condition that
disables it on Windows and a `win-` one carries the opposite, so a `local`
that inherits a `win-` configure preset and `dev-` build presets configures
and then refuses to build.  Change them together, or start from the template
for the family you want.

`pybind11_path` is what `python3 -m pybind11 --cmakedir` prints for the
interpreter named above it.  After that, `cmake --preset local` configures
from a bare checkout, and `local` appears in the IDE preset pickers alongside
the checked-in presets.

In CLion the `enablePythonIntegration` key in the `jetbrains.com/clion` vendor
map, already set in the checked-in presets, hands the IDE's selected
interpreter to the configure step.  A CLion user can therefore drop the
`PYTHON_EXECUTABLE` line and keep only the other two.

## IDE notes

Nothing else needs to be configured, and a few things must not be.

- VS Code CMake Tools turns preset mode on by itself once `CMakePresets.json`
  exists.  In that mode it ignores `cmake.buildDirectory`,
  `cmake.generator`, and `cmake.configureSettings` in `settings.json`, and it
  disables kit selection, so a settings file that restates any of those only
  causes confusion.
- CLion imports the presets as read-only profiles and leaves a newly seen one
  disabled until it is enabled in the CMake settings.
- CLion reads configure and build presets only.  Anything expressed solely as
  a test or workflow preset is invisible there, so the configure and build
  presets stay self-sufficient.
- Code navigation works from `compile_commands.json`, which the presets export.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
