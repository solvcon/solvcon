# Agent Instructions

This file provides guidance to AI coding agents working in this repository
(Claude Code and Cursor). `AGENTS.md` at the repo root and
`.cursor/AGENTS.md` are symlinks to this file.

## Project Overview

solvcon is a hybrid C++/Python library for solving conservation laws using the
space-time Conservation Element and Solution Element (CESE) method with
unstructured meshes. The codebase emphasizes:

- High-performance numerical computing through C++ with Python bindings
- Multi-dimensional array operations and contiguous buffer management
- One-dimensional solvers demonstrating the CESE method
- Qt-based GUI (pilot) for spatial data visualization
- Integrated runtime profiler for performance analysis

## Agent Tooling (Claude Code and Cursor)

This repository ships `.claude/` and `.cursor/` directories with permissions,
hooks, and skills tuned to this codebase. General behavioral rules live in
`contrib/prompt/general-rule.md` (not auto-imported). This section indexes
the tools.

### Cursor (`.cursor/`)

- `cli.json` -- project shell/file permissions (translated from
  `.claude/settings.json`; `Shell(...)` replaces Claude's `Bash(...)`).
- `hooks.json` -- `postToolUse` on `Write|StrReplace` and `afterFileEdit` on
  `Write|TabWrite`, wired to the shared `check-source.sh` script.
- `hooks/` -- symlink to `.claude/hooks/`.
- `skills/` -- symlink to `.claude/skills/`.
- `AGENTS.md` -- symlink to `CLAUDE.md` at the repo root.
- `statusline.sh` -- symlink to `.claude/statusline.sh`. Point your
  `~/.cursor/cli-config.json` `statusLine.command` at
  `$PROJECT/.cursor/statusline.sh` (or `.claude/statusline.sh`).

## Claude Code Tooling

### Skills (`.claude/skills/`)

- `cpp-style-review` -- judgment-call C++ review (`m_` prefix, function-body
  placement, `SimpleCollector` preference, pybind11 binding split,
  `const_cast`). Scoped to `git diff`. Invoke after editing files in `cpp/`
  or `gtests/`.
- `python-style-review` -- judgment-call Python review (naming, test intent,
  project conventions). Scoped to `git diff`. Invoke after editing files in
  `solvcon/` or `tests/`.

Skills inherit the caller's model rather than pinning their own. Deterministic
style checks (ASCII, trailing whitespace, modeline, 79-char Python lines) are
owned by hooks, not skills.

### Hooks (`.claude/hooks/`)

- `check-source.sh` -- PostToolUse on `Write|Edit` of source files.  Surfaces
  non-ASCII bytes, trailing whitespace, missing modeline, and Python `>79`
  chars; exits 2 with `path:line -- rule -- fix`.
- `check-doc-images.sh` -- PreToolUse on `Bash`.  Blocks a `git commit` that
  stages a raster or pre-rendered image blob (`.png`, `.jpg`, `.svg`, `.eps`,
  `.pdf`, ...) under `doc/`; exits 2 with the offending paths. Documentation
  schematics are authored as `.tex` PSTricks rendered by `pstake`, so an image
  blob means a build artifact is being checked in instead of its source.

### Settings (`.claude/settings.json`)

- `permissions.allow` whitelists the safe `make` targets, `cmake`, `pytest`,
  lint/format tools, and read-only git/gh. `make clean` and `make cmakeclean`
  deliberately prompt.
- `permissions.deny` hard-blocks only `sudo` and `rm -rf` of root/home.
  Destructive git operations (force-push, `git reset --hard`, `git clean -fd`)
  are discouraged but not blocked -- use them deliberately and only when asked.
- `hooks` wires the scripts above (`check-source.sh` on `Write|Edit`,
  `check-doc-images.sh` on `Bash`).
- `statusLine` runs `.claude/statusline.sh` -- shows model, project, branch
  (with `*` if dirty), and context-window usage.

## Build, Test, Lint, Format

All workflows are driven through `make` from the repo root. The Makefile sets
`PYTHONPATH=$(SOLVCON_ROOT)` so the in-tree `_solvcon` extension is picked up
without installation, and works around macOS SIP stripping `DYLD_LIBRARY_PATH`.

**Build**

- `make` -- build the `_solvcon` Python extension (default target).
- `make pilot` -- build the Qt pilot GUI binary.
- `make clean` / `make cmakeclean` -- remove build artifacts.

**Test**

- `make pytest` -- full Python test suite.
- `make pytest PYTEST_OPTS="tests/test_buffer.py::SimpleArrayBasicTC::test_sort"`
  -- run a single test or subset; `PYTEST_OPTS` is forwarded verbatim to
  pytest.
- `make run_pilot_pytest` -- Python tests that require the pilot GUI;
  accepts `PYTEST_OPTS` the same way.
- `make gtest` -- build and run the full C++ test suite.
- `./build/rel<pyvminor>/gtests/run_gtest --gtest_filter=Suite.Test` -- run a
  single gtest after `make gtest` has built the binary
  (`<pyvminor>` is the Python major+minor, e.g. `314`).
- `make pyprof` -- run profiling benchmarks; results land in
  `profiling/results/`.

**Lint** (`make lint` runs all five)

- `make cformat` -- check C++ formatting (read-only; use
  `make FORCE_CLANG_FORMAT=inplace cformat` to fix).
- `make cinclude` -- check `#include` ordering and style.
- `make flake8` -- Python style and 79-char line limit.
- `make checkascii` -- reject non-ASCII bytes in source.
- `make checktws` -- reject trailing whitespace.

**Format**

Automatic formatting is still work in progress. Do not run `make format` or
`make pyformat`.

Any target whose tool (`clang-format`, `flake8`) is missing prints an install
hint and exits 1. `make cformat` also warns when the local `clang-format` major
version differs from the CI pin (`CLANG_FORMAT_CI_VERSION` in the Makefile).

Build variables such as `CMAKE_BUILD_TYPE`, `BUILD_QT`, and `SOLVCON_PROFILE`
are set in `setup.mk` or as environment variables.

## Architecture

### Hybrid C++/Python Design

solvcon uses a dual-layer hybrid architecture:

1. **C++ Core** (`cpp/solvcon/`): High-performance numerical code
   - Compiled to native libraries with pybind11 bindings
   - Exposed to Python through the `_solvcon` extension module

2. **Python Interface** (`solvcon/`): High-level API and utilities
   - Imports C++ components via `from .core import *`
   - Provides Python-native functionality (plotting, utilities, etc.)

### C++ Component Structure

C++ core lives in `cpp/solvcon/`. `spacetime/` is an old, incorrect CESE
implementation kept for reference only -- do not extend it.

### Testing Structure

Python tests are the default. Prefer writing tests in Python (`tests/`); reach
for C++ gtest only when the code cannot or should not be exercised from Python
-- for example, internals with no Python binding, or behavior that must be
verified at the C++ level.

- **Python tests** (`tests/`): pytest-based, files named `test_*.py`. The
  preferred place for tests.
- **C++ tests** (`gtests/`): googletest-based, files named
  `test_nopython_*.cpp`. Use only when Python cannot or should not reach the
  code under test.
- **Profiling benchmarks** (`profiling/`): files named `profile_*.py`.

See "Build, Test, Lint, Format" above for the `make` invocations.

## Code Style

`STYLE.md` is the canonical source. At a glance:

- **Line economy and breathing room**: Keep a single thought compact (group
  short declarations, do not spread simple logic across many lines), but let a
  longer body breathe by separating its distinct steps with one blank line.
  Do not pad short blocks, and do not compress a function into an unbroken
  wall of statements. Always respect the linting line-width limits; never
  sacrifice them to shorten the line count. See `STYLE.md` "Line Economy and
  Breathing Room".
- **C++**: 4-space indent, `m_` prefix on member vars, angle-bracket includes,
  C++23, prefer `SimpleCollector` / `small_vector` over STL for fundamentals.
- **Python**: PEP-8, 79-char hard limit, flake8.
- **Comments**: Default to none. Add one only for what the code cannot say
  (the why, units, invariants, a non-obvious algorithm). Never restate the
  code, narrate a step, or label obvious structure. When in doubt, leave it
  out. See `STYLE.md` "Comments" for examples.
- **All source**: UTF-8, Unix LF, ASCII-only, no trailing whitespace, modeline
  at EOF.
- **Documentation schematics**: author figures as `.tex` PSTricks sources
  rendered by `solvcon/pstake` (symlinked into `doc/ext/`) at build time. Never
  check in image blobs (`.png`, `.jpg`, `.svg`, `.eps`, `.pdf`, ...) under
  `doc/`; commit the `.tex` source instead, so the figure stays diffable and
  rebuilds from source.

How style is enforced in this repo:

- `.claude/hooks/check-source.sh` owns the deterministic checks (ASCII bytes,
  trailing whitespace, modeline at EOF, Python `>79`-char lines).
- `.claude/hooks/check-doc-images.sh` blocks committing image blobs under
  `doc/`, keeping schematics in `.tex`/PSTricks form (rendered by `pstake`).
- The `cpp-style-review` and `python-style-review` skills in
  `.claude/skills/` own the judgment-call rules (`m_` prefix in context,
  function-body placement, container choice, pybind11 binding split, test
  intent). They are scoped to `git diff`.

For the full rule set with examples, see `STYLE.md`.

## Pull Request Guidelines

When opening a pull request, reference the related issue (e.g., "Related to
#725") instead of using closing keywords like "close #725", "closes #725", or
"fixes #725". We do not let PR and commit log comments to mandate the
management.

## Writing Style for Prose

This applies to every passage you write for humans: code comments, commit
messages, PR and issue descriptions and comments, and documentation.

- Minimize em-dashes (the "--" rendered as a long dash, or the Unicode `U+2014`
  character). They are hard to read. Prefer a comma, a colon, parentheses, or
  two separate sentences. Reserve a dash only when no other punctuation reads
  as clearly.
- Source files are ASCII-only (see "Code Style"), so never emit the Unicode
  em-dash, en-dash, or "smart quotes" in them. The same restraint is expected
  in GitHub prose even though GitHub accepts Unicode.
- Write plainly. Short sentences beat long ones strung together with dashes.

## Development Workflow

### Dependencies

Install scripts available in `contrib/dependency/`

### Virtual Environments

**IMPORTANT**: Using Python virtual environments (venv, conda) is **strongly
discouraged** for solvcon development. The project is designed to work with
system Python. Virtual environment bugs are not actively resolved.

Use https://github.com/solvcon/devenv to build dependency from source and
install in user space. Do not install dependency system-wide. Installation of
any dependency requires user review and consent.

### Platform-Specific Notes

**macOS**: System Integrity Protection (SIP) may interfere with
`DYLD_LIBRARY_PATH`. The Makefile sets `PYTHONPATH` as a workaround.

## Profiling System

solvcon includes an integrated runtime profiler:

1. Enable with `SOLVCON_PROFILE=ON` during build
2. Use `toggle.py` API to enable/disable profiling regions
3. Run profiling scripts with `make pyprof`
4. Results written to `profiling/results/`

## Qt GUI Development

The pilot application (`cpp/binary/pilot/`) is a standalone Qt6-based viewer:

- Requires `BUILD_QT=ON` (default)
- Uses PySide6 for Python-Qt integration
- Resource files in `resources/pilot/`
- Can be disabled with `BUILD_QT=OFF` for headless builds

## Common Development Patterns

### Adding a New C++ Component

1. Create a directory under `cpp/solvcon/`.
2. Add header files with proper include guards.
3. Update `cpp/solvcon/CMakeLists.txt` to include new sources.
4. Add pybind11 bindings if Python access is needed.
5. Write tests. Prefer Python tests in `tests/`; add a `gtests/` test only
   when the behavior cannot or should not be exercised from Python.

### Adding Python-Only Functionality

1. Add a module to `solvcon/`.
2. Update `solvcon/__init__.py` if needed.
3. Write tests in `tests/`.
4. Update `setup.py` packages list if adding a new package.

### Memory Management

- Use `ConcreteBuffer` for raw memory.
- Use `SimpleArray` for typed multi-dimensional arrays.
- Buffers support both ownership and non-owning views.
- Python and C++ share the same buffer memory (zero-copy).

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
