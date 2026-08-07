# Testing

Tests are driven through `make` from the repository root.  Python tests are the
default and live in `tests/` as `test_*.py`. C++ tests live in `gtests/` as
`test_nopython_*.cpp` and are reserved for behaviour that cannot or should not
be reached from Python.

- `make pytest`: run the full Python test suite.
- `make pytest PYTEST_OPTS="tests/test_buffer.py::SimpleArrayBasicTC"`:
  forward options verbatim to pytest to run a subset.
- `make run_pilot_pytest`: Python tests that need the pilot GUI.
- `make gtest`: build and run the full C++ test suite.
- `make pyprof`: run the profiling benchmarks (see {doc}`/compute/profiling`).

After `make gtest` has built the binary, a single C++ test can be run directly:

```sh
./build/rel<pyvminor>/gtests/run_gtest --gtest_filter=Suite.Test
```

where `<pyvminor>` is the active Python major and minor version, e.g. `314`.

## Through CTest

Every suite is also registered with CTest, so `ctest` runs the C++ cases, the
Python suite, and the pilot suite from one command against a configured build
tree.  This is what an IDE drives, and it is how the C++ cases become
individually selectable.

```sh
ctest --preset dev-rel            # every suite
ctest --preset dev-rel-cpp        # the C++ cases alone
ctest --preset dev-rel-python     # the Python suite alone
ctest --preset dev-rel-pilot      # the Python suite inside the pilot binary
```

The presets are described in {doc}`/devguide/cmake`.  Against a build tree
that was configured without one, the same selection is `ctest -L cpp`,
`ctest -L python`, or `ctest -L pilot` from inside the tree.

The C++ cases are registered by `gtest_discover_tests`, which enumerates them
by running the built binary, so `test_nopython` has to be built before `ctest`
can see them.  The plain `<preset>` build preset builds the module and the
pilot, not the test binary, so build `<preset>-gtest` as well before a run
that includes the C++ cases.  Skipping it leaves CTest with the placeholder
case `test_nopython_NOT_BUILT`, which fails.

`make pytest` and `make run_pilot_pytest` are unchanged and remain the way to
forward `PYTEST_OPTS` to a subset.

## Automatic Testing on GitHub Actions

Continuous integration runs on GitHub Actions. The workflows live in
`.github/workflows/`. Most checks run on every change, while the heavier and
slowly-changing ones run on a nightly schedule. Each job drives the same `make`
targets described above, so a failure can usually be reproduced locally.

The fast set runs on every pull request and `master` push:

- `check_skip`: decides whether to skip the heavy jobs (see below).
- `lint`: `make cformat`, `cinclude`, `checkascii`, `checktws`, `flake8`, and
  clang-tidy on the diff, on ubuntu and macOS.
- `standalone_buffer`: the standalone buffer build on ubuntu.
- `build`: `make gtest` plus `make pytest` with Qt off and on, and the pilot,
  on ubuntu and macOS (Release).
- `build_windows` (Release): the Windows build and tests, driven by the
  `ci-win-rel` workflow preset, which chains configure, build, and the CTest
  run over the C++ cases and the pilot suite.

The heavy set runs on the nightly schedule only:

- `build_windows` (Debug): the second Windows configuration.
- `nouse_install`: the `setup.py install` packaging path.
- the ASAN/UBSAN sanitizer build (`-DUSE_SANITIZER=ON` over the gtest suite).
- `profiling`: the benchmark suite.
- the Windows portable artifact.

The nightly run is the superset: the fast set plus these extras.

Compiler caches (`ccache` on Linux and macOS, `sccache` for MSVC on Windows)
are saved by pushes to `master` and by nightly runs, and restored by pull
requests. A pull request on a non-default base branch cannot read them and runs
cold.

### Skipping in a pull request

- A pull request skips the heavy jobs when it carries the `skip-ci` label or a
  repository member writes `[skip-ci]` alone on a line of its description or a
  comment. A documentation-only pull request (only `doc/**`, `*.md`, `*.rst`,
  or `contrib/prompt/**`) skips them automatically.
- A pull request that touches no C++ or build file skips the Windows build but
  still runs the Python build and lint.

### Repository variables

These variables tune the workflows. Set them as repository variables (under
Settings, then Secrets and variables, then Actions, then the Variables tab).
Each is read with a default, so an unset variable keeps the default behavior.

- `MMGH_NIGHTLY`: set to `enable` to let the nightly schedule run its jobs.
  Unset, every scheduled run is skipped.
- `MMGH_PUSH_RUN_BRANCH`: which branches run the fast set on a `push`. Use `*`
  for all branches, or a branch name (matched as a substring). Unset, a push
  runs nothing.
- `MMGH_FORCE_PROFILE`, `MMGH_FORCE_NOUSE_INSTALL`, `MMGH_FORCE_SANITIZER`: set
  any to `enable` to force that nightly-only job to run on any event, so it can
  be exercised from a pull request.
- `MMGH_TIMEOUT_BUILD` (45), `MMGH_TIMEOUT_LINT` (45),
  `MMGH_TIMEOUT_STANDALONE_BUFFER` (10), `MMGH_TIMEOUT_NOUSE_INSTALL` (30),
  `MMGH_TIMEOUT_PROFILE` (30): per-job `timeout-minutes`. The number is the
  default used when the variable is unset. `MMGH_TIMEOUT_BUILD` is shared by
  the ubuntu, macOS, and sanitizer builds (default 45) and the Windows build,
  which defaults to 60 because its vcpkg plus MSVC compile runs slower than the
  others.
- `MMGH_REMIND_REPOSITORY` (`solvcon/solvcon`): the repository whose monthly
  `Update Contributors` cron files its reminder issue. The job also requires a
  non-fork repository, so a fork never files one whatever this is set to.

### Behavior on a forked repository

- A fork inherits neither these variables nor the secrets, so `MMGH_NIGHTLY`
  and `MMGH_PUSH_RUN_BRANCH` are unset there. By default only `pull_request`
  events run, so the fast set runs on a pull request opened in the fork, while
  pushes and the nightly schedule run nothing until the variables are set.
- GitHub creates a run entry for the nightly cron in every fork that has
  Actions enabled, and offers no way to suppress it. `MMGH_NIGHTLY` gates
  `check_skip` as well as the heavy jobs, so that entry holds no job and
  takes no runner.
- Scheduled workflows run only on the default branch, and GitHub keeps Actions
  disabled on a new fork until it is enabled (a public fork's schedule also
  pauses after 60 days without activity).
- To exercise a single nightly-only job on a fork, set the matching
  `MMGH_FORCE_*` variable and open a pull request.
- A fork pull request on a non-default base branch runs cold, since it cannot
  read the warmed caches. The failure-notification job is guarded by
  `github.event.repository.fork == false`, so it never runs on a fork (and the
  email secrets it needs are unavailable there).

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
