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
./build/reldbg<pyvminor>/gtests/run_gtest --gtest_filter=Suite.Test
```

where `<pyvminor>` is the active Python major and minor version, e.g. `314`.
The directory follows `CMAKE_BUILD_TYPE`; see {doc}`/start/build_solvcon`.

## Through CTest

Every suite is also registered with CTest, so `ctest` runs the C++ cases, the
Python suite, and the pilot suite from one command against a configured build
tree.  This is what an IDE drives, and it is how the C++ cases become
individually selectable.

```sh
ctest --preset dev-reldbg         # every suite
ctest --preset dev-reldbg-cpp     # the C++ cases alone
ctest --preset dev-reldbg-python  # the Python suite alone
ctest --preset dev-reldbg-pilot   # the Python suite inside the pilot binary
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
`.github/workflows/` and form two sets. Only a `nightly_` workflow runs tests
on a cron, so the name states which set a workflow belongs to. (Two crons
outside those workflows do maintenance rather than testing: `cache_cleanup`
sweeps stale caches daily, and `update_contributors` files a monthly issue.)
Each job drives the `make` targets above, so you can reproduce a failure
locally.

The fast set runs on every pull request and `master` push:

- `check_skip`: the gate that decides whether the heavy jobs run (see below).
- `lint`: `make cformat`, `cinclude`, `checkascii`, `checktws`, `checktests`,
  `flake8`, and clang-tidy on the diff, on ubuntu and macOS.
- `standalone_buffer` (in `devbuild`): the standalone buffer build on ubuntu.
- `build` (in `devbuild`): `make gtest` plus `make pytest` with Qt off and on,
  and the pilot, on ubuntu (Release) and macOS (RelWithDebInfo).
- `build_windows` (in `devbuild_windows`, Release): the Windows build and
  tests, driven by the `ci-win-rel` workflow preset, which chains configure,
  build, and the CTest run over the C++ cases and the pilot suite. Windows
  sits in its own workflow, so `nightly_devbuild` can call all of `devbuild`.
  The Windows build no longer waits on `standalone_buffer`, because a job in
  another workflow cannot be a dependency.

The heavy set runs on the cron, one workflow per concern:

- `nightly_devbuild`: the standalone buffer build, the ubuntu and macOS
  build, the pilot tests, and the lint targets, through a call into `devbuild`
  and `lint`.
- `nightly_build_windows`: the Windows build in Release and Debug, and the
  portable artifact packaged from the Release tree.
- `nightly_nouse_install`: the `setup.py install` packaging path.
- `nightly_sanitizer`: the ASAN/UBSAN build on ubuntu (`-DUSE_SANITIZER=ON`
  over the gtest suite), and the MSVC ASan build on Windows.
- `nightly_profiling`: the benchmark suite. It starts an hour after the other
  nightly workflows, so that it does not compile the same objects as the
  `devbuild` build job at the same time.

With `SCGH_PUSH_RUN_BRANCH` set, `nightly_devbuild` runs the same jobs as a
`master` push. It adds no coverage. It finds environment drift between pushes,
and it mails the failure. Three other events reach a heavy workflow:
`workflow_dispatch` on any of them, a release-tag push for
`nightly_nouse_install`, and a `SCGH_FORCE_*` variable (see below).

Only a nightly workflow mails a failure. Each `send_email_on_failure` job calls
`send_email_on_fail.yml` under `github.event_name == 'schedule'`, so a force
variable that drives one of these jobs from a pull request mails no one.

A `master` push and a nightly run save the compiler caches (`ccache` on Linux
and macOS, `sccache` on Windows). A pull request restores them, unless its base
branch is not the default one, in which case it runs cold.

### Skipping in a pull request

- A pull request skips the fast set when it carries the `skip-ci` label or a
  repository member writes `[skip-ci]` alone on a line of its description or a
  comment. A documentation-only pull request (only `doc/**`, `*.md`, `*.rst`,
  or `contrib/prompt/**`) skips it automatically. Only `devbuild`,
  `devbuild_windows`, and `lint` consult `check_skip`, so a `SCGH_FORCE_*`
  variable overrides the label.
- A pull request that touches no C++ or build file skips the Windows build but
  still runs the Python build and lint.

### Repository variables

These variables tune the workflows. Set them as repository variables (under
Settings, then Secrets and variables, then Actions, then the Variables tab).
Each is read with a default, so an unset variable keeps the default behavior.

- `SCGH_NIGHTLY`: set to `enable` to let the nightly cron run its jobs. Unset,
  the cron skips every job.
- `SCGH_PUSH_RUN_BRANCH`: which branches run the fast set on a `push`. Use `*`
  for all branches, or a branch name (matched as a substring). Unset, a push
  runs nothing.
- `SCGH_FORCE_PROFILE`, `SCGH_FORCE_NOUSE_INSTALL`, `SCGH_FORCE_SANITIZER`: set
  any to `enable` to run that nightly job on any event, so a pull request can
  exercise it. `nightly_build_windows` and `nightly_devbuild` read no such
  variable, so use `workflow_dispatch` for them. A pull request runs
  `devbuild` and `lint` in a reduced shape, with no pilot build and
  `pytest-fast` in place of `pytest`, so it does not stand in for a nightly
  run. Every nightly workflow accepts a manual run.
- `SCGH_TIMEOUT_BUILD` (45), `SCGH_TIMEOUT_LINT` (45),
  `SCGH_TIMEOUT_STANDALONE_BUFFER` (10), `SCGH_TIMEOUT_NOUSE_INSTALL` (30),
  `SCGH_TIMEOUT_PROFILE` (30): per-job `timeout-minutes`, with the default in
  parentheses. `SCGH_TIMEOUT_BUILD` covers the ubuntu and macOS builds and
  the ubuntu sanitizer at 45, and the slower Windows builds, MSVC ASan
  included, at 60.
- `SCGH_REMIND_REPOSITORY` (`solvcon/solvcon`): the repository whose monthly
  `update_contributors` cron files its reminder issue. The job also requires a
  non-fork repository, so a fork never files one whatever this is set to.

### Behavior on a forked repository

- A fork inherits neither these variables nor the secrets, so only
  `pull_request` events and a manual run start jobs there. A push and the cron
  run nothing until you set the variables.
- GitHub creates a run entry for the nightly cron in every fork that enables
  Actions, and offers no way to suppress it. `SCGH_NIGHTLY` gates every job the
  cron reaches, so the entry takes no runner and marks its jobs skipped.
- A cron workflow runs on the default branch only. GitHub also keeps Actions
  off on a new fork until someone enables it, and pauses a public fork's cron
  after 60 days without activity.
- To exercise one nightly job on a fork, set the matching `SCGH_FORCE_*`
  variable and open a pull request. `nightly_build_windows` and
  `nightly_devbuild` have no such variable; run them from the Actions tab
  instead.
- A fork pull request on a non-default base branch runs cold, because it cannot
  read the warm caches. The failure mail requires
  `github.event.repository.fork == false`, and a fork has no email secrets.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
