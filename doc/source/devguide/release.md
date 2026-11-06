# Release

A release ships the pilot as a self-contained macOS application in a DMG. The
`release` workflow (`.github/workflows/release.yml`) builds the DMG, tests it
on a clean runner, and drafts a GitHub release from it. The steps call the
`make bundle*` targets, which run `contrib/bundle/bundle-with-aqt.sh`, so every
stage also runs by hand on a Mac.

The DMG targets Apple Silicon only, because the bundler supports nothing else.
It is the only release artifact so far.

## Cutting a release

1. Merge everything the release needs into `master`.
2. Tag the commit with the version and push the tag:

   ```sh
   git tag v1.2.3
   git push upstream v1.2.3
   ```

3. Wait for the `release` workflow to finish. A tag push runs every stage and
   then drafts a release named after the tag, with the DMG and its SHA-256
   checksum attached and notes generated from the merged pull requests.
4. Review the draft on the Releases page: the notes, the assets, and the
   workflow run. Publish it when it is right, or delete it.

The workflow never publishes on its own. A draft stays private to the
maintainers until someone publishes it.

## Triggers

| Trigger | Builds | Publishes |
|---|---|---|
| Push of a tag `vX.Y.Z` | the tag | a draft release |
| Push of a tag `vX.Y.Z-<suffix>`, e.g. `v1.2.3-rc1` | the tag | a draft release |
| Manual run (`workflow_dispatch`) | the `ref` input, or the run's branch | nothing |

The tag patterns are `v[0-9]+.[0-9]+.[0-9]+` and `v[0-9]+.[0-9]+.[0-9]+-*`. A
tag of any other shape starts nothing.

A manual run takes two optional inputs:

- `ref`: the branch, tag, or commit to build. Empty builds the branch the run
  is started from.
- `version`: the version in the artifact name. Empty uses the short commit
  hash.

To start a manual run from the command line:

```sh
gh workflow run release.yml --ref <branch> -f ref=<commit> -f version=<name>
```

GitHub offers a manual run only for a workflow that exists on the default
branch. On a fork whose default branch lacks `release.yml`, push a tag instead,
or change the fork's default branch.

Pass a commit hash as `ref` when the branch may move during the run. The
`build` and `unbox_test` jobs each check out `ref` when they start, so a push
between them would otherwise test a different commit than the one built.

## Stages

| Job | Step | What it runs |
|---|---|---|
| `build` | `setup` | `make bundle-prepare` and `make bundle-precheck` |
| `build` | `build` | `make bundle`: builds the pilot and makes `dist/pilot.dmg` |
| `build` | `package` | names the DMG and writes its SHA-256 checksum |
| `unbox_test` | `unbox test` | checks the checksum and runs `make bundle-test` |
| `unbox_test` | `gui test` | `make bundle-guitest`: runs `tests/gui` in the app |
| `publish` | `draft release` | drafts the GitHub release (tag pushes only) |

The DMG is named `solvcon-pilot-<version>-macos-<arch>.dmg`, for example
`solvcon-pilot-v1.2.3-macos-arm64.dmg`. The version is the tag for a tag push,
and the `version` input or the short commit hash for a manual run.

`make bundle-prepare` downloads pinned, checksum-verified builds of Python, Qt,
and the Python packages into `build/bundle-deps` (see
`contrib/bundle/prepare-with-aqt.sh`). The versions are pinned in that script.
`make bundle-precheck` checks that the tools and modules the bundler needs are
there.

The `unbox_test` job runs on a fresh runner and prepares nothing. It sees the
DMG the way a user's machine does, so a library that the app takes from the
build host instead of its bundle fails there. It checks out only the
`Makefile`, the bundler, and the GUI tests.

`make bundle-guitest` runs the GUI tests in the Python interpreter inside the
app. pytest is not part of the release, so the bundler installs it outside the
app with the host's `python3` and puts it on `PYTHONPATH`. The bundler clears
the environment and runs from a scratch directory, so the tests import the
bundled solvcon, not the checkout. The `gui test` step runs even when
`unbox test` fails, so each check reports on its own.

## Running the stages by hand

Run the targets from the repository root on an Apple Silicon Mac:

```sh
make bundle-prepare
make bundle-precheck
SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" make bundle RELEASE_OUTPUT=dist
make bundle-test RELEASE_ARTIFACT=dist/pilot.dmg
make bundle-guitest RELEASE_ARTIFACT=dist/pilot.dmg
```

`make bundle-prepare` deletes `build/bundle-deps` and downloads everything
anew, so run it once and skip it while `make bundle-precheck` passes.

`RELEASE_OUTPUT` is the directory `make bundle` writes `pilot.dmg` into
(default: `build`). `RELEASE_ARTIFACT` is the DMG that `make bundle-test` and
`make bundle-guitest` check (default: `$(RELEASE_OUTPUT)/pilot.dmg`).
`SOURCE_DATE_EPOCH` matters only to compare builds; see Reproducibility below.

`make bundle-guitest` installs the pytest version in `GUITEST_PYTEST` (default:
`pytest==9.1.1`). The `pip` of the host's `python3` must be recent enough to
install it for the app's Python. The pip 21 of the Xcode `/usr/bin/python3` is
not, so put a newer `python3` first on `PATH`.

The smoke launch in `make bundle` and `make bundle-test` starts the pilot for a
few seconds, so a window opens on the desktop.

## Notes

### Signing

The app carries an ad-hoc signature, not a Developer ID, and it is not
notarized. `spctl` rejects it during the unbox test, and the test reports that
as expected. Gatekeeper on a user's Mac blocks the first launch of a
downloaded copy. The user has to approve it in System Settings, under Privacy
& Security.

### Reproducibility

Two builds of one commit produce different DMG checksums. `hdiutil create`
writes its own volume UUID and timestamps into every image, and the free space
it adds varies between runs.

The application inside the DMG is the part to compare. Mount both images and
compare the file lists and the hash of every file. In two builds of one commit
on CI, every file but the `pilot` binary matched.

The `pilot` binary differed because `rcc` embeds the modification time of each
Qt resource: the checkout time for `resources/pilot/icon.ico`, and the build
time for the compiled shaders. The linker UUID and the signature follow from
those bytes. The `build` step sets `SOURCE_DATE_EPOCH` to the commit time,
which `rcc` uses in place of the file times.

### Runner capacity

The jobs run on GitHub-hosted macOS arm64 runners. When none is free, a job
waits and GitHub cancels it with "The job was not acquired by Runner of type
hosted even after multiple attempts". The job has no steps, and the run
fails. Run it again later.

### Time limits

The `build` job has a limit of 90 minutes, which the repository variable
`SCGH_TIMEOUT_RELEASE` overrides (see {doc}`/devguide/testing`). The
`unbox_test` job has 30 minutes and `publish` has 10.

### Concurrency

Runs of one ref share a concurrency group and never cancel each other. A
second run of the same branch or tag waits for the first to finish.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
