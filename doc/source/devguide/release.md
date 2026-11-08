# Release Process

A solvcon release is a self-contained pilot application for Apple Silicon
macOS, packed in a DMG and attached to a GitHub release.  This page describes
what a release contains, how a maintainer cuts one, and how each stage can be
reproduced locally.

## What a release is

- A release is marked by an annotated git tag `vX.Y.Z` on a commit of
  `master`.
- The DMG is named `pilot-<version>-<platform>.dmg`, where `<version>` is the
  tag and `<platform>` is the operating system and the architecture, currently
  `macos-arm64`.  Tag `v0.2.0` gives `pilot-v0.2.0-macos-arm64.dmg`.
- The release attaches four files that share that name: the DMG, its
  `.dmg.sha256`, its `.manifest.txt` (see [Reproducibility](#reproducibility)),
  and the `.build-info.md` that also opens the release notes.
- The app reports its version.  `CFBundleVersion` in `Info.plist` and
  `solvcon.__version__` carry the tag, and `CFBundleShortVersionString` carries
  its `X.Y.Z`.  A build that is not on a tag carries the `git describe` string,
  e.g. `v0.1.0-12-gabc1234`, and a source tree reports `0.0.0+dev`.
- The release notes record the build host macOS, Xcode, Apple clang, the macOS
  SDK, libc++, CMake, pybind11, and the bundled Python, Qt, PySide6, NumPy, and
  Matplotlib, followed by the changes since the previous tag.

The app is ad-hoc signed and not notarized.  macOS blocks a downloaded copy
until its quarantine flag is cleared, which the release notes explain:

```sh
xattr -d com.apple.quarantine pilot-v0.2.0-macos-arm64.dmg
```

## The bundle identity

`cpp/binary/pilot/Info.plist.in` is the `Info.plist` template of the pilot app.
CMake fills it at configure time:

| Key | Value |
|---|---|
| `CFBundleIdentifier` | `net.solvcon.pilot` |
| `CFBundleName`, `CFBundleDisplayName` | `solvcon Pilot` |
| `CFBundleVersion` | the `SOLVCON_VERSION` cache variable, or `0.0.0+dev` |
| `CFBundleShortVersionString` | the `X.Y.Z` of `SOLVCON_VERSION`, or `0.0.0` |
| `LSMinimumSystemVersion` | `CMAKE_OSX_DEPLOYMENT_TARGET`, or the SDK version |
| `NSHumanReadableCopyright` | the solvcon copyright line |

A release builds with `CMAKE_OSX_DEPLOYMENT_TARGET=26.0`, so it runs on macOS
26 and later.  The bundle step then adds `LSEnvironment` with
`PYTHONNOUSERSITE=1`.

## The stages

The workflow `.github/workflows/release.yml` only calls `make`.  Each target
runs a subcommand of `contrib/bundle/bundle-with-aqt.sh`, so every stage runs
the same way on a developer's Mac.

| Stage | Target | What it does |
|---|---|---|
| Tag check | `make bundle-check-tag` | Checks the tag is an annotated `vX.Y.Z` on the checked-out commit and on `master` |
| Dependencies | `make bundle-prepare` | Installs the pinned Python, Qt, PySide6, and Python packages under `build/bundle-deps` |
| Clean build and pack | `make bundle` | Removes `build/relbundle`, builds the pilot in Release, bundles it, and writes the DMG and its checksum |
| Verification | `make bundle-test` | Checks the app is self-contained, launches it from the DMG, and checks the embedded version matches the DMG name |
| Unboxing test | `make bundle-unbox-test` | Runs `tests/gui/` inside the bundled interpreter, Qt, and solvcon |
| Build information | `make bundle-info` | Writes `<dmg>.build-info.md` |
| Manifest | `make bundle-manifest` | Writes `<dmg>.manifest.txt` |

Every target names its files by `RELEASE_VERSION`, which defaults to `git
describe --tags --always --dirty`, and writes into `RELEASE_OUTPUT` (default:
`build/`).

The unboxing test copies the app out of the DMG and installs pytest into a
scratch directory outside it, so the shipped app carries no test tooling.  It
fails unless the tests import the solvcon inside the app.

`make bundle-info` describes the machine it runs on, so run it on the host that
built the DMG.

## Cutting a release

1. Make sure the `master` CI is green on the commit to release.
2. Run the `release` workflow by hand on `master` (Actions, release, Run
   workflow).  This is a dry run: it builds, packs, and tests, and keeps the
   files as a workflow artifact without touching the releases.
3. Tag the commit and push the tag to the solvcon repository:

   ```sh
   git tag -a v0.2.0 -m "solvcon v0.2.0"
   git push upstream v0.2.0
   ```

4. The tag push runs the workflow, which drafts the GitHub release with the
   four files attached.
5. Review the draft: the version, the build information, the generated change
   list, and the checksum.  Download the DMG and open the app.
6. Publish the draft.

If any stage fails, nothing is released.  Delete the tag, fix the problem on
`master`, and tag again:

```sh
git push upstream :refs/tags/v0.2.0
git tag -d v0.2.0
```

Delete a draft that a failed attempt left behind before pushing the tag again;
the workflow does not overwrite an existing release.

## Building a release locally

The same targets run on an Apple Silicon Mac with an Xcode that supports the
deployment target:

```sh
make bundle-prepare
make bundle RELEASE_VERSION=v0.2.0
make bundle-test RELEASE_VERSION=v0.2.0
make bundle-unbox-test RELEASE_VERSION=v0.2.0
make bundle-info RELEASE_VERSION=v0.2.0
make bundle-manifest RELEASE_VERSION=v0.2.0
```

`make bundle-check-tag` fetches the tag and `master` from `RELEASE_REMOTE`
(default: `origin`).  In a fork checkout, point it at the solvcon repository,
e.g. `make bundle-check-tag RELEASE_VERSION=v0.2.0 RELEASE_REMOTE=upstream`.

## Reproducibility

Two builds of one commit with the same toolchain bundle the same files.  The
build supports this by:

- Pinning every input: the Python archive by checksum, Qt, and every Python
  package in `contrib/bundle/sdk-constraints.txt`, and Xcode in the workflow.
- Building from an empty tree.
- Setting `SOURCE_DATE_EPOCH` to the commit time, `ZERO_AR_DATE=1`, and
  `-ffile-prefix-map` so no clock or checkout path reaches the binaries.
- Shipping hash-based `.pyc` files and dating every file in the app by the
  commit.

The manifest lists the SHA-256 of every file in the app, so comparing two
builds is a `diff` of their manifests.  The version lives in `Info.plist` and
`solvcon/_version.py`, so two builds named by different versions differ in
those files, their bytecode, and the signature of the app; everything else
must match.  The DMG itself is not compared, because `hdiutil` writes volume
identifiers that change on every run.

## Future work

- Sign with a Developer ID and notarize, so the app opens without clearing the
  quarantine flag.
- Release for more platforms.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
