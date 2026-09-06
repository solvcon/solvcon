---
name: ide-user-presets
description: Install or refresh an IDE's CMakeUserPresets.json with contrib/cmake/install-user-presets.py, which substitutes the scdv prefix so the `ide-scdv-reldbg` preset carries a real path. Use when setting an IDE up on a checkout, when an IDE cannot find Qt6 or pybind11, or after the template changes. Not a build step; agents build with `make`.
---

# IDE User Presets (solvcon)

An IDE uses `CMakeUserPresets.json`. It is gitignored, it holds one machine's
literal paths, and its presets are named `ide-*` to keep them apart from the
shared presets in `CMakePresets.json`. Install it when an IDE needs it, never
as preparation for a build: an agent builds with `make`, which selects a
checked-in preset and layers the activated environment on top.

`contrib/cmake/install-user-presets.py` does the whole job: it resolves the
prefix, picks the template for the host, substitutes, and verifies. Read this
page for what the script decides on your behalf and when to stop and ask.

The `scdv` templates in `contrib/cmake/` name the prefix once, as
`$penv{SCDV_USRDIR}` in the preset's `environment` map, and the three cache
variables read it back as `$env{SCDV_USRDIR}`. Installing is a copy with that
single occurrence replaced by the prefix it names.

Substituting rather than leaving the expansion in place is the whole point.
An IDE started from the desktop inherits the session environment, not the
shell where an scdv was activated, so `$penv{SCDV_USRDIR}` would expand to
nothing there and the preset would configure against empty paths.

## When to use

The user is setting an IDE up on a checkout, an IDE cannot find Qt6 or
pybind11, or the template in `contrib/cmake/` changed and the installed copy
is stale. Nothing else triggers it. Entering a worktree does not, and neither
does a task that happens to build.

## Install or refresh

```bash
contrib/cmake/install-user-presets.py
```

The script installs into the checkout it lives in, which in a worktree is the
worktree, so it can be run from any directory. It writes the file, then
reports the preset names and the prefix they carry. That is the whole
procedure; the sections below cover the two decisions it hands back.

Never commit the result and never move a machine path into
`CMakePresets.json`.

### The prefix

The script resolves `$SCDV_USRDIR`, then `$SCDV_BASE/usr`, and takes
`--prefix` over both. Pass `--prefix` when the shell has no scdv activated
but the user names an environment, or to point at the `usr` directory of a
build under `~/var/scdv/`.

Nothing else is a candidate. Do not invent a path and do not fall back to a
system prefix. When the script reports that no prefix resolved, stop and ask:
an unsubstituted file is worse than no file, because CMake then reports
missing packages rather than a missing environment.

Report which environment was used. A machine often holds several, and the
installed file pins the one that was resolved.

### An installed copy that differs

The script refuses to overwrite a file that differs from the rendered
template, and prints the diff instead. The diff tells the two causes apart: a
stale template, or the user's own presets. Show it and ask before rerunning
with `--force`.

`--check` renders and compares without writing, and exits non-zero when the
installed copy is missing or stale. Use it to answer whether a refresh is
needed.

## What the script verifies

It checks that the substituted `SCDV_USRDIR` names a directory that exists.
Checking for a leftover `penv{` would not be enough: an empty prefix
substitutes cleanly and leaves `"SCDV_USRDIR": ""`, which passes that grep and
then fails a configure with missing packages instead of a missing environment.
The remaining `$env{SCDV_USRDIR}` in the cache variables is correct, and
resolves from the `environment` map above them.

Then it runs `cmake --list-presets=configure` and `--list-presets=build` with
`SCDV_USRDIR` removed from the environment, which is what an IDE sees, and
requires `ide-scdv-reldbg` in the first and `ide-scdv-reldbg`,
`ide-scdv-reldbg-module` and `ide-scdv-reldbg-gtest` in the second.

Listing is the whole verification. Configuring or building through an `ide-*`
preset is the IDE's job, and a build of your own is `make`. Report the preset
names and the prefix they carry, then stop.

## Doing it by hand

Only when the script cannot run. Pick the template for the host,
`CMakeUserPresets.scdv.json` on Linux and macOS,
`CMakeUserPresets.win-scdv.json` on Windows, and substitute into the
repository root. The `example` files beside them are for a prefix that is not
an scdv; they are hand-edited, not installed from here.

```bash
PREFIX=${SCDV_USRDIR:-${SCDV_BASE:?resolve an scdv prefix first}/usr}
sed "s|\$penv{SCDV_USRDIR}|${PREFIX}|g" \
    contrib/cmake/CMakeUserPresets.scdv.json > CMakeUserPresets.json
```

```powershell
(Get-Content contrib\cmake\CMakeUserPresets.win-scdv.json -Raw).Replace(
    '$penv{SCDV_USRDIR}', $prefix) |
    Set-Content CMakeUserPresets.json
```

Then verify by hand what the script would have verified:

```bash
test -d "$(sed -n 's/.*"SCDV_USRDIR": "\(.*\)".*/\1/p' CMakeUserPresets.json)"
env -u SCDV_USRDIR cmake --list-presets
env -u SCDV_USRDIR cmake --list-presets=build
```

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
