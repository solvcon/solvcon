---
name: cmake-user-presets
description: Install or refresh CMakeUserPresets.json from the checked-in template, substituting the scdv prefix so the `scdv-rel` preset carries a real path. Use in a fresh checkout or worktree, when `scdv-rel` is missing from `cmake --list-presets`, or after the template changes.
---

# CMake User Presets (solvcon)

`CMakeUserPresets.json` is gitignored, so every fresh checkout and every new
worktree starts without it and without the `scdv-rel` preset. The `scdv`
templates in `contrib/cmake/` name the prefix once, as `$penv{SCDV_USRDIR}` in
the preset's `environment` map, and the three cache variables read it back
from there as `$env{SCDV_USRDIR}`. Installing one is a copy with that single
occurrence replaced by the prefix it names.

Substituting rather than leaving the expansion in place is the whole point.
An IDE started from the desktop inherits the session environment, not the
shell where an scdv was activated, so `$penv{SCDV_USRDIR}` would expand to
nothing there and the preset would configure against empty paths. A real path
in the installed file works in a shell, in a desktop IDE, and in an agent
session alike.

## When to use

A checkout has no `CMakeUserPresets.json`, `cmake --list-presets` does not
offer `scdv-rel`, an IDE cannot find Qt6 or pybind11, or the template in
`contrib/cmake/` changed and the installed copy is stale. The `worktree` skill
calls this after creating a worktree.

## 1. Resolve the scdv prefix

The substitution needs a prefix, so resolve one before touching the file:
`$SCDV_USRDIR` when a shell has an scdv activated, `$SCDV_BASE/usr` when the
user names an environment, or the `usr` directory of a build under
`~/var/scdv/`. Do not invent a path and do not fall back to a system prefix.

Carry the result in a shell variable, because only the first of those three is
already one:

```bash
PREFIX=${SCDV_USRDIR:-${SCDV_BASE:?resolve an scdv prefix first}/usr}
```

Report which environment was used. A machine often holds several, and the
installed file pins the one that was resolved.

Stop and ask when nothing resolves. An unsubstituted file is worse than no
file: CMake reports missing packages rather than a missing environment.

## 2. Install or refresh

Work from the repository root (`git rev-parse --show-toplevel`), which in a
worktree is the worktree, not the main checkout. Pick the template for the
host: `CMakeUserPresets.scdv.json` on Linux and macOS,
`CMakeUserPresets.win-scdv.json` on Windows. The `example` files beside them
are for a prefix that is not an scdv and are hand-edited, not installed from
here.

```bash
sed "s|\$penv{SCDV_USRDIR}|${PREFIX}|g" \
    contrib/cmake/CMakeUserPresets.scdv.json > CMakeUserPresets.json
```

```powershell
(Get-Content contrib\cmake\CMakeUserPresets.win-scdv.json -Raw).Replace(
    '$penv{SCDV_USRDIR}', $prefix) |
    Set-Content CMakeUserPresets.json
```

When the file already exists, render to a scratch path and compare. Identical
output means the installed copy is current, so say so and stop. A difference
is either a stale template or the user's own presets, which the diff tells
apart: show it and ask before overwriting.

Never commit the result and never move a machine path into
`CMakePresets.json`.

## 3. Verify

The `SCDV_USRDIR` entry has to name a directory that exists. Checking for a
leftover `penv{` is not enough: an empty `PREFIX` substitutes cleanly and
leaves `"SCDV_USRDIR": ""`, which passes that grep and then fails a configure
with missing packages instead of a missing environment.

```bash
test -d "$(sed -n 's/.*"SCDV_USRDIR": "\(.*\)".*/\1/p' CMakeUserPresets.json)"
```

The remaining `$env{SCDV_USRDIR}` in the cache variables is correct, and
resolves from the `environment` map above them. Then `cmake --list-presets`
has to show `scdv-rel`, and `cmake --list-presets=build` has to show
`scdv-rel`, `scdv-rel-module`, and `scdv-rel-gtest`. Report the preset names
and the prefix they carry; do not configure or build unless the user asked for
it.

Both listings have to work with the scdv deactivated, which is what an IDE
sees. `env -u SCDV_USRDIR cmake --list-presets` is the cheap way to prove it.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
