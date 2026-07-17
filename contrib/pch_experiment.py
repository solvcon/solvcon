#!/usr/bin/env python3
"""Measure the SOLVCON_USE_PCH build-time effect on a CI runner.

For each repetition and each PCH setting, wipe the build tree, configure
with BUILD_QT=ON and no compiler cache, then time a from-scratch build of
the ``_solvcon_py`` target (which compiles ``solvcon_primary``, the only
target the precompiled headers attach to). Wall time comes from a
monotonic clock; CPU time (user + system, including every compiler child)
comes from ``getrusage(RUSAGE_CHILDREN)``, the portable equivalent of
``/usr/bin/time`` that works the same on Linux and macOS.

The point is a clean A/B: the two configurations differ only in the PCH,
so the delta is the PCH effect and nothing else.
"""

import os
import re
import resource
import shutil
import subprocess
import sys
import time

REPS = int(os.environ.get("PCH_REPS", "3"))
PARALLEL = os.environ.get("PCH_PARALLEL", "2")
TARGET = "_solvcon_py"

_vi = sys.version_info
BUILD_PATH = f"build/rel{_vi.major}{_vi.minor}"
PYTHON_EXE = shutil.which("python3")


def configure(pch):
    """Reconfigure from scratch and confirm the effective PCH state."""
    if os.path.isdir("build"):
        shutil.rmtree("build")
    cmake_args = f"-DPYTHON_EXECUTABLE={PYTHON_EXE} -DSOLVCON_USE_PCH={pch}"
    proc = subprocess.run(
        ["make", "cmake", "VERBOSE=0", "USE_CLANG_TIDY=OFF", "BUILD_QT=ON",
         "CMAKE_BUILD_TYPE=Release", f"CMAKE_ARGS={cmake_args}"],
        capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        sys.stderr.write(log)
        raise SystemExit(f"configure failed for PCH={pch}")
    match = re.search(r"SOLVCON_USE_PCH:\s*(\w+)", log)
    effective = match.group(1) if match else "UNKNOWN"
    if effective != pch:
        sys.stderr.write(log)
        raise SystemExit(
            f"PCH gate mismatch: requested {pch}, effective {effective}")
    return effective


def timed_build():
    """Build the target once, returning (wall_seconds, cpu_seconds)."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    proc = subprocess.run(
        ["cmake", "--build", BUILD_PATH, "--target", TARGET,
         "--parallel", PARALLEL])
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = ((after.ru_utime - before.ru_utime)
           + (after.ru_stime - before.ru_stime))
    if proc.returncode != 0:
        raise SystemExit("build failed")
    return wall, cpu


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    rows = []
    for rep in range(1, REPS + 1):
        for pch in ("ON", "OFF"):
            configure(pch)
            wall, cpu = timed_build()
            rows.append((rep, pch, wall, cpu))
            print(f"REP {rep} PCH {pch}: wall={wall:.1f}s cpu={cpu:.1f}s",
                  flush=True)

    on_wall = [w for _, p, w, _ in rows if p == "ON"]
    off_wall = [w for _, p, w, _ in rows if p == "OFF"]
    on_cpu = [c for _, p, _, c in rows if p == "ON"]
    off_cpu = [c for _, p, _, c in rows if p == "OFF"]

    out = [
        "| rep | pch | wall_s | cpu_s |",
        "| --- | --- | ---: | ---: |",
    ]
    for rep, pch, wall, cpu in rows:
        out.append(f"| {rep} | {pch} | {wall:.1f} | {cpu:.1f} |")
    out.append("")
    out.append(f"platform: {sys.platform}, python {_vi.major}.{_vi.minor}, "
               f"parallel {PARALLEL}, reps {REPS}")
    out.append(f"mean wall ON  {mean(on_wall):.1f}s  "
               f"OFF {mean(off_wall):.1f}s")
    out.append(f"mean cpu  ON  {mean(on_cpu):.1f}s  "
               f"OFF {mean(off_cpu):.1f}s")
    if mean(off_cpu):
        cpu_cut = (mean(off_cpu) - mean(on_cpu)) / mean(off_cpu) * 100.0
        out.append(f"cpu reduction with PCH: {cpu_cut:.1f}%")
    if mean(off_wall):
        wall_cut = (mean(off_wall) - mean(on_wall)) / mean(off_wall) * 100.0
        out.append(f"wall reduction with PCH: {wall_cut:.1f}%")

    report = "\n".join(out)
    print("\n" + report, flush=True)
    with open("pch_experiment_result.md", "w") as handle:
        handle.write(report + "\n")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(report + "\n")


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
