# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Run pytest in parallel jobs.

Pytest doesn't support running tests in parallel natively (need
pytest-xdist plugin). This script provides a simple way to run pytest tests
in parallel without requiring additional plugins.

Job i of N runs every N-th collected test, starting from the i-th.

Usage:

    python3 contrib/pytest_parallel.py --jobs 8 [PYTEST_ARGS]
"""

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile


# Each job loads this file as a pytest plugin for the two hooks below.
def pytest_addoption(parser):
    parser.addoption('--job', help='"index/count" of this job')


def pytest_collection_modifyitems(config, items):
    index, count = map(int, config.getoption('job').split('/'))
    items[:] = items[index::count]


def main():
    parser = argparse.ArgumentParser(
        description='Run pytest in parallel jobs.', allow_abbrev=False)
    parser.add_argument('--jobs', '-j', type=int, default=os.cpu_count(),
                        help='number of jobs (default: processor count)')
    # Pass every other argument to pytest.
    args, pytest_args = parser.parse_known_args()

    here = pathlib.Path(__file__).resolve()
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        filter(None, [str(here.parent), env.get('PYTHONPATH')]))
    if sys.stdout.isatty():
        env.setdefault('FORCE_COLOR', '1')

    jobs = []
    for index in range(args.jobs):
        cmd = [sys.executable, '-m', 'pytest', '-p', here.stem,
               f'--job={index}/{args.jobs}', '-p', 'no:cacheprovider',
               *pytest_args]
        out = tempfile.TemporaryFile()
        proc = subprocess.Popen(cmd, env=env, stdout=out,
                                stderr=subprocess.STDOUT)
        jobs.append((proc, out))

    codes = []
    try:
        for index, (proc, out) in enumerate(jobs):
            codes.append(proc.wait())
            print(f'\n{"#" * 20} job {index + 1}/{args.jobs} exited '
                  f'{codes[-1]} {"#" * 20}', flush=True)
            out.seek(0)
            sys.stdout.buffer.write(out.read())
            sys.stdout.flush()
    finally:
        # An interrupted run must not leave its jobs behind.
        for proc, _ in jobs:
            proc.terminate()

    # Skip the jobs that got no test (exit 5) and return the worst code of
    # the rest. A job killed by a signal returns a negative code, so compare
    # by absolute value. If every job got no test, return 5.
    return max((c for c in codes if c != 5), key=abs, default=5)


if __name__ == '__main__':
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
