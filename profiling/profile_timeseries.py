# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Time the time-series kernels against the pure-Python loops they replace, on
an hour-long log sampled at one hundred hertz.
"""

import bisect
import functools
import time

import numpy as np

import solvcon
from solvcon import timeseries as ts

HOUR_NS = 3_600 * 10**9
RATE_HZ = 100
SPAN_NS = 200_000_000


def profile_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ = solvcon.CallProfilerProbe(func.__name__)
        return func(*args, **kwargs)
    return wrapper


def make_log(nsample, rng):
    # Jittered sampling with a few repeated stamps, as a recorded log has.
    step = HOUR_NS // nsample
    jitter = rng.integers(0, step // 4, nsample, dtype='int64')
    times = (np.arange(nsample, dtype='int64') * step + jitter)
    dup = rng.random(nsample) < 0.01
    times[1:][dup[1:]] = times[:-1][dup[1:]]
    times = np.sort(times).astype('uint64')
    speed = np.cumsum(rng.normal(0.0, 0.1, nsample)) + 20.0
    brake = rng.random(nsample) < 0.9
    return times, speed, brake


@profile_function
def py_merge_sorted_unique(a, b):
    return sorted(set(a.tolist()) | set(b.tolist()))


@profile_function
def py_dedup_last(times, values):
    out_t, out_v = [], []
    for t, v in zip(times.tolist(), values.tolist()):
        if out_t and out_t[-1] == t:
            out_v[-1] = v
        else:
            out_t.append(t)
            out_v.append(v)
    return out_t, out_v


@profile_function
def py_deriv(times, values):
    tl, vl = times.tolist(), values.tolist()
    return tl[1:], [(vl[i] - vl[i - 1]) / (tl[i] - tl[i - 1])
                    for i in range(1, len(tl))]


@profile_function
def py_movavg(times, values, span):
    tl, vl = times.tolist(), values.tolist()
    out = []
    for i, t in enumerate(tl):
        lo = bisect.bisect_right(tl, t - span)
        hi = bisect.bisect_right(tl, t)
        out.append(sum(vl[lo:hi]) / (hi - lo))
    return tl, out


@profile_function
def py_held(times, values, span):
    tl, vl = times.tolist(), values.tolist()
    out = []
    for t in tl:
        lo = bisect.bisect_right(tl, t - span)
        hi = bisect.bisect_right(tl, t)
        out.append(lo > 0 and vl[lo - 1] and all(vl[lo:hi]))
    return tl, out


@profile_function
def py_true_intervals(times, values):
    tl, vl = times.tolist(), values.tolist()
    runs, start = [], None
    for i, (t, v) in enumerate(zip(tl, vl)):
        if i + 1 < len(tl) and tl[i + 1] == t:
            continue
        if v and start is None:
            start = t
        elif not v and start is not None:
            runs.append((start, t, t - start))
            start = None
    if start is not None:
        runs.append((start, tl[-1], tl[-1] - start))
    return runs


@profile_function
def py_searchsorted_zoh(times, grid):
    tl = times.tolist()
    return [bisect.bisect_right(tl, g) - 1 for g in grid.tolist()]


@profile_function
def sa_merge_sorted_unique(a, b):
    return ts.merge_sorted_unique(a, b)


@profile_function
def sa_dedup_last(times, values):
    return ts.dedup_last(times, values)


@profile_function
def sa_deriv(times, values):
    return ts.deriv(times, values)


@profile_function
def sa_movavg(times, values, span):
    return ts.movavg(times, values, span)


@profile_function
def sa_held(times, values, span):
    return ts.held(times, values, span)


@profile_function
def sa_true_intervals(times, values):
    return ts.true_intervals(times, values)


@profile_function
def sa_searchsorted_zoh(times, grid):
    return times.searchsorted(grid, side='right')


def run(nsample, it):
    rng = np.random.default_rng(20260821)
    ntimes, nspeed, nbrake = make_log(nsample, rng)
    ngrid = np.sort(rng.integers(0, HOUR_NS, nsample, dtype='uint64'))
    times = solvcon.SimpleArrayUint64(array=ntimes)
    speed = solvcon.SimpleArrayFloat64(array=nspeed)
    brake = solvcon.SimpleArrayBool(array=nbrake)
    grid = solvcon.SimpleArrayUint64(array=ngrid)
    times_u, speed_u = ts.dedup_last(times, speed)
    ntimes_u, nspeed_u = times_u.ndarray, speed_u.ndarray

    cases = [
        ('merge_sorted_unique', (ntimes, ngrid), (times, grid)),
        ('dedup_last', (ntimes, nspeed), (times, speed)),
        ('deriv', (ntimes_u, nspeed_u), (times_u, speed_u)),
        ('movavg', (ntimes, nspeed, SPAN_NS), (times, speed, SPAN_NS)),
        ('held', (ntimes, nbrake, SPAN_NS), (times, brake, SPAN_NS)),
        ('true_intervals', (ntimes, nbrake), (times, brake)),
        ('searchsorted_zoh', (ntimes, ngrid), (times, grid)),
    ]

    print(f"\n# N = {nsample} samples over one hour, {it} iterations")
    print("| {:20s} | {:>14s} | {:>14s} | {:>9s} |".format(
        'kernel', 'python (ms)', 'solvcon (ms)', 'speedup'))
    print("| {:20s} | {:>14s} | {:>14s} | {:>9s} |".format(
        '-' * 20, '-' * 14, '-' * 14, '-' * 9))
    for name, pyargs, saargs in cases:
        pyfunc = globals()['py_' + name]
        safunc = globals()['sa_' + name]
        solvcon.call_profiler.reset()
        for _ in range(it):
            pyfunc(*pyargs)
            safunc(*saargs)
        per_call = {r['name']: r['total_time'] / r['count']
                    for r in solvcon.call_profiler.result()['children']}
        pytime = per_call['py_' + name]
        satime = per_call['sa_' + name]
        print("| {:20s} | {:14.3f} | {:14.3f} | {:8.1f}x |".format(
            name, pytime, satime, pytime / satime))


def main():
    start = time.perf_counter()
    run(HOUR_NS // 10**9 * RATE_HZ, it=3)
    print(f"\ntotal wall time: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
