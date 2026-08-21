# Time Series

`solvcon.timeseries` holds kernels over recorded time series. A series is a
pair of one-dimensional arrays of the same length. The first holds the
timestamps in integer nanoseconds as a non-decreasing `SimpleArrayUint64`;
the second holds the values sampled at them as a SimpleArray. Each kernel
reads its input and returns new arrays.
The C++ kernels live in `cpp/solvcon/timeseries/` in the namespace
`solvcon::timeseries`.

A kernel raises `TypeError` when `times` is not a `SimpleArrayUint64` or
`values` is a class the kernel does not take. It raises `ValueError` for an
array that is not one-dimensional or carries ghost elements, for arrays of
different length, or for a decreasing timestamp. Sample lookup
on the timestamps is `searchsorted`, which
{doc}`the reduce page <../buffer/reduce>` describes.

## Repeated Timestamps

A recorded log can stamp two messages in the same nanosecond, so a repeated
timestamp is valid input. A group of equal timestamps resolves to its last
sample. `deriv` is the exception: a zero time step has no derivative, so
`deriv` raises `ValueError`. `dedup_last` collapses a repeat, so its result
is valid input to `deriv`.

## The `merge_sorted_unique` Function

`merge_sorted_unique(*arrays)` merges sorted `SimpleArrayUint64` timestamp
arrays into one sorted array that holds every distinct timestamp once. This
is the union time grid that compares two signals sampled at different rates:

```python
from solvcon import timeseries as ts

t1 = solvcon.SimpleArrayUint64(array=np.array([0, 10, 20], dtype='uint64'))
t2 = solvcon.SimpleArrayUint64(array=np.array([5, 10, 10], dtype='uint64'))
assert ts.merge_sorted_unique(t1, t2).ndarray.tolist() == [0, 5, 10, 20]
```

No argument, or only empty arrays, gives an empty result. Every argument must
be a `SimpleArrayUint64`; another class raises `TypeError`, and a
`ValueError` names the offending array by its position.

## The `dedup_last` Function

`dedup_last(times, values)` keeps the last sample of every group of equal
timestamps, so the result timestamps are strictly increasing:

```python
times = solvcon.SimpleArrayUint64(array=np.array([0, 1, 1, 2], dtype='uint64'))
speed = solvcon.SimpleArrayFloat64(array=np.array([0.5, 1.0, 1.5, 2.0]))
t_u, speed_u = ts.dedup_last(times, speed)
assert t_u.ndarray.tolist() == [0, 1, 2]
assert speed_u.ndarray.tolist() == [0.5, 1.5, 2.0]
```

The kernel takes every typed SimpleArray class, and the result values keep
the class of the input values. A series without a repeat comes back as a
copy.

## The `deriv` Function

`deriv(times, values)` differentiates a series by the backward difference
`(x_i - x_{i-1}) / (t_i - t_{i-1})` and returns `(times[1:], derivatives)`:

```python
times = solvcon.SimpleArrayUint64(array=np.array([0, 10, 30], dtype='uint64'))
speed = solvcon.SimpleArrayFloat64(array=np.array([1.0, 3.0, 2.0]))
t_acc, acc = ts.deriv(times, speed)
assert t_acc.ndarray.tolist() == [10, 30]
assert acc.ndarray.tolist() == [0.2, -0.05]  # (3.0-1.0)/10 and (2.0-3.0)/20
```

The first sample has no predecessor, so `n` samples give `n - 1` derivatives
and fewer than two samples give empty arrays. The output timestamps are
strictly increasing, so a second `deriv` chains directly. The kernel takes
the real-number classes only; `SimpleArrayBool` and the complex classes raise
`TypeError`. A `SimpleArrayFloat32` input gives `SimpleArrayFloat32`
derivatives, and every other class gives `SimpleArrayFloat64`.

## Trailing Windows

`movavg` and `held` answer at every timestamp of the series from the
trailing half-open window `(t - span, t]`. `span` is a length in integer
nanoseconds and must be positive; a zero `span` raises `ValueError`. A
`span` wider than the whole log is valid. Every sample of a group of equal
timestamps is in the window together and gets the same answer. Both kernels
return the tuple `(times, results)` with the input timestamps, so a chain of
kernels keeps its length and time base.

## The `movavg` Function

`movavg(times, values, span)` takes the arithmetic mean of the samples in
the window at each timestamp, which smooths a noisy signal:

```python
times = solvcon.SimpleArrayUint64(
    array=np.array([0, 10, 20, 30], dtype='uint64'))
noisy = solvcon.SimpleArrayFloat64(array=np.array([1.0, 5.0, 1.0, 5.0]))
t_s, smooth = ts.movavg(times, noisy, span=20)
assert t_s.ndarray.tolist() == [0, 10, 20, 30]
# The window at t=20 is (0, 20] and averages 5.0 and 1.0.
assert smooth.ndarray.tolist() == [1.0, 3.0, 3.0, 3.0]
```

The kernel takes the real-number classes only, and the result class follows
the same rule as `deriv`. The sweep carries one running sum, so one
non-finite sample makes every later mean NaN; drop such a sample first.

## The `held` Function

`held(times, values, span)` reports whether a `SimpleArrayBool` series was
true over the whole window `(t - span, t]` at every timestamp and returns a
`SimpleArrayBool`. The boundary sample, the last one stamped at or before
`t - span`, must be true as well. No boundary sample exists over the first
`span` of the log, so the answer there is false:

```python
times = solvcon.SimpleArrayUint64(
    array=np.array([0, 10, 20, 30, 40], dtype='uint64'))
brake = solvcon.SimpleArrayBool(
    array=np.array([True, True, False, True, True]))
t_h, brake_held = ts.held(times, brake, span=10)

# t=0: no boundary sample, answer is False
# t=10: boundary at 0 is True, window (0, 10] holds True, answer is True
# t=20: window (10, 20] holds the False at 20, answer is False
# t=30: boundary at 20 is False, answer is False
# t=40: boundary at 30 is True, window (30, 40] holds True, answer is True
assert t_h.ndarray.tolist() == [0, 10, 20, 30, 40]
assert brake_held.ndarray.tolist() == [False, True, False, False, True]
```

## The `true_intervals` Function

`true_intervals(times, values)` run-length encodes a `SimpleArrayBool`
series into the intervals where it was true. The result is a
`SimpleArrayUint64` of shape `(nrun, 3)` whose columns are the start, the
end, and the duration in nanoseconds; a series that is never true gives
shape `(0, 3)`. A run starts at the timestamp of the sample that turns the
series true and ends at the timestamp of the sample that turns it false
again, so each row spans the half-open interval `[start, end)`:

```python
times = solvcon.SimpleArrayUint64(
    array=np.array([0, 10, 20, 30, 40], dtype='uint64'))
over = solvcon.SimpleArrayBool(
    array=np.array([False, True, True, False, True]))
runs = ts.true_intervals(times, over)
assert runs.ndarray.tolist() == [[10, 30, 20], [40, 40, 0]]
assert int(runs.ndarray[:, 2].sum()) == 20  # total time over the limit
```

## Benchmarks

`profiling/profile_timeseries.py` times every kernel against the pure-Python
loop it replaces on an hour-long log sampled at 100 Hz. `make pyprof` runs
it with the other profiling scripts and writes the table to
`profiling/results/`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
