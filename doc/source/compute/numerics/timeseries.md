# Time Series

`solvcon.timeseries` holds kernels over recorded time series. A series is a
pair of one-dimensional arrays of the same length. The first holds the
timestamps in integer nanoseconds as a non-decreasing `SimpleArrayUint64`. The
second holds the values sampled at them as a SimpleArray. Each kernel states
the value classes it takes, reads its input, and returns new arrays. The C++
kernels live in `cpp/solvcon/timeseries/` in the namespace
`solvcon::timeseries`.

The kernels replace the per-sample Python loops that a recorded log otherwise
needs. Sample lookup on the timestamps is not a kernel here; it is
`searchsorted`, which {doc}`the reduce page <../buffer/reduce>` describes.

A kernel that takes a series raises `TypeError` when `times` is not a
`SimpleArrayUint64` or `values` is a class the kernel does not take. It raises
`ValueError` for an array that is not one-dimensional, for an array that
carries ghost elements, for arrays of different length, or for a decreasing
timestamp.

## Repeated Timestamps

A recorded log can stamp two messages in the same nanosecond, so a repeated
timestamp is valid input. A group of equal timestamps resolves to its last
sample. `deriv` is the exception: a zero time step has no derivative, so
`deriv` raises `ValueError`. `dedup_last` collapses a repeat, so its result is
valid input to `deriv`.

## The `merge_sorted_unique` Function

`merge_sorted_unique(*arrays)` merges any number of sorted `SimpleArrayUint64`
timestamp arrays into one sorted `SimpleArrayUint64` that holds every
distinct timestamp once. This is the union time grid that compares two
signals sampled at different rates:

```python
from solvcon import timeseries as ts

t1 = solvcon.SimpleArrayUint64(array=np.array([0, 10, 20], dtype='uint64'))
t2 = solvcon.SimpleArrayUint64(array=np.array([5, 10, 10], dtype='uint64'))
assert ts.merge_sorted_unique(t1, t2).ndarray.tolist() == [0, 5, 10, 20]
assert ts.merge_sorted_unique().shape == (0,)
```

A timestamp repeated within one array or shared between arrays appears once.
No argument, or only empty arrays, gives an empty result. Every argument must
be a `SimpleArrayUint64`; another class raises `TypeError`. A `ValueError`
names the offending array by its position.

## The `dedup_last` Function

`dedup_last(times, values)` keeps the last sample of every group of equal
timestamps. It returns a new array of the kept timestamps and a new array of
the kept values. The result timestamps are strictly increasing, and the result
values keep the class of the input values. The kernel takes every typed
SimpleArray class. A series without a repeat comes back as a copy:

```python
times = solvcon.SimpleArrayUint64(array=np.array([0, 1, 1, 2], dtype='uint64'))
speed = solvcon.SimpleArrayFloat64(array=np.array([0.5, 1.0, 1.5, 2.0]))
t_u, speed_u = ts.dedup_last(times, speed)
assert t_u.ndarray.tolist() == [0, 1, 2]
assert speed_u.ndarray.tolist() == [0.5, 1.5, 2.0]
```

## The `deriv` Function

`deriv(times, values)` differentiates a series by the backward difference
`(x_i - x_{i-1}) / (t_i - t_{i-1})` and returns the tuple
`(times[1:], derivatives)`. The first sample has no predecessor, so `n`
samples give `n - 1` derivatives and fewer than two samples give empty
arrays. The output timestamps are strictly increasing, so a second `deriv`
chains directly. The kernel takes the real-number classes only, so
`SimpleArrayBool` and the complex classes raise `TypeError`. A
`SimpleArrayFloat32` input gives `SimpleArrayFloat32` derivatives; every
other class gives `SimpleArrayFloat64`:

```python
times = solvcon.SimpleArrayUint64(array=np.array([0, 10, 30], dtype='uint64'))
speed = solvcon.SimpleArrayFloat64(array=np.array([1.0, 3.0, 2.0]))
t_acc, acc = ts.deriv(times, speed)
assert t_acc.ndarray.tolist() == [10, 30]
# (3.0-1.0)/(10.0-0.0) and (2.0-3.0)/(30.0-10.0)
assert acc.ndarray.tolist() == [0.2, -0.05]
t_jerk, jerk = ts.deriv(t_acc, acc)
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
