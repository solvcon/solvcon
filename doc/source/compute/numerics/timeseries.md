# Time Series

`solvcon.timeseries` holds kernels over recorded time series. A series is a
pair of one-dimensional arrays of the same length. The first array holds the
timestamps in integer nanoseconds as a `SimpleArrayUint64` in non-decreasing
order. The second holds the values sampled at them as a SimpleArray of any
class. Each kernel reads its input, returns new arrays, and never changes the
input. The C++ kernels live in `cpp/solvcon/timeseries/` in the namespace
`solvcon::timeseries`.

The kernels replace the per-sample Python loops that a recorded log otherwise
needs. Sample lookup on the timestamps is not a kernel here; it is
`searchsorted`, which {doc}`the reduce page <../buffer/reduce>` describes.

## Duplicate Timestamps

A recorded log may stamp two messages in the same nanosecond, so a repeated
timestamp is valid input. A group of equal timestamps resolves to its last
sample.

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
be a `SimpleArrayUint64`; another class raises `TypeError`. An array that is
not one-dimensional, or that holds a decreasing timestamp, raises
`ValueError` and names the array by its position.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
