# Reductions, Statistics, Sorting, and Searching

SimpleArray provides the reductions `min`, `max`, and `sum`, the statistics
`mean`, `average`, `median`, `var`, and `std`, the sorting group `sort`,
`argsort`, `searchsorted`, and `take_along_axis`, and the searching group
`argmin`, `argmax`, and `argwhere`.

## Whole-Array Reductions

`min()`, `max()`, and `sum()` take no argument and reduce the whole array to
one scalar of the element type:

```python
sarr = solvcon.SimpleArrayFloat64(shape=(2, 4), value=1.0)
assert sarr.sum() == 8.0
sarr[1, 0] = 9.2
sarr[0, 3] = -2.3
assert sarr.min() == -2.3
assert sarr.max() == 9.2
```

The scalar result matches the numpy reductions without an axis. The numpy
`axis` keyword is not accepted: `sarr.sum(axis=0)` raises `TypeError` from the
binding's argument matching. The statistics and the searching group below do
take an axis, so the gap is specific to these three; whether they should grow
the same axis form is an open decision.

`sum()` follows the logical indices, so it is verified on strided,
non-contiguous arrays and on both C- and F-contiguous layouts, and it returns
zero on an empty array. `min()` and `max()` address their elements through the
linear storage; the verified scope is contiguous arrays, and the tests
exercise the integer and floating-point classes.

On `SimpleArrayBool` the sum accumulates with logical or, so `sum()` answers
whether any element is true. The boolean branch is explicit in the kernel, so
the behavior is deliberate; it diverges from numpy, where summing a boolean
array counts the true elements:

```python
sarr = solvcon.SimpleArrayBool(shape=(3, 2), value=1)
assert sarr.sum() is True     # numpy would count: 6
```

## Statistics

`mean`, `average`, `median`, `var`, and `std` each come in two forms: without
an axis they reduce the whole array to a scalar, and with an axis they return
an array of the same class with the reduced axes removed. The axis accepts a
single integer or a list of integers:

```python
narr = np.arange(24, dtype='float64').reshape((2, 3, 4))
sarr = solvcon.SimpleArrayFloat64(array=narr)
assert sarr.mean() == np.mean(narr)
sres = sarr.mean(axis=[0, 2])
assert (sres.ndarray == np.mean(narr, axis=(0, 2))).all()
```

Three error cases guard the axis form: an axis outside `[0, ndim)` raises
`IndexError` (`reduce: axis out of range`), so the negative axis spelling of
numpy is rejected instead of counting from the end, and reducing no axis or
every axis raises `RuntimeError`
(`reduce: no axis to reduce or all axes are reduced`), where numpy would
return a scalar.

### The `mean` and `average` Methods

`mean()` is the arithmetic mean, `sum()` over the element count. An empty
array raises `RuntimeError` (`SimpleArray::mean(): empty array`), where numpy
warns and returns NaN.

`average(weight=None)` without a weight is `mean()`. The keyword is named
`weight`, not the numpy `weights`, and it takes an array of the same class. In
the whole-array form the weight must have the receiver's shape and weights
elementwise; in the axis form `average(axis, weight=None)` the weight supplies
one value per element of each reduced slice:

```python
narr = np.arange(6, dtype='float64').reshape((2, 3))
weights = np.array([0.5, 0.3, 0.2], dtype='float64')
sarr = solvcon.SimpleArrayFloat64(array=narr)
swei = solvcon.SimpleArrayFloat64(array=weights)
sres = sarr.average(axis=1, weight=swei)
assert np.allclose(sres.ndarray, np.average(narr, weights=weights,
                                            axis=1))
```

A weight of the wrong shape and a weight summing to zero each raise
`RuntimeError`. The whole-array form reports
`SimpleArray::average(): weight shape does not match array shape` and
`SimpleArray::average(): total weight is zero`; the axis form checks per
reduced slice and reports
`SimpleArray::average_op(): weight size does not match array size` and
`SimpleArray::average_op(): total weight is zero`. Numpy raises
`ZeroDivisionError` for the zero total.

### The `median` Method

`median()` returns the middle element, averaging the two middle elements for
an even count, equal to `numpy.median` on the floating-point classes. The
complex classes order lexicographically by the real part and then the
imaginary part, reproducing the numpy ordering, and the result is verified
equal to `numpy.median` on `complex128` data:

```python
narr = np.array([1 + 10j, 2 + 1j, 3 + 0j, 0 + 3j], dtype='complex128')
sarr = solvcon.SimpleArrayComplex128(array=narr)
med = sarr.median()
assert complex(med.real, med.imag) == np.median(narr)  # 1.5+5.5j
```

The 8-bit and boolean classes compute the median by frequency counting instead
of sorting; the result is verified against numpy within the element type.

### The `var` and `std` Methods

`var(ddof=0)` and `std(ddof=0)` take the delta degrees of freedom as numpy
does, dividing by `n - ddof`; a `ddof` not smaller than the element count
raises `RuntimeError`. On the floating-point classes both match numpy:

```python
narr = np.arange(24, dtype='float64').reshape((2, 3, 4))
sarr = solvcon.SimpleArrayFloat64(array=narr)
assert sarr.var() == np.var(narr)
assert sarr.std(ddof=1) == np.std(narr, ddof=1)
assert (sarr.var(axis=1).ndarray == np.var(narr, axis=1)).all()
```

On the complex classes the variance accumulates the squared magnitude and the
result is real, matching numpy; in the axis form the result array is the
matching real-typed class.

### Integer Statistics Keep the Element Type

On the integer classes the statistics compute in the element type, so every
division truncates, where numpy promotes to `float64`. The kernels return
`value_type` for `mean`, `average`, and `median`, and the real-typed `var` and
`std` reduce to the element type for the integer classes:

```python
sarr = solvcon.SimpleArrayInt32(array=np.array([1, 2, 3, 4],
                                               dtype='int32'))
assert sarr.mean() == 2       # numpy: 2.5
assert sarr.var() == 3        # numpy: 1.25, with the truncated mean
```

The tests verify the statistics only on the floating-point and complex
classes, plus the boolean and 8-bit median; the integer truncation is
established from the kernel source and the bound signatures. Whether the
integer statistics should promote to a floating-point result as numpy does is
an open decision; this page records the truncating behavior as fact.

## Sorting and Gathering

### The `sort` Method

`sort()` sorts the receiver in place, ascending, and returns `None`, the
in-place counterpart of the numpy `ndarray.sort`. Only one-dimensional arrays
are supported; any other rank raises `RuntimeError`:

```python
sarr = solvcon.SimpleArrayFloat64(array=np.array([3.0, 1.0, 2.0]))
sarr.sort()
assert sarr.ndarray.tolist() == [1.0, 2.0, 3.0]
```

The order is the numpy one rather than the one the built-in `<` gives. A NaN
sorts after every number and counts equal to another NaN, and a complex value
carrying a NaN in either component goes past all the others as one group,
ordered lexicographically within it. `sort()`, `argsort()`, and
`searchsorted()` share that order, so an array `sort()` has ordered is one
`searchsorted()` can search. It covers those three only: the reductions and
`argmin`/`argmax` order through the built-in comparison and so skip a NaN
where numpy propagates it.

```python
narr = np.array([3.0, float('nan'), 1.0, 2.0])
sarr = solvcon.SimpleArrayFloat64(array=narr)
sarr.sort()
assert np.array_equal(sarr.ndarray, np.sort(narr), equal_nan=True)
```

### The `argsort` Method

`argsort()` returns the indices that sort the receiver, under the same
one-dimensional restriction and error form as `sort()`. The ordering matches
`numpy.argsort`, but the return type diverges from numpy: the result is a
`SimpleArrayUint64`, where numpy returns a signed `intp` array:

```python
sarr = solvcon.SimpleArrayFloat64(array=np.array([3.0, 1.0, 2.0]))
args = sarr.argsort()
assert type(args) is solvcon.SimpleArrayUint64
assert args.ndarray.tolist() == [1, 2, 0]
```

### The `searchsorted` Method

`searchsorted(values, side="left")` returns the insertion points that keep a
sorted one-dimensional receiver sorted, following `numpy.searchsorted`. The
operand is either a scalar, giving a Python `int`, or a SimpleArray, giving a
`SimpleArrayUint64` of the operand's shape. `side="left"` gives the first
position where the value may be inserted and `side="right"` the position after
the last equal element, so the two differ only where the receiver holds a run
of equal values:

```python
sarr = solvcon.SimpleArrayFloat64(array=np.array([1.5, 2.5, 2.5, 4.0]))
assert sarr.searchsorted(2.5) == 1
assert sarr.searchsorted(2.5, side='right') == 3

varr = solvcon.SimpleArrayFloat64(array=np.array([2.5, 4.0]))
assert sarr.searchsorted(varr).ndarray.tolist() == [1, 3]
```

The receiver must already be sorted under the order described for `sort()`;
searching an unsorted receiver returns unspecified indices rather than raising.
A sorted operand is answered by stepping forward from the previous result
rather than searching the whole receiver each time, which is the resampling
case and the faster one; an unsorted operand costs a full search per value.
Only one-dimensional arrays are supported, for both the receiver and the
operand, and any other rank raises `RuntimeError`. A `side` that is neither
`"left"` nor `"right"` raises `ValueError`.

Two things diverge from numpy. The result is a `SimpleArrayUint64`, where
numpy returns a signed `intp` array, and the array operand must be a
SimpleArray of the same class as the receiver; a numpy array or another
SimpleArray class raises `TypeError`.

The unsigned result needs care in the zeroth-order-hold lookup
`searchsorted(side="right") - 1`, the index of the last element at or before
each query. Where numpy's signed result gives `-1` for a query before the
first element, the unsigned one wraps to `2**64 - 1`, so subtract only after
testing for 0:

```python
index = solvcon.SimpleArrayUint64(array=np.array([10, 20, 30],
                                                 dtype='uint64'))
grid = solvcon.SimpleArrayUint64(array=np.array([5, 10, 25, 40],
                                                dtype='uint64'))
found = index.searchsorted(grid, side='right').ndarray
assert found.tolist() == [0, 1, 2, 3]
assert (found - 1).tolist() == [2**64 - 1, 0, 1, 2]  # numpy: [-1, 0, 1, 2]

held = found[found > 0] - 1                          # queries with a match
assert held.tolist() == [0, 1, 2]
```

On a ghosted receiver the indices count from the start of the buffer and so
include the ghost part, the same basis `argsort()` and `take_along_axis()` use.
A ghosted operand is searched over its whole buffer for the same reason, and
the result carries no ghost of its own, so it is not ghost-aligned with the
operand.

### The `take_along_axis` Method

`take_along_axis(indices)` gathers elements of a one-dimensional receiver by
flat index. The indices operand is a SimpleArray of any integer class and any
shape, and the result takes the operand's shape, so composing with `argsort`
yields the sorted values without disturbing the receiver:

```python
sarr = solvcon.SimpleArrayFloat64(array=np.array([3.0, 1.0, 2.0]))
sres = sarr.take_along_axis(sarr.argsort())
assert sres.ndarray.tolist() == [1.0, 2.0, 3.0]
assert sarr.ndarray.tolist() == [3.0, 1.0, 2.0]
```

Despite the name, the semantics are those of `numpy.take` on a flat array; the
numpy `take_along_axis`, which gathers along one axis of a same-rank index
array, does not apply to the one-dimensional receiver. The naming diverges
from numpy and is recorded here.

An out-of-range index raises `IndexError` naming the offending position in the
indices operand:

```python
data = solvcon.SimpleArrayInt32(array=np.arange(10, dtype='int32'))
idx = solvcon.SimpleArrayUint64(
    array=np.array([[0, 1], [2, 3], [4, 20]], dtype='uint64'))
data.take_along_axis(idx)
# IndexError: SimpleArray::take_along_axis(): indices[2, 1] is 20,
# which is out of range of the array size 10
```

An operand that is not an integer-classed SimpleArray is not rejected: the
binding falls through without gathering and returns the receiver itself,
silently ignoring the operand. The explicit list of accepted classes in the
binding makes the intent clear, so raising `TypeError` for other operands is
target behavior; do not rely on the fall-through.

`take_along_axis_simd(indices)` is the performance-explicit variant with
identical desired semantics. The current implementation validates all indices
up front and then gathers with the same scalar loop; no vector gather kernel
backs it yet. Its out-of-range message carries the `_simd` name.

## Searching

### The `argmin` and `argmax` Methods

`argmin(axis=None)` and `argmax(axis=None)` come in the two forms of the numpy
methods. Without an axis they return the flat index of the smallest and
largest element as a Python `int`; ties resolve to the first occurrence, as in
numpy:

```python
narr = np.array([[1, 3, 5, 7, 9],
                 [2, 4, 6, 8, 10],
                 [1, 10, 1, 10, 1]], dtype='float64')
sarr = solvcon.SimpleArrayFloat64(array=narr)
assert sarr.argmin() == narr.argmin() == 0
assert sarr.argmax() == narr.argmax() == 9
```

With an axis they reduce that axis away and return the offsets along it as a
`SimpleArrayUint64`, whose shape is the receiver's with the reduced axis
removed. The values equal the numpy methods; the dtype diverges, unsigned
against the numpy signed `intp`, the same divergence as `argsort`:

```python
narr = np.arange(24, dtype='float64').reshape((2, 3, 4))
sarr = solvcon.SimpleArrayFloat64(array=narr)
sres = sarr.argmax(axis=1)
assert type(sres) is solvcon.SimpleArrayUint64
assert sres.shape == (2, 4)
assert (sres.ndarray == np.argmax(narr, axis=1)).all()
assert (sarr.argmin(axis=-1).ndarray == np.argmin(narr, axis=-1)).all()
```

A negative axis counts from the end, as in numpy, and as the last line above
shows. This is the opposite of the statistics above, which reject a negative
axis; the two axis conventions inside one class are inconsistent, and
reconciling them is an open decision.

On a one-dimensional receiver the binding maps `axis=0` and `axis=-1` onto the
whole-array form, so the result is a Python `int` rather than a one-element
array. Numpy returns its scalar for the same call, so the shapes agree:

```python
sarr = solvcon.SimpleArrayFloat64(array=np.array([3.0, 1.0, 2.0]))
assert sarr.argmin(axis=0) == 1
assert sarr.argmin(axis=-1) == 1
```

Two axis errors raise `ValueError`. The type matches numpy, which raises its
`AxisError` (a `ValueError` subclass) for an axis out of bounds and a plain
`ValueError` for an empty reduced axis; only the message text differs. An
out-of-bounds axis is reported after the negative wrap, so `axis=-4` on a
three-dimensional array names axis -1:

```python
solvcon.SimpleArrayFloat64((2, 3, 4)).argmin(axis=3)
# ValueError: SimpleArray::argmin(): axis 3 is out of bounds for array
# of dimension 3
solvcon.SimpleArrayFloat64((0, 3), value=0.0).argmin(axis=0)
# ValueError: SimpleArray::argmin(): axis 0 has size 0, cannot compute
```

A zero-extent axis that is retained rather than reduced is not an error: the
result is simply empty, as in numpy.

A NaN in the reduced run wins immediately, whichever method is called, and the
first NaN wins when several are present. Both rules match numpy:

```python
narr = np.array([[1.0, np.nan, 0.0]], dtype='float64')
sarr = solvcon.SimpleArrayFloat64(array=narr)
assert sarr.argmin(axis=1).ndarray.tolist() == [1]
assert np.argmin(narr, axis=1).tolist() == [1]
```

Both forms follow the logical indices, so a strided or Fortran-ordered
receiver reports the offsets of the viewed elements; the tests cover a
reversed two-dimensional view alongside the contiguous ranks up to four
dimensions.

### The `argwhere` Method

`argwhere()` maps the nonzero elements to their coordinates as a
`SimpleArrayUint64` of shape `(count, ndim)`, one row per selected element in
row-major order. The values equal `numpy.argwhere`; the dtype diverges,
unsigned against the numpy signed `intp`, the same divergence as `argsort`.
The method is bound on every typed class, and the boolean array of a
comparison is its intended condition form;
{doc}`Elementwise Arithmetic, Comparison, and Selection <elementwise>` fixes
that spelling and the rest of the selection family:

```python
narr = np.array([[1, 3, 5], [10, 4, 10]], dtype='float64')
sarr = solvcon.SimpleArrayFloat64(array=narr)
ret = sarr.eq(10).argwhere()
assert (ret.ndarray == np.argwhere(narr == 10)).all()
```

Unlike `argmin` and `argmax`, `argwhere` addresses its elements through the
linear storage, as `min()` and `max()` do; the verified scope is C-contiguous
arrays.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
