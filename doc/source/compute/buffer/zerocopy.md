# Zero-Copy between C++ and Python

The SimpleArray family shares memory with numpy in both directions without
copying. A typed array can wrap the memory of an existing `ndarray`, and every
array exposes its memory back to numpy through the `ndarray` property and the
Python buffer protocol. This page defines the sharing semantics: which source
layouts are accepted, which are rejected, how lifetimes are tied together, and
how the complex element types map to numpy dtypes. The constructor forms
themselves are defined in {doc}`Construction and Data Types <construct>`; this
page covers the behavior of the shared memory.

## Wrapping an Ndarray Zero-Copy

The typed `array=` constructor form wraps the memory of a numpy array instead
of allocating. The wrap direction diverges from numpy: `numpy.asarray`
converts the dtype and copies when it must, while the typed wrap never copies
and never converts. It either shares the memory of the source exactly as it is
laid out, or raises (see the rejection rules below). The array and the wrapper
address the same elements, so a mutation made through either side is visible
through the other:

```python
import numpy as np
ndarr = np.arange(24, dtype='float64').reshape((2, 3, 4))
sarr = solvcon.SimpleArrayFloat64(array=ndarr)

sarr.ndarray.fill(1)
assert (ndarr == 1).all()
sarr[0, 0, 0] = 10
assert ndarr[0, 0, 0] == 10
```

### Lifetime

The wrapper keeps the source memory alive. When the source is an array that
owns its memory, the wrapper holds a reference to it; when the source is a
view, the wrapper walks the `base` chain of the view and holds the outermost
ndarray of the chain. When the chain ends at an ndarray that owns its memory,
the held array is the memory owner; when the outermost ndarray is itself built
over another exporter, the true owner stays alive transitively through the
held ndarray. Either way the wrapped memory outlives the Python names of both
the view and the owner:

```python
src = np.arange(24, dtype='float64').reshape((2, 3, 4))
sarr = solvcon.SimpleArrayFloat64(array=src[1:, ::2])
del src
assert sarr[0, 0, 0] == 12.0  # the owning array is kept alive
```

The `is_from_python` property reports the provenance: `True` for an array
wrapping numpy memory and `False` for an array that allocated its own buffer.
`clone()` always allocates, so the clone of a wrapping array reports `False`
and detaches from the numpy source.

## Supported Source Layouts

The wrap accepts the strided layouts numpy produces over a non-empty block of
elements, not only C-contiguous ones:

- C-ordered (row-major) arrays.
- Fortran-ordered (column-major) arrays.
- Sliced views, including slices with a step and views of a subrange.
- Views with negative strides, such as reversed slices.
- Transposed views.

"Supported" means the wrapper records the shape and stride of the source view
and addresses exactly the viewed elements: element reads agree with the source
view index by index, reductions such as `sum()` cover only the viewed
elements, and a write through the wrapper lands in the viewed region of the
original array and nowhere else:

```python
ndarr = np.arange(1000, dtype='float64').reshape((10, 10, 10))
view = ndarr[1:7:3, 6:2:-1, 3:9]
sarr = solvcon.SimpleArrayFloat64(array=view)
assert sarr.shape == (2, 4, 6)
assert sarr[0, 0, 0] == view[0, 0, 0]
assert sarr.stride == (300, -10, 1)  # elements
assert view.strides == (2400, -80, 8)  # numpy reports bytes
```

The wrapper records the stride of the view in elements, per the stride
convention defined in {doc}`Construction and Data Types <construct>`, where
numpy `strides` reports bytes; the last two lines above show the same view in
both units.

The `is_c_contiguous` and `is_f_contiguous` properties report the layout of
the wrapped view, and a degenerate shape reports both as `True`, as defined in
{doc}`Construction and Data Types <construct>`. An empty source wraps whatever
its extents are, in either memory order, and also reports both flags as
`True`, matching how numpy flags an empty array:

```python
sarr = solvcon.SimpleArrayFloat64(array=np.empty((0, 3), dtype='float64'))
assert sarr.shape == (0, 3)
assert sarr.stride == (0, 0)
assert sarr.is_c_contiguous and sarr.is_f_contiguous
```

The wrapped stride is zero in every dimension because an empty source
addresses no element; `(3, 0)`, `(0, 0)`, and `(3, 0, 2)` wrap the same way.

## Rejection Rules

The typed wrap validates the source before sharing its memory. The source must
be writable; a read-only array, including the views produced by
`numpy.broadcast_to`, raises `ValueError`:

```python
ndarr = np.arange(6, dtype='float64')
ndarr.setflags(write=False)
solvcon.SimpleArrayFloat64(array=ndarr)
# ValueError: array is not writeable
```

Beyond writability, the wrap raises `RuntimeError` in three cases. First, the
dtype of the source must equal the element type of the class; no conversion is
attempted:

```python
ndarr = np.arange(24, dtype='float64').reshape((2, 3, 4))
solvcon.SimpleArrayInt8(array=ndarr)
# RuntimeError: dtype mismatch
```

Second, the byte stride of every dimension must be divisible by the item size,
so that the stride can be represented in whole elements:

```python
ndarr = np.ndarray((3,), dtype='int32', buffer=bytearray(range(32)),
                   strides=(1,))
solvcon.SimpleArrayInt32(array=ndarr)
# RuntimeError: NumPy byte stride 1 in dimension 0 is not divisible by
# item size 4
```

Third, the data pointer of the source must be aligned for the element type:

```python
ndarr = np.ndarray((3,), dtype='int32', buffer=bytearray(range(32)),
                   offset=1)
solvcon.SimpleArrayInt32(array=ndarr)
# RuntimeError: NumPy data pointer is not aligned for item alignment 4
```

Writability aside, these three are the only rejections. In particular an empty
source is not one: the constructor's stride check treats a zero-extent axis as
contiguous, so an empty shape wraps whatever its other extents are, as the
supported-layout section above shows.

These rules apply to the typed `array=` form. The dtype-erased `SimpleArray`
constructor infers the dtype from the source instead of checking it, as
{doc}`Construction and Data Types <construct>` states.

## The Ndarray Property

Every typed array exposes the `ndarray` property: a zero-copy numpy view of
the array's memory. The view matches numpy exactly, because it is a numpy
array: it carries the dtype of the element type, the shape of the array, and
the stride converted to bytes. It is writable, and a mutation made through
either the view or the array is visible through the other:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
ndarr = sarr.ndarray
assert ndarr.dtype == np.float64
assert ndarr.shape == (2, 3, 4)
sarr[0, 0, 0] = 5.0
assert ndarr[0, 0, 0] == 5.0
ndarr[1, 2, 3] = 7.0
assert sarr[1, 2, 3] == 7.0
```

The view holds a reference to the array's `ConcreteBuffer` as its `base`, so
the memory stays alive as long as the view does, even after the array object
itself is released. The property reflects the layout of the array it is taken
from: the view of a Fortran-ordered or negative-stride wrapper carries the
same strides as the source, and the view of a transposed or reshaped array
describes that array's layout.

On an array carrying a ghost region, the property diverges from the array's
own indexing: the view spans the full first dimension, including the ghost
region, and indexes from the start of storage. Element `-nghost` of the array
is element zero of the view:

```python
sarr = solvcon.SimpleArrayInt8(8)
sarr.ndarray[:] = np.arange(8, dtype='int8')
sarr.nghost = 3
assert sarr[-3] == 0   # first ghost element
assert sarr[0] == 3    # first body element
assert sarr.ndarray[0] == 0  # the view starts at the ghost region
```

## The Buffer Protocol

The typed classes implement the Python buffer protocol, matching numpy: any
consumer of the protocol sees the array's dtype, shape, and byte strides.
`numpy.array` with `copy=False` builds a shared view equivalent to the
`ndarray` property, and `memoryview` exposes the same description:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3))
ndarr = np.array(sarr, copy=False)
assert ndarr.dtype == np.float64
ndarr[1, 2] = 7.0
assert sarr[1, 2] == 7.0

view = memoryview(sarr)
assert view.format == 'd'
assert view.shape == (2, 3)
assert view.strides == (24, 8)  # bytes, per the protocol
```

The protocol ignores a ghost region the same way the `ndarray` property does:
the description spans the full first dimension and starts at the first ghost
element, so numpy code sees one plain array and the partition exists only in
the array's own subscript arithmetic.

Without `copy=False`, `numpy.array` copies by default, and the copy does not
share memory with the array. The protocol also describes non-contiguous arrays
faithfully; comparing the `memoryview` of two arrays compares the elements
they address, so a transposed view compares equal to an array transposed in
place over the same memory.

## Complex Dtypes

The complex element types match numpy at the interoperation boundary.
`SimpleArrayComplex64` maps to `numpy.complex64` and `SimpleArrayComplex128`
to `numpy.complex128`: the `ndarray` property and the buffer protocol both
present the standard numpy complex dtypes, so a shared view reads and writes
numpy complex scalars directly:

```python
sarr = solvcon.SimpleArrayComplex64(4)
sarr.fill(solvcon.complex64(real=1.5, imag=2.5))
assert sarr.ndarray.dtype == np.complex64
ndarr = np.array(sarr, copy=False)
ndarr[1] = 3 + 4j
assert complex(sarr[1]) == 3 + 4j
```

Element reads through the array return solvcon's own scalar types,
`solvcon.complex64` and `solvcon.complex128`, which carry `real` and `imag`
attributes and convert to the Python built-in through `complex()`. Element
writes accept both spellings: a solvcon complex scalar or a Python or numpy
complex value:

```python
sarr = solvcon.SimpleArrayComplex128(2)
sarr[0] = solvcon.complex128(1.0, 2.0)
sarr[1] = 3 + 4j
assert type(sarr[0]) is solvcon.complex128
assert complex(sarr[1]) == 3 + 4j
```

Each scalar type reports its numpy dtype through the `dtype()` method on the
class: `solvcon.complex64.dtype()` equals `numpy.dtype('complex64')` and
likewise for `complex128`. Wrapping a complex ndarray with the `array=` form
follows the same rules as the other dtypes, including the exact-match dtype
check: a `complex64` source wraps only into `SimpleArrayComplex64`, and
passing it to `SimpleArrayComplex128` raises `RuntimeError`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
