# Indexing, Shape, and Layout

Every array in the SimpleArray family reads and writes single elements through
the subscript operator, describes its layout through a set of properties, and
manipulates its shape and memory order through `reshape` and the transpose
family. This page defines those operations: which keys the subscript accepts,
which right-hand sides assignment takes, what the layout properties report,
and how the shape and layout of an array are changed. Arrays carrying a ghost
region shift the index origin; the sections up to the layout conversions
assume arrays without one, and
{ref}`the last section of this page <ghost-region>` defines the partition and
every rule it changes.

## Element Access

Element access matches numpy in the index arithmetic and the error behavior
(negative wrapping and `IndexError`), and diverges from numpy in the subscript
scope and the return type: a subscript must select exactly one element, and
the result is a Python scalar, never a subarray or a view. A one-dimensional
array takes a single integer, and a multi-dimensional array takes a full tuple
with one integer per dimension:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
sarr.ndarray.flat[:] = range(24)
assert sarr[1, 2, 3] == 23.0

sarr1d = solvcon.SimpleArrayInt32(7, value=3)
assert sarr1d[0] == 3
```

The returned scalar is the Python built-in matching the element type: `float`
for the floating-point classes, `int` for the integer classes, and `bool` for
`SimpleArrayBool`. The complex classes return solvcon's own scalar types, as
defined in {doc}`Zero-Copy between C++ and Python <zerocopy>`. Numpy instead
returns its own scalar types (`numpy.float64` and friends); returning the
plain Python scalar is the desired behavior.

Partial indexing does not produce subarrays. Where numpy resolves `ndarr[0]`
on a three-dimensional array to a two-dimensional view, the SimpleArray
classes require the index to address one element and raise `IndexError`
otherwise:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
sarr[0]
# IndexError: SimpleArray::normalize_index(): cannot use scalar index
# for 3-dimensional array
sarr[0, 1]
# IndexError: SimpleArray: dimension of input indices [0, 1] != array
# dimension 3
```

### Negative Indices

Negative indices wrap from the end of each dimension, matching the Python
sequence convention that numpy also follows: a negative index `i` resolves to
`n + i` for a dimension of length `n`, so the valid interval per dimension is
`[-n, n)`:

```python
sarr = solvcon.SimpleArrayFloat64((4, 3, 2))
sarr.ndarray.flat[:] = range(24)
assert sarr[-1, -1, -1] == 23.0
assert sarr[-4, -3, -2] == 0.0
sarr[-1, -1, -1] = 230.0
assert sarr[3, 2, 1] == 230.0
```

The first axis is the exception, and only when the array carries a ghost
region. The negative indices there address the ghost elements rather than
wrapping: index `-1` is the last ghost element and index `-nghost` the first,
so the wrap begins one position further down, at `-nghost - 1`. The error
messages of this page carry the arithmetic in their `nghost` term, which is 0
on the ghost-free arrays assumed here.
{ref}`The last section of this page <ghost-region>` defines the shifted
interval in full.

### Out-of-Range Errors

An index outside the valid interval raises `IndexError`, matching numpy's
exception type. The message names the offending index and the violated bound:

```python
sarr = solvcon.SimpleArrayFloat64(3)
sarr[3]
# IndexError: SimpleArray: index 3 >= 3 (shape[0]: 3 - nghost: 0)
sarr[-4]
# IndexError: SimpleArray: index -4 < -nghost - shape[0]: -3

sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
sarr[0, 3, 0]
# IndexError: SimpleArray: dim 1 in [0, 3, 0] >= shape[1]: 3
```

### No Slices on Read

`__getitem__` accepts no slice and no ellipsis; only the integer and
integer-tuple forms above exist. Passing a slice raises `TypeError` from the
binding's argument matching:

```python
sarr = solvcon.SimpleArrayFloat64(6)
sarr[0:3]
# TypeError: __getitem__(): incompatible function arguments. ...
```

This diverges from numpy, where a slice returns a view sharing the memory of
the source. Whether the family should grow slice reads, and whether such a
read would return a sharing view or a copy, is an open decision; this page
records only the current behavior. Until the decision lands, the zero-copy
path to sliced reads is the `ndarray` property: `sarr.ndarray[0:3]` is a numpy
view over the array's memory.

## Element and Region Assignment

`__setitem__` accepts two families of keys: the scalar keys of the read path,
assigning one element, and slice or ellipsis keys, assigning a whole region
from a sequence. A key and value combination outside the two families raises
`RuntimeError`; in particular a scalar value cannot be assigned to a slice key
(numpy would broadcast it over the region). The message depends on the
rejection path: a scalar on a lone slice or an ellipsis reports "unsupported
operation.", while a scalar on a tuple of slices fails earlier, in the key
cast, with a pybind11 "Unable to cast" message.

### Scalar Assignment

With an integer key on a one-dimensional array or a full integer tuple on a
multi-dimensional array, the value is cast to the element type and stored:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
sarr[0, 0, 0] = 10      # a Python int converts to float64
assert sarr[0, 0, 0] == 10.0
```

The cast follows pybind11 conversion, not numpy value coercion: a Python `int`
converts to a floating-point element, but a value the element type cannot
represent exactly is rejected with `RuntimeError` rather than truncated.
Assigning `2.5` to an integer array or `300` to an `int8` array both raise,
where numpy truncates the float (storing 2) and raises `OverflowError` for the
out-of-range integer. The complex classes accept both solvcon's own complex
scalars and Python or numpy complex values, as defined in
{doc}`Zero-Copy between C++ and Python <zerocopy>`.

### Slice and Ellipsis Assignment

With a slice or ellipsis key, the right-hand side is a sequence whose elements
fill the selected region. Four key shapes are accepted:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
ndarr = np.arange(24, dtype='float64').reshape((2, 3, 4))

sarr[...] = ndarr                    # ellipsis: the whole array
sarr[0:1] = np.ones((1, 3, 4))       # lone slice: first dimension
sarr[::2, ::3, ::4] = np.zeros((1, 1, 1))  # tuple of slices
sarr[::2, ...] = np.ones((1, 3, 4))  # tuple mixing slices and ellipsis
```

A lone slice applies to the first dimension and the remaining dimensions take
their full extent. In a tuple, slices fill dimensions from the left, an
ellipsis expands to full-extent slices for the unnamed middle dimensions, and
slices after the ellipsis fill from the right. Steps and negative bounds
follow the Python slice rules. The syntax is validated: more slices than
dimensions raise `RuntimeError` ("syntax error. dimensions mismatches"), more
than one ellipsis raises `RuntimeError` ("syntax error. no more than one
ellipsis."), and a zero step raises `ValueError` ("slice step cannot be
zero").

### Accepted Right-Hand Sides

The sequence on the right-hand side may be a numpy `ndarray`, a `list`, or a
`tuple` (including nested lists and tuples, which convert through
`numpy.array`):

```python
sarr = solvcon.SimpleArrayFloat64((2, 3))
sarr[:, :] = [[1, 2, 3], [4, 5, 6]]
sarr[:1, :2] = ((7, 8),)
```

A SimpleArray is not accepted: assigning one array into a slice of another
raises `RuntimeError` ("unsupported operation."). This diverges from numpy,
where an array of the library's own kind is the most natural right-hand side.
Whether the accepted set should grow SimpleArray sources is an open decision;
the working spelling today routes through numpy, for example
`sarr[...] = other.ndarray`.

### Shape Checking

The shape of the right-hand side must equal the shape selected by the key
exactly, dimension count included. There is no numpy-style broadcasting of
scalars or lower-dimensional sources, which diverges from numpy assignment. A
mismatch raises `RuntimeError` naming both shapes:

```python
sarr = solvcon.SimpleArrayFloat64((4, 6, 8))
sarr[::2, ::3, ::4] = np.zeros((2, 3, 4))
# RuntimeError: Broadcast input array from shape(2, 3, 4) into
# shape(2, 2, 2)
```

### Dtype Casting

The element type of a sequence right-hand side does not need to match the
array: any dtype of the element-type table in
{doc}`Construction and Data Types <construct>` is converted element-wise
during the copy, so an `int32` or `float32` source fills a `float64` array.
Three conversions are refused. Mixing complex and non-complex types raises
`RuntimeError` ("Cannot convert between complex and non-complex types"). A
complex source fills only the complex array of the same precision, so a
`complex64` source into a `SimpleArrayComplex128` also raises `RuntimeError`,
reusing the same message even though both sides are complex. A dtype outside
the table, such as a string dtype, raises `RuntimeError` ("input array data
type not support!").

## Shape and Layout Properties

Five read-only properties describe the layout. `shape`, `size`, `itemsize`,
and `nbytes` match numpy: the shape tuple, the total element count, the byte
size of one element, and the total byte count. `stride` diverges from numpy:
it counts elements where numpy `strides` counts bytes, per the convention
defined in {doc}`Construction and Data Types <construct>`;
{doc}`Zero-Copy between C++ and Python <zerocopy>` shows the same view in both
units:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
assert sarr.shape == (2, 3, 4)
assert sarr.stride == (12, 4, 1)  # elements; numpy reports (96, 32, 8)
assert sarr.size == 24
assert sarr.nbytes == 192
plex = solvcon.SimpleArray((2, 3, 4), dtype='float64')
assert plex.itemsize == 8
```

On the typed classes, reading `itemsize` currently raises `TypeError`: the
binding registers the zero-argument C++ getter as an instance property, which
pybind11 rejects at access time. This is a defect, not an intended difference;
the example above reads the property through the erased wrapper, where it
works.

### The `len()` Function

`len()` diverges from numpy: it returns the total element count, equal to
`size`, for any dimensionality. Numpy returns the length of the first
dimension:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
assert len(sarr) == 24
assert len(sarr.ndarray) == 2  # numpy counts the first dimension
```

The desired behavior is the total element count: the arrays serve the solvers
as element containers, and `len()` reports the container size the way `len()`
does on a `ConcreteBuffer` or a collector.
{doc}`Construction and Data Types <construct>` notes the divergence where the
property first appears; this page carries the full statement.

## Reshape

`reshape(shape)` returns a new array of the given shape over the same buffer.
The receiver keeps its shape; the result shares the memory, so a write through
either side is visible through the other:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
sarr2 = sarr.reshape(24)
assert sarr2.shape == (24,)
sarr2[5] = 42.0
assert sarr[0, 1, 1] == 42.0
assert sarr.shape == (2, 3, 4)  # the receiver is unchanged
```

The shape argument takes a single integer or a tuple, like the constructors,
and the result is always row-major. The element count of the new shape must
equal the element count of the receiver; a mismatch raises `RuntimeError`:

```python
sarr.reshape(23)
# RuntimeError: SimpleArray: cannot reshape size 24 into size 23
```

The result reads in the logical element order, matching
`numpy.reshape(order='C')`, and the memory is shared only when that order is
already the storage order. A receiver that is C-contiguous over a buffer
holding exactly its elements yields a sharing view, as the first example
shows; any other layout (transposed, Fortran-ordered, or strided) is copied
element by element into a fresh dense buffer, and the copy is independent of
the receiver:

```python
sarr = solvcon.SimpleArrayFloat64(array=np.arange(6.).reshape((2, 3)))
sarr.transpose()
flat = sarr.reshape(6)
assert [flat[i] for i in range(6)] == [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
ndarr = np.arange(6.).reshape((2, 3))
assert ndarr.T.reshape(6).tolist() == [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
flat[0] = 9.0
assert sarr[0, 0] == 0.0          # the copy does not share memory
```

Numpy infers a dimension given as `-1` and raises `ValueError` on a count
mismatch; the SimpleArray `reshape` never infers and raises `RuntimeError`. It
also rejects an array carrying a ghost region outright, because the split of
the first axis has no image under a new shape;
{ref}`the ghost section <ghost-region>` states that rejection with its
message.

## Transpose

The transpose family diverges from numpy, which has no in-place transpose and
returns sharing views from `.transpose()` and `.T`.

### The `transpose` Method

The full signature is `transpose(axis=None, inplace=True, copy=False)`. With
`axis=None` all axes are reversed; with a tuple, the i-th new axis is sourced
from the `axis[i]`-th old axis, following the numpy `transpose` axis
convention except that the entries must be non-negative, where numpy also
accepts negative axis numbers:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
sarr.transpose()
assert sarr.shape == (4, 3, 2)

sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
sarr.transpose((0, 2, 1))
assert sarr.shape == (2, 4, 3)
```

The `axis` tuple must have one entry per dimension and every entry must be a
valid non-negative axis; violations raise `RuntimeError`
("SimpleArray::transpose: axis size mismatch" and "SimpleArray::transpose:
axis out of range"). A repeated axis is not currently detected, where numpy
raises on a repeated axis; rejecting the repeat is target behavior.

The `inplace` and `copy` flags select what is transposed and how:

- `inplace=True` (default) transposes the receiver itself; `inplace=False`
  leaves the receiver untouched and transposes an independent deep copy.
- `copy=False` (default) flips only the metadata: shape and stride are
  permuted and no element moves. Under the full axis reversal the flip of a
  C-contiguous source is F-contiguous; a partial permutation such as
  `(0, 2, 1)` generally yields a layout that is neither. `copy=True`
  physically rearranges the elements into a fresh C-contiguous buffer.

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
sarr.transpose()                  # metadata flip of sarr itself
assert sarr.is_f_contiguous

sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
sarr.transpose(copy=True)         # physical transpose of sarr itself
assert sarr.is_c_contiguous

sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
sarr2 = sarr.transpose(inplace=False)
assert sarr.shape == (2, 3, 4)    # the receiver is unchanged
assert sarr2.shape == (4, 3, 2)
```

The method also returns an array, and the returned object never shares memory
with the receiver: even with `inplace=True` the return value is an independent
deep copy taken after the mutation. Treat the in-place mutation as the primary
effect and the return value as a detached copy. Whether the return value
should instead be the receiver (to support chaining) is an open decision; do
not rely on the returned object aliasing the receiver.

### The `transpose_copy` Method

`transpose_copy()` returns a fresh C-contiguous array with the axes reversed
and the elements physically rearranged, leaving the receiver untouched; it is
the counterpart of `numpy.ascontiguousarray(ndarr.T)`:

```python
sarr = solvcon.SimpleArrayFloat64(array=np.arange(6.).reshape((2, 3)))
tc = sarr.transpose_copy()
assert tc.shape == (3, 2)
assert tc.is_c_contiguous
assert sarr.shape == (2, 3)
```

A zero- or one-dimensional array has no axes to reverse, so the result is a
plain deep copy. Two applications round-trip: transposing the transpose
reproduces the original shape and content.

### The `T` Property

The `T` property returns a deep-copied transposed array: the buffer is cloned
and the metadata of the clone is reversed, so the result never shares memory
with the receiver and the receiver is unchanged. This diverges from numpy,
where `.T` is a zero-copy view:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3), value=1.0)
t = sarr.T
assert t.shape == (3, 2)
assert t.is_f_contiguous          # metadata flip of a C-ordered clone
sarr[0, 0] = 5.0
assert t[0, 0] == 1.0             # the copy does not see the write
```

## Contiguity

### The `is_c_contiguous` and `is_f_contiguous` Properties

The two read-only properties report whether the stride describes a row-major
(C) or column-major (Fortran) layout over the shape, matching the numpy
`flags.c_contiguous` and `flags.f_contiguous` semantics under the property
spelling of the family. Dimensions of extent one place no constraint, so a
degenerate shape (a single row, column, or element) reports both as `True`, as
numpy does. A zero-extent dimension places no constraint either, so every
empty array reports both flags whatever its other extents are, again as numpy
does:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
assert sarr.is_c_contiguous and not sarr.is_f_contiguous
sarr.transpose()
assert sarr.is_f_contiguous and not sarr.is_c_contiguous
assert solvcon.SimpleArrayFloat64((1, 4)).is_f_contiguous
zarr = solvcon.SimpleArrayFloat64((0, 4))
assert zarr.is_c_contiguous and zarr.is_f_contiguous
```

{doc}`Zero-Copy between C++ and Python <zerocopy>` records the wrap-side
consequence of the same flags.

### The `to_row_major` and `to_column_major` Conversions

`to_row_major()` and `to_column_major()` return a fresh array with the same
shape and values whose stride is C-contiguous or F-contiguous respectively.
The result is always a new buffer: when the receiver already has the requested
layout the buffer is cloned, and otherwise a fresh buffer is allocated and the
elements are copied in the new order. The receiver is never modified:

```python
ndarr = np.arange(6, dtype='float64').reshape((2, 3))
sarr = solvcon.SimpleArrayFloat64(array=ndarr[::-1, ::-1])
rm = sarr.to_row_major()
assert rm.is_c_contiguous
cm = sarr.to_column_major()
assert cm.is_f_contiguous
assert cm.stride == (1, 2)
```

The numpy counterparts `numpy.ascontiguousarray` and `numpy.asfortranarray`
return the input unchanged when it already has the requested layout; always
returning an independent copy diverges from numpy, keeping buffer ownership
explicit per the design stance of {doc}`the family overview <index>`.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
