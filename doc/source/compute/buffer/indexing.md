# Indexing, Shape, and Layout

Every array in the SimpleArray family reads and writes elements and regions
through the subscript operator, describes its layout through a set of
properties, and manipulates its shape and memory order through `reshape` and
the transpose family. This page defines those operations: which keys the
subscript accepts, which right-hand sides assignment takes, what the layout
properties report, and how the shape and layout of an array are changed.
Arrays carrying a ghost region shift the index origin; the sections up to the
layout conversions assume arrays without one, and
{ref}`the last section of this page <ghost-region>` defines the partition and
every rule it changes.

## Element Access

The subscript reads two families of keys. An integer key addresses exactly one
element and returns a scalar, and that is what this section defines; a slice
or an ellipsis selects a region and returns a view, which
{ref}`the next section <slice-read>` defines. Element access matches numpy in
the index arithmetic and the error behavior (negative wrapping and
`IndexError`), and diverges from numpy in the return type, which is a Python
scalar rather than a numpy one. A one-dimensional array takes a single
integer, and a multi-dimensional array takes a full tuple with one integer per
dimension:

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
classes require an integer key to address one element and raise `IndexError`
otherwise; a subarray is named with explicit slices instead:

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

(slice-read)=
## Region Access

A slice, an ellipsis, or a tuple containing either selects a region, and the
read returns an array of the same class holding that region. The key grammar
is the one {ref}`the assignment section <region-assignment>` defines, so a key
that writes a region reads the same region back:

```python
sarr = solvcon.SimpleArrayFloat64(6)
sarr[...] = np.arange(6, dtype='float64')

sub = sarr[1:4]
assert isinstance(sub, solvcon.SimpleArrayFloat64)
assert sub.shape == (3,)
assert sub.ndarray.tolist() == [1.0, 2.0, 3.0]
```

The result shares the memory of the source, matching numpy and the sharing
`reshape` below: a write through either side is visible through the other.
The `ndarray` property gives the same view as a numpy array, so
`sarr[1:4].ndarray` and `sarr.ndarray[1:4]` describe the same memory.

```python
sub[0] = 100.0
assert sarr[1] == 100.0
sarr[3] = 200.0
assert sub[2] == 200.0
```

Steps and negative bounds follow the Python slice rules. A step other than 1
gives the view a strided layout, a negative step reverses it, and a key that
selects nothing yields an array with a zero-length axis:

```python
sarr = solvcon.SimpleArrayFloat64(array=np.arange(6, dtype='float64'))
assert sarr[::2].ndarray.tolist() == [0.0, 2.0, 4.0]
assert sarr[::-1].ndarray.tolist() == [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
assert sarr[3:3].shape == (0,)
```

A tuple key names one dimension per slice from the left, and an ellipsis
stands for the full-extent slices of the dimensions it replaces. The syntax
checks are those of the assignment section: more slices than dimensions raise
`RuntimeError` ("syntax error. dimensions mismatches"), more than one ellipsis
raises `RuntimeError` ("syntax error. no more than one ellipsis."), and a zero
step raises `ValueError` ("slice step cannot be zero"):

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
sarr[...] = np.arange(24, dtype='float64').reshape((2, 3, 4))
assert sarr[0:1, ::2, ...].shape == (1, 2, 4)
assert sarr[..., ::2].shape == (2, 3, 2)
```

Mixing an integer with a slice in one tuple key is not accepted, so a region
key cannot drop a dimension the way `ndarr[0, 1:3]` does in numpy. Such a key
raises `RuntimeError` ("unsupported operation."), and a one-element slice is
the working spelling: `sarr[0:1, 1:3]` keeps the first dimension with extent
1. Chaining is the other route, since a view is itself subscriptable:

```python
sarr = solvcon.SimpleArrayFloat64(array=np.arange(12, dtype='float64'))
assert sarr[2:10][::3].ndarray.tolist() == [2.0, 5.0, 8.0]
```

The dtype-erased `SimpleArray` reads a region the same way and returns another
dtype-erased array over the shared memory, keeping its interface aligned with
the typed classes.

(region-assignment)=
## Element and Region Assignment

`__setitem__` accepts the same two families of keys as the read path: the
integer keys, assigning one element, and slice or ellipsis keys, assigning a
whole region from a sequence. A key and value combination outside the two
families raises `RuntimeError`; in particular a scalar value cannot be
assigned to a slice key (numpy would broadcast it over the region). The
message depends on the rejection path: a scalar on a lone slice or an ellipsis
reports "unsupported operation.", while a scalar on a tuple of slices fails
earlier, in the key cast, with a pybind11 "Unable to cast" message.

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

(ghost-region)=
## Ghost Region Support

"Ghost" elements are the elements indexed with a negative integer, in contrast
to the "body" elements indexed with a non-negative integer. The `nghost`
property splits the first axis of an array into a ghost region and a body. The
ghost elements hold the boundary (halo) data, which lives in the same storage
as the interior of a physical computing domain. The feature is entirely
solvcon-specific and numpy has no counterpart to it. The rules below extend
the ghost-free semantics that the rest of this page defines.

### The Partition Model

Every array starts without a ghost region: `nghost` is 0 and `has_ghost` is
`False`. Assigning a positive `nghost` designates the first `nghost` positions
along the first axis as the ghost region and the remaining `nbody` positions
as the body. No memory moves and the shape does not change; only the index
origin shifts. Index 0 becomes the first body element, and the ghost elements
sit at the negative indices `-nghost` through `-1`:

```python
sarr = solvcon.SimpleArrayFloat64(24)
sarr.ndarray[:] = np.arange(24)
sarr.nghost = 10

assert sarr.has_ghost
assert sarr.nbody == 14
assert sarr.shape == (24,)   # the shape is unchanged
assert sarr[-10] == 0.0      # first ghost element
assert sarr[0] == 10.0       # first body element
assert sarr[13] == 23.0      # last body element
```

Only the first axis carries the partition. On a multi-dimensional array the
later axes keep the plain index arithmetic of the element-access rules above:

```python
sarr = solvcon.SimpleArrayFloat64((4, 3, 2))
sarr.ndarray.flat[:] = range(24)
sarr.nghost = 1

assert sarr.nbody == 3
assert sarr[-1, 0, 0] == 0.0    # the ghost row
assert sarr[0, 0, 0] == 6.0     # the first body row
assert sarr[0, -1, 0] == 10.0   # later axes wrap plainly
```

#### Valid Index Range and Wrapping

The valid interval on the first axis is `[-(shape[0] + nghost), nbody)`.
Indices in `[-nghost, nbody)` address the partition directly, as above. An
index below `-nghost` wraps python-style over the storage: the ghost shift
makes it negative relative to the start of storage, and the wrap adds
`shape[0]`, so index `i` resolves to storage position `i + nghost + shape[0]`.
In particular `sarr[-nghost - 1]` is the last storage element, mirroring how
`sarr[-1]` on a ghost-free array is the last element:

```python
sarr = solvcon.SimpleArrayInt8(8)
sarr.ndarray[:] = np.arange(8, dtype='int8')
sarr.nghost = 3

assert sarr[-3] == 0     # first ghost element
assert sarr[-4] == 7     # wraps to the last storage element
assert sarr[-11] == 0    # wraps to the first storage element
sarr[-4] = 70
assert sarr.ndarray[7] == 70
```

An index outside the interval raises `IndexError`, and the message carries the
ghost arithmetic. The one-dimensional form:

```python
sarr = solvcon.SimpleArrayFloat64(24)
sarr.nghost = 10
sarr[14]
# IndexError: SimpleArray: index 14 >= 14 (shape[0]: 24 - nghost: 10)
sarr[-35]
# IndexError: SimpleArray: index -35 < -nghost - shape[0]: -34
```

The multi-dimensional form names the offending dimension:

```python
sarr = solvcon.SimpleArrayFloat64((4, 3, 2))
sarr.nghost = 1
sarr[3, 0, 0]
# IndexError: SimpleArray: dim 0 in [3, 0, 0] >= nbody: 3
# (shape[0]: 4 - nghost: 1)
sarr[-6, 0, 0]
# IndexError: SimpleArray: dim 0 in [-6, 0, 0] < -nghost - shape[0]: -5
```

Both errors apply to reads and writes alike.

### Setting the `nghost` Property

The `nghost` setter accepts any value from 0 through `shape[0]`. Setting it
back to 0 removes the region, and setting it to the full first-axis extent
makes the whole axis ghost:

```python
sarr = solvcon.SimpleArrayInt8(10)
sarr.nghost = 10            # the whole first axis may be ghost
assert sarr.nbody == 0
sarr.nghost = 0             # zero removes the region
assert not sarr.has_ghost
```

Three violations raise `IndexError`. The value cannot exceed the first-axis
extent, cannot be negative, and cannot be positive on a zero-dimensional array
(which the message calls empty); an array whose first axis has zero extent
falls under the `shape(0)` bound instead:

```python
sarr = solvcon.SimpleArrayInt8(10)
sarr.nghost = 11
# IndexError: SimpleArray: cannot set nghost 11 > shape(0) 10
sarr.nghost = -1
# IndexError: SimpleArray: cannot set negative nghost -1
solvcon.SimpleArrayInt8(()).nghost = 1
# IndexError: SimpleArray: cannot set nghost 1 > 0 to an empty array
```

### The `has_ghost` and `nbody` Properties

`has_ghost` reports whether `nghost` is nonzero. `nbody` counts the body
positions along the first axis, `shape[0] - nghost`; it is not an element
count, so a ghost-free `(4, 3, 2)` array reports `nbody == 4` and the same
array with `nghost = 1` reports 3. A zero-dimensional array reports 0. Neither
`shape`, `size`, nor `len()` changes with the partition; they keep describing
the full storage, as the layout properties above define.

### Ghost-Shifted Slice Bounds

Reads and writes parse a slice key the same way, so the bounds below name the
same region on either side. The slice keys interpret their explicit bounds on
the first axis in the logical, ghost-shifted coordinates of this page: the
parser adds `nghost` to an explicit start or stop bound and then applies the
ordinary Python slice rules over the full first-axis extent. An omitted bound
is not shifted; it means the storage edge, so with a forward step an omitted
start begins at the first ghost element and an omitted stop runs to the end of
storage. The stop bound `0` therefore selects exactly the ghost region, and
the start bound `0` selects the body:

```python
sarr = solvcon.SimpleArrayFloat64(shape=5, value=0)
sarr.nghost = 2

sarr[-2:0] = np.array([10.0, 11.0])        # the ghost region
sarr[0:] = np.array([12.0, 13.0, 14.0])    # the body
assert sarr.ndarray.tolist() == [10, 11, 12, 13, 14]

assert sarr[-2:0].ndarray.tolist() == [10, 11]      # read it back
assert sarr[0:].ndarray.tolist() == [12, 13, 14]

sarr[:0] = np.array([20.0, 21.0])          # also the ghost region
assert sarr[:0].ndarray.tolist() == [20, 21]
assert sarr.ndarray.tolist() == [20, 21, 12, 13, 14]
```

The read is a view of the same memory the write reaches, so the body region
that `nghost` defines is addressable as an array in its own right, without
leaving the class for `ndarray`:

```python
body = sarr[0:]
body[0] = 120.0
assert sarr[0] == 120.0
```

A region read does not carry the partition to its result. The view reports
`nghost == 0`, because its first axis is the selected positions and they have
no ghost and body split of their own; assign `nghost` on the view if the
sub-array needs one:

```python
assert sarr[0:].nghost == 0
assert sarr[...].nghost == 0     # even when the key covers the whole storage
```

Because both bounds default to the storage edges, a bare slice, a stepped
slice, or an ellipsis covers the whole storage including the ghost region, and
a negative step reverses over it:

```python
sarr = solvcon.SimpleArrayFloat64(shape=5, value=0)
sarr.nghost = 2
sarr[::2] = np.array([10.0, 11.0, 12.0])
assert sarr.ndarray.tolist() == [10, 0, 11, 0, 12]

sarr[...] = np.arange(5, dtype='float64')
assert sarr[-2] == 0.0 and sarr[2] == 4.0
assert sarr[...].ndarray.tolist() == [0, 1, 2, 3, 4]
```

In a tuple key only the first-axis slice is shifted; slices on the later axes
keep the ghost-free semantics:

```python
sarr = solvcon.SimpleArrayFloat64(shape=(5, 3), value=0)
sarr.nghost = 2
sarr[-2:0, ...] = np.arange(6, dtype='float64').reshape((2, 3))
assert (sarr.ndarray[0:2] == np.arange(6).reshape((2, 3))).all()
assert (sarr[-2:0, ...].ndarray == np.arange(6).reshape((2, 3))).all()
```

The accepted right-hand sides, the exact-shape check, and the dtype conversion
rules are those of the assignment section above, unchanged by the partition.

#### Failure Preserves the Partition

A rejected assignment does not disturb the ghost setting. When the right-hand
side fails the dtype conversion, the array keeps its `nghost` as before the
statement:

```python
sarr = solvcon.SimpleArrayFloat64(shape=(2, 2), value=0)
sarr.nghost = 1
sarr[...] = np.ones((2, 2), dtype='complex128')
# RuntimeError: Cannot convert between complex and non-complex types
assert sarr.nghost == 1
```

### Ghost Regions on Strided Arrays

The partition composes with the strided layouts of
{doc}`Zero-Copy between C++ and Python <zerocopy>`. On an array wrapping a
strided view, the ghost indices address the viewed elements, and a write
through a ghost or wrapped index lands in the viewed region of the original
memory:

```python
base = np.arange(12, dtype='float64')
sarr = solvcon.SimpleArrayFloat64(array=base[::2])
sarr.nghost = 2

assert sarr[-2] == 0.0     # first viewed element
assert sarr[-3] == 10.0    # wraps to the last viewed element
sarr[-3] = 200.0
assert base[10] == 200.0
```

### Ghost Regions and the Layout Operations

`reshape` refuses a ghosted array outright. The split of the first axis has no
well-defined image under a new shape, so both the typed classes and the
dtype-erased `SimpleArray` raise `RuntimeError` naming the ghost count:

```python
sarr = solvcon.SimpleArrayFloat64(6)
sarr.nghost = 2
sarr.reshape((3, 2))
# RuntimeError: SimpleArray: cannot reshape an array with 2 ghost cells
```

The layout operations of this page split into two groups by how they build
their result. Those that duplicate the storage as it lies, `clone()`, the `T`
property, and `to_row_major()` or `to_column_major()` on a receiver that
already has the requested layout, copy the whole storage and carry `nghost` to
the result. The in-place `transpose()` keeps `nghost` as well, since it only
permutes the metadata.

Those that physically rearrange the elements, `transpose_copy()`,
`transpose(copy=True)`, and a `to_row_major()` or `to_column_major()` that
must reorder, reset `nghost` to 0 in the result, and their read is defective
under a ghost region: the rearranging loop walks the shape from the body
pointer, which sits `nghost` positions above the start of the storage, so it
skips the ghost region and runs the same number of positions past the end. The
trailing rows of the result hold uninitialized values.

A region read forms a third group. It neither duplicates nor rearranges the
storage, and it re-bounds the first axis, so it drops `nghost` the way the
rearranging operations do, as the slice section above states.

```python
sarr = solvcon.SimpleArrayFloat64((4, 3))
sarr.nghost = 2
assert sarr.clone().nghost == 2
assert sarr.to_row_major().nghost == 2   # already row-major: a copy
assert sarr.T.nghost == 2
assert sarr.transpose_copy().nghost == 0
assert sarr[...].nghost == 0
```

Carrying `nghost` through a transpose reattaches the partition to a different
axis, because the count is kept while the axes are permuted: the region that
split the old first axis now splits the new one. The count is not rechecked
against the new first-axis extent, so a permutation onto a shorter axis leaves
`nghost` above it and drives `nbody` negative:

```python
sarr = solvcon.SimpleArrayFloat64((4, 3))
sarr.nghost = 4
sarr.transpose()
assert sarr.shape == (3, 4) and sarr.nghost == 4
assert sarr.nbody == -1     # the partition no longer fits the axis
```

```{caution}
The `nghost` setter refuses a count above `shape(0)`, so a transpose is
the only way into that state.  The valid interval of this page,
`[-(shape[0] + nghost), nbody)`, then lies entirely below zero: on the
`(3, 4)` array above it is `[-7, -1)`, so every non-negative index
raises and so does `-1`, which the ghost range nominally covers.
Whether a transpose should permute the
partition with the axes, drop it as the copying transposes do, or
reject a ghosted receiver is an open decision; until it lands, set
`nghost` after transposing rather than before.
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
