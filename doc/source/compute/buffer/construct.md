# Construction and Data Types

The SimpleArray family is constructed in two styles. The 13 typed classes fix
the element type in the class name and take a shape; the dtype-erased
`SimpleArray` takes a shape plus a dtype string and wraps the matching typed
class behind a single Python type. Both styles either allocate a
`ConcreteBuffer` or wrap the memory of an existing numpy array. This page
defines the constructor forms, the element types, the `fill` and `clone`
operations, and the alignment extension.

## Element Types

Each typed class binds one C++ element type. The dtype string in the table
names the numpy dtype of the array and selects the typed class in the
`SimpleArray` constructor. The dtype naming matches numpy: the strings are the
numpy dtype names, so an array round-trips to numpy without translation.

| Class | C++ value type | dtype string | | ----------------------- |
----------------- | ------------ | | `SimpleArrayBool` | `bool` | `bool` | |
`SimpleArrayInt8` | `int8_t` | `int8` | | `SimpleArrayInt16` | `int16_t` |
`int16` | | `SimpleArrayInt32` | `int32_t` | `int32` | | `SimpleArrayInt64` |
`int64_t` | `int64` | | `SimpleArrayUint8` | `uint8_t` | `uint8` | |
`SimpleArrayUint16` | `uint16_t` | `uint16` | | `SimpleArrayUint32` |
`uint32_t` | `uint32` | | `SimpleArrayUint64` | `uint64_t` | `uint64` | |
`SimpleArrayFloat32` | `float` | `float32` | | `SimpleArrayFloat64` | `double`
| `float64` | | `SimpleArrayComplex64` | `Complex<float>` | `complex64` | |
`SimpleArrayComplex128` | `Complex<double>` | `complex128` |

The complex classes use solvcon's own `Complex` value type in C++ but expose
the standard numpy complex dtypes through the buffer protocol. The item size
follows the element type, and the numpy view of an array carries the matching
dtype:

```python
import numpy as np
assert solvcon.SimpleArrayInt8((2, 3)).nbytes == 6
assert solvcon.SimpleArrayInt32(7).nbytes == 28
assert solvcon.SimpleArrayFloat64((2, 3, 4)).nbytes == 192
assert solvcon.SimpleArrayFloat64((2, 3)).ndarray.dtype == np.float64
```

## Typed Constructors

Every typed class offers the same five constructor forms, shown here on
`SimpleArrayFloat64`:

```python
solvcon.SimpleArrayFloat64(shape)
solvcon.SimpleArrayFloat64(shape, alignment=64)
solvcon.SimpleArrayFloat64(shape, value=1.0)
solvcon.SimpleArrayFloat64(shape, value=1.0, alignment=64)
solvcon.SimpleArrayFloat64(array=ndarr)
```

The first four allocate; the last wraps existing numpy memory. The sections
below define each form.

### Construction from a Shape

A typed array is constructed from a shape given as a single integer (a
one-dimensional array) or a tuple of integers:

```python
sarr = solvcon.SimpleArrayInt32(7)          # shape (7,)
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
```

The element storage is allocated immediately and is not initialized; the
allocation semantics match `numpy.empty`. The `shape`, `stride`, `size`,
`itemsize`, and `nbytes` properties describe the layout of the row-major
result, and `len()` returns the element count:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
assert sarr.shape == (2, 3, 4)
assert sarr.stride == (12, 4, 1)  # counted in elements, not bytes
assert sarr.size == 24
assert sarr.itemsize == 8
assert sarr.nbytes == 24 * 8
assert len(sarr) == 24
```

`len()` diverges from numpy: it returns the total element count, where numpy
returns the length of the first dimension (24 versus 2 for the shape above).

A negative shape dimension raises `IndexError`:

```python
solvcon.SimpleArrayFloat64((-1, 2))
# IndexError: SimpleArray: shape dimension must be non-negative ...
```

### Fill Value

The second constructor form takes an initial value and fills every element
with it, in the spirit of `numpy.full`:

```python
sarr = solvcon.SimpleArrayFloat64((4, 4), value=3.14159)
assert sarr[0, 0] == 3.14159
```

Pass the value with the `value=` keyword. An integer given positionally in the
second slot resolves to the alignment overload instead of the fill value,
because the alignment constructor is matched first; the keyword removes the
ambiguity.

### Wrapping a Numpy Array

The `array=` keyword form wraps the memory of an existing numpy array instead
of allocating:

```python
ndarr = np.arange(24, dtype='float64').reshape((2, 3, 4))
sarr = solvcon.SimpleArrayFloat64(array=ndarr)
```

The array and the wrapping `SimpleArray` share memory zero-copy, and the
wrapper keeps the source alive by holding a reference. The dtype of the source
must equal the element type of the class; a mismatch raises `RuntimeError`. As
on `ConcreteBuffer`, the `is_from_python` property reports the provenance:
`True` for an array wrapping numpy memory and `False` for an array that
allocated its own buffer.

Unlike the buffer-level wrap, the source does not need to be contiguous.
Strided views are supported, including negative strides from reversed slices;
the stride of the view is recorded in elements, and writes made through the
wrapper land in the viewed region of the original array:

```python
ndarr = np.arange(6, dtype='float64').reshape((2, 3))
view = ndarr[::-1, ::-1]
sarr = solvcon.SimpleArrayFloat64(array=view)
assert sarr.stride == (-3, -1)
sarr[0, 0] = 200.0
assert ndarr[1, 2] == 200.0
```

A source whose byte stride is not divisible by the item size, or whose data
pointer is not aligned for the element type, raises `RuntimeError`. A
Fortran-ordered source is likewise wrapped with its own stride; the
`is_c_contiguous` and `is_f_contiguous` properties report the layout, and a
degenerate shape (a single row, column, or element) reports both as `True`.

## The Dtype-Erased SimpleArray

The Python class `SimpleArray` erases the element type from the class name and
moves it into a `dtype` string argument, closer to the numpy calling
convention:

```python
sarr = solvcon.SimpleArray((2, 3, 4), dtype='float64')
sarr = solvcon.SimpleArray((2, 3, 4), value=3.0, dtype='float64')
sarr = solvcon.SimpleArray(np.arange(6, dtype='int32'))
```

The dtype string selects the typed class per the table above; a string outside
the table raises `ValueError`. The third form infers the dtype from the numpy
array and shares its memory.

The `value=` form diverges from numpy: where `numpy.full` casts the fill value
to the array dtype, `SimpleArray` validates the Python type of the value
strictly and raises `TypeError` on mismatch. A `bool` dtype requires a Python
`bool`, the integer dtypes require a Python `int`, and the floating-point
dtypes require a Python `float`:

```python
solvcon.SimpleArray((2, 3), dtype='bool', value=3.3)
# TypeError: Data type mismatch, expected Python bool
solvcon.SimpleArray((2, 3), dtype='int32', value=3.3)
# TypeError: Data type mismatch, expected Python int
solvcon.SimpleArray((2, 3), dtype='float64', value=3)
# TypeError: Data type mismatch, expected Python float
```

The erased wrapper carries the typed interface, as
{doc}`the family overview <index>` states, so every operation of this document
reads the same on both. The `typed` property converts to the concrete class:
it returns an independent copy of the wrapped array as its typed class, and
the `plex` property on a typed array returns an erased copy the same way.
Neither direction shares memory with the original, so the bridge serves
read-style workflows; a write made through the bridged object stays in the
copy. The content survives the round trip in both directions:

```python
plex = solvcon.SimpleArray((2, 3, 4), dtype='float64', value=1.5)
typed = plex.typed
assert type(typed) is solvcon.SimpleArrayFloat64
assert typed[0, 0, 0] == 1.5
assert type(typed.plex) is solvcon.SimpleArray
```

## Filling

`fill(value)` assigns the value to every element in place, matching numpy
`ndarray.fill`:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
sarr.fill(2.0)
```

On the dtype-erased `SimpleArray`, `fill` applies the same strict value typing
as the `value=` constructor and raises `TypeError` when the Python type of the
value does not match the dtype.

## Cloning

`clone()` returns a deep copy: a new array with the same shape, stride,
alignment, and a copy of the content, on a freshly allocated buffer. The copy
never shares memory with the source, regardless of how the source was
constructed; cloning an array that wraps numpy memory yields an independent
array whose `is_from_python` is `False`:

```python
sarr = solvcon.SimpleArrayFloat64((2, 3, 4), value=2.0)
clone = sarr.clone()
sarr[0, 0, 3] = 3.0
assert clone[0, 0, 3] == 2.0
```

The dtype-erased `SimpleArray` clones the same way and returns another erased
wrapper.

The spelling diverges from numpy: numpy calls this operation `copy()`. The
target behavior is that `clone()` remains the canonical spelling for a deep
copy across the family, and no `copy()` alias is provided.

## Alignment

Alignment is a solvcon-specific extension with no numpy counterpart. Every
allocating constructor form accepts an optional `alignment` argument that
aligns the start of the buffer for SIMD kernels, which require 16-, 32-, or
64-byte alignment depending on the vector width:

```python
sarr = solvcon.SimpleArrayFloat64((4, 4), alignment=16)
sarr = solvcon.SimpleArrayFloat64((4, 4), value=2.7, alignment=16)
sarr = solvcon.SimpleArray((4, 4), dtype='float64', alignment=16)
```

Valid alignment values are 0 (the default, no specific alignment), 16, 32, and
64 bytes; any other value raises `ValueError`. When a non-zero alignment is
requested, the total byte count of the array must be a multiple of the
alignment, or the allocation raises `ValueError`:

```python
solvcon.SimpleArrayFloat64((4, 4), alignment=17)
# ValueError: ... alignment must be 0, 16, 32, or 64, but got 17
solvcon.SimpleArrayFloat64((5, 1), alignment=16)
# ValueError: ConcreteBuffer::allocate: size ... must be a multiple ...
```

The read-only `alignment` property returns the requested alignment, and
`clone()` preserves it. Arrays that wrap numpy memory do not take the
argument, because nothing is allocated. An aligned array behaves like any
other array elsewhere: it fills, computes, converts to numpy, and clones the
same way, and additionally satisfies the memory precondition of the `_simd`
operation variants.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
