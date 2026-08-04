# Memory Buffers

The raw memory layer of the SimpleArray family consists of two untyped byte
buffers. `ConcreteBuffer` owns a fixed-size block of contiguous memory and is
the storage that every typed array sits on. `BufferExpander` is a growable
staging buffer for code that must accumulate bytes before the final size is
known, and can hand its content over as a `ConcreteBuffer`.

## ConcreteBuffer

`ConcreteBuffer` is an untyped byte buffer: no dtype, no shape, and no stride.
Once constructed, its size never changes.

### Construction

A buffer is constructed with a byte count and an optional alignment:

```python
buf = solvcon.ConcreteBuffer(1024)
buf = solvcon.ConcreteBuffer(1024, alignment=64)
```

The memory is allocated immediately and is not initialized. The `nbytes`
property returns the byte count and `alignment` returns the requested
alignment. Valid alignment values are 0 (the default, no specific alignment),
16, 32, and 64 bytes; any other value raises `ValueError`. When a non-zero
alignment is requested, the byte count must be a multiple of the alignment, or
the allocation raises `ValueError`. A zero-byte buffer is permitted and
records the requested alignment without allocating.

### Size and Byte Access

The buffer sizes and indexes like a Python sequence of bytes. `len(buf)`
equals `buf.nbytes`. Indexing with `buf[i]` reads one byte and assignment
`buf[i] = value` writes one byte; the element type is a signed 8-bit integer.
Indexing at or beyond the size raises `IndexError`:

```python
buf = solvcon.ConcreteBuffer(10)
for it in range(len(buf)):
    buf[it] = it
buf[10]  # IndexError: ConcreteBuffer: index 10 is out of bounds ...
```

Iteration follows from indexing, so `list(buf)` returns the byte values.

### Sharing with Numpy

`ConcreteBuffer` implements the Python buffer protocol, exposing its memory as
a one-dimensional `int8` sequence. A numpy array built on the buffer shares
the memory rather than copying it:

```python
import numpy as np
buf = solvcon.ConcreteBuffer(10)
ndarr = np.array(buf, copy=False)  # dtype int8, shape (10,)
buf[3] = 7
assert ndarr[3] == 7
```

The `ndarray` property is a shortcut that returns such an `int8` view
directly. The returned array holds a reference to the buffer, so the memory
stays alive as long as the array does.

### Cloning

`clone()` returns a deep copy: a new buffer with the same byte count, the same
alignment, and a copy of the content. Writes to either buffer are not visible
in the other.

### Wrapping a Numpy Array

The second constructor form wraps the memory of an existing numpy array
instead of allocating:

```python
ndarr = np.arange(24, dtype='float64').reshape((2, 3, 4))
buf = solvcon.ConcreteBuffer(array=ndarr)
assert buf.nbytes == ndarr.nbytes
```

The dtype and shape of the source array do not matter, but the array must be
contiguous; the buffer covers its `nbytes` bytes and shares them zero-copy, so
writing through one side is visible on the other. The constructor also accepts
the optional `alignment` keyword, validated against the same set (0, 16, 32,
or 64); no size-multiple check applies because nothing is allocated. The
buffer holds a reference to the source array, tying the lifetime of the memory
to the Python object. The `is_from_python` property reports the provenance: it
is `True` for a buffer wrapping a numpy array and `False` for a buffer that
allocated its own memory.

## BufferExpander

`BufferExpander` is an untyped byte buffer that grows. It distinguishes its
size (the bytes currently in use) from its capacity (the bytes allocated). Its
role is a staging area: accumulate bytes while the final length is unknown,
then hand the result over as a `ConcreteBuffer` for fixed-size access. The
internal expandable memory is never exposed to other components.

### Construction

Three forms are supported:

```python
ep = solvcon.BufferExpander()          # empty: size 0, capacity 0
ep = solvcon.BufferExpander(10)        # size 10, capacity 10
ep = solvcon.BufferExpander(buf)       # copy of a ConcreteBuffer
```

The third form initializes size and capacity to the byte count of the given
`ConcreteBuffer` and copies its content; the expander does not alias the
source buffer, so later writes to the expander leave the source unchanged. The
sized and buffer-copying forms accept an optional `alignment` keyword with the
same valid values as `ConcreteBuffer` (0, 16, 32, or 64); the empty form fixes
the alignment at 0. The read-only `alignment` property returns the value given
at construction.

### Growing

`reserve(cap)` grows the capacity to at least `cap` bytes while keeping the
size and the existing content; when `cap` does not exceed the current capacity
it does nothing. Capacity never shrinks. `expand(length)` reserves `length`
bytes and then sets the size to `length`:

```python
ep = solvcon.BufferExpander()
ep.reserve(10)   # capacity 10, size still 0
ep.expand(10)    # capacity 10, size 10
```

The `capacity` property returns the allocated byte count and `len(ep)` returns
the size.

### Byte Access

Indexing works as on `ConcreteBuffer`: `ep[i]` reads and `ep[i] = value`
writes single signed 8-bit bytes. Bounds are checked against the size, not the
capacity, so a reserved-but-unexpanded region is not addressable and raises
`IndexError`.

### Producing a ConcreteBuffer

Two methods convert the staged bytes into a `ConcreteBuffer`, differing in
whether the result shares memory with the expander:

- `copy_concrete(cap=0)` returns an independent copy. The new buffer holds
  `max(cap, len(ep))` bytes with the staged content copied in; later writes on
  either side do not affect the other.
- `as_concrete(cap=0)` converts the expander in place. The expander adopts a
  `ConcreteBuffer` as its storage and returns it; from then on the two objects
  share memory, so a write through one is visible through the other. When the
  expander is not yet concrete, the adopted buffer holds `max(cap, len(ep))`
  bytes; when it already is, `cap` is ignored and the existing buffer is
  returned.

`clone()` returns a new independent `BufferExpander` with a copy of the
content.

The `is_concrete` property reports whether the expander is currently backed by
a `ConcreteBuffer`. It is `False` for a freshly expanded buffer, `True` after
`as_concrete()`, and `True` for an expander constructed from a
`ConcreteBuffer`:

```python
ep = solvcon.BufferExpander(10)
assert not ep.is_concrete
cbuf = ep.as_concrete()
assert ep.is_concrete
cbuf[0] = 42
assert ep[0] == 42  # memory is shared
```

Growing the capacity after `as_concrete()` reallocates the storage and
detaches the expander from the concrete buffer: `is_concrete` drops back to
`False` and the previously returned buffer keeps the old memory.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
