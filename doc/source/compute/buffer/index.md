# Buffers and Arrays

The SimpleArray family provides typed multi-dimensional arrays whose memory is
owned by C++ and shared with Python without copying. The classes are
implemented under `cpp/solvcon/buffer/` and exposed through the `_solvcon`
extension module, so the same contiguous memory backs both the numerical
methods in C++ and control of flow in Python scripting. The arrays are the
data backbone of the numerical calculations in solvcon.

This document defines the desired behavior of the Python API. The family
consists of three groups of classes, all importable from the top-level
`solvcon` package: raw memory buffers, array classes, and collector classes.

## Raw Memory Buffers

Raw memory buffers manage untyped bytes:

- `ConcreteBuffer`: an untyped, fixed-size byte buffer.
- `BufferExpander`: an untyped, growable staging buffer that can hand its
  content over as a `ConcreteBuffer`.

## Array Classes

Typed array classes provide multi-dimensional access on top of a
`ConcreteBuffer`. There are 13 of them, one per element type:

- `SimpleArrayBool`
- `SimpleArrayInt8`, `SimpleArrayInt16`, `SimpleArrayInt32`,
  `SimpleArrayInt64`
- `SimpleArrayUint8`, `SimpleArrayUint16`, `SimpleArrayUint32`,
  `SimpleArrayUint64`
- `SimpleArrayFloat32`, `SimpleArrayFloat64`
- `SimpleArrayComplex64`, `SimpleArrayComplex128`

The dtype-erased class `SimpleArray` wraps any of the typed classes behind a
single Python type. It is constructed with a shape and a dtype string (or from
a numpy `ndarray`) and dispatches operations to the concrete typed class it
holds. The `typed` property returns a typed copy of the wrapped array, and the
`plex` property on a typed array returns an erased copy; neither bridge shares
memory with the original.

The erased class carries the typed interface. Every operation this document
defines behaves the same way on both. A member that is missing from one or
behaves differently there is a defect.

## Collector Classes

Growable typed buffers collect elements one by one:

- `SimpleCollectorBool`
- `SimpleCollectorInt8`, `SimpleCollectorInt16`, `SimpleCollectorInt32`,
  `SimpleCollectorInt64`
- `SimpleCollectorUint8`, `SimpleCollectorUint16`, `SimpleCollectorUint32`,
  `SimpleCollectorUint64`
- `SimpleCollectorFloat32`, `SimpleCollectorFloat64`
- `SimpleCollectorComplex64`, `SimpleCollectorComplex128`

## Relation to Numpy

The SimpleArray Python API is designed against numpy, but not as a clone of
it. The related behavior falls into three categories:

1. Some behavior deliberately matches numpy. The index arithmetic of element
   access, the buffer protocol, and dtype naming follow numpy so that arrays
   move between the two worlds without surprises.
2. Some behavior deliberately diverges from numpy. The arrays serve solvcon
   solvers first: ghost regions extend an array below index zero for halo
   data, alignment is an explicit constructor argument for SIMD kernels, and
   buffer ownership stays explicit instead of numpy's implicit base-object
   chain.
3. Some behavior is converging toward numpy.

## Contents

```{toctree}
:maxdepth: 2

memory
construct
collector
zerocopy
indexing
elementwise
reduce
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
