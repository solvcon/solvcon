# Elementwise Arithmetic, Comparison, and Selection

Every typed array in the SimpleArray family computes elementwise: arithmetic
through the `add`, `sub`, `mul`, and `div` method families, comparison through
`eq`, `ne`, `lt`, `le`, `gt`, and `ge` together with the rich-comparison
operators, and selection through `argwhere` and `where`. This page defines
those operations, and it is where the family stands farthest from numpy:
arithmetic is spelled as method calls under strict operand rules, while
comparison and selection are converging to numpy semantics per the policy of
{doc}`the family overview <index>`. The table at the center of the page gives
the spellings of each family side by side.

## Arithmetic Methods

`add`, `sub`, `mul`, and `div` compute a new array elementwise from the
receiver and one operand, leaving the receiver untouched. The operand is
either an array of exactly the same class or a scalar:

```python
narr1 = np.array([1, 2, 3], dtype='float64')
narr2 = np.array([10, 20, 30], dtype='float64')
sarr1 = solvcon.SimpleArrayFloat64(array=narr1)
sarr2 = solvcon.SimpleArrayFloat64(array=narr2)

sres = sarr1.add(sarr2)
assert sres.ndarray.tolist() == [11.0, 22.0, 33.0]
sres = sarr1.mul(2.0)
assert sres.ndarray.tolist() == [2.0, 4.0, 6.0]
assert sarr1.ndarray.tolist() == [1.0, 2.0, 3.0]  # receiver unchanged
```

The result has the shape and element type of the receiver, and the operations
run over the whole storage, ghost region included, over the partition that
{doc}`Indexing, Shape, and Layout <indexing>` defines.

### Operand Rules

An array operand must be the same typed class as the receiver: there is no
dtype promotion, so adding a `SimpleArrayFloat32` to a `SimpleArrayFloat64`
raises `TypeError` from the binding's argument matching, where numpy would
promote both sides to `float64`. The shapes must also be identical: there is
no broadcasting, and a mismatch raises `ValueError` naming the operation and
both shapes:

```python
lhs = solvcon.SimpleArrayFloat64((2, 3), value=1)
rhs = solvcon.SimpleArrayFloat64((3, 2), value=2)
lhs.add(rhs)
# ValueError: SimpleArray::add(): shape mismatch: this=(2, 3)
# other=(3, 2)
```

Both restrictions diverge from numpy, whose arithmetic promotes dtypes and
broadcasts shapes. Whether the family should grow promotion or broadcasting is
an open decision; this page records only the strict rules.

A scalar operand converts to the element type through pybind11 conversion,
following the same rules as scalar assignment in
{doc}`Indexing, Shape, and Layout <indexing>`: a Python `int` converts to a
floating-point element, but a `float` operand on an integer array raises
`TypeError` instead of truncating.

### Integer Division Keeps the Dtype

`div` always produces the receiver's element type. On the integer classes the
quotient is the C++ integer quotient, truncated toward zero, where numpy
`true_divide` promotes to `float64`:

```python
sarr1 = solvcon.SimpleArrayInt32(array=np.array([7, 8], dtype='int32'))
sarr2 = solvcon.SimpleArrayInt32(array=np.array([2, 2], dtype='int32'))
assert sarr1.div(sarr2).ndarray.tolist() == [3, 4]
assert np.divide(sarr1.ndarray, sarr2.ndarray).tolist() == [3.5, 4.0]
```

### Boolean Arithmetic

On `SimpleArrayBool` the arithmetic reduces to logic, matching the numpy
behavior for `add` and `mul`: `add` is elementwise logical or and `mul` is
elementwise logical and. `sub` and `div` are unsupported and raise
`RuntimeError`; numpy also rejects boolean subtraction, with `TypeError`.
Because the methods delegate to their in-place forms, the message names the
in-place spelling even for the copying call:

```python
sarr1 = solvcon.SimpleArrayBool(array=np.array([True, False]))
sarr2 = solvcon.SimpleArrayBool(array=np.array([True, True]))
assert sarr1.add(sarr2).ndarray.tolist() == [True, True]
assert sarr1.mul(sarr2).ndarray.tolist() == [True, False]
sarr1.sub(sarr2)
# RuntimeError: SimpleArray<bool>::isub(): boolean value doesn't
# support this operation
```

The complex classes bind the four methods with the C++ complex arithmetic;
their scalar operands take the complex scalar types of
{doc}`Zero-Copy between C++ and Python <zerocopy>`. The Python built-in
`complex` and the numpy complex scalars are rejected with `TypeError`: no
implicit conversion is registered for the arithmetic operands, unlike element
writes, which accept both spellings.

## In-Place Methods

`iadd`, `isub`, `imul`, and `idiv` apply the same elementwise update to the
receiver itself. They take the same operands under the same rules as the
copying forms, including the same-class and same-shape requirements, and a
failed validation leaves the receiver untouched:

```python
sarr = solvcon.SimpleArrayFloat64(array=np.array([1.0, 2.0]))
sarr.iadd(10.0)
assert sarr.ndarray.tolist() == [11.0, 12.0]
```

The methods return `None`: the mutation is the whole effect. This diverges
from the numpy augmented operators, whose `a += b` statement rebinds `a` to
the returned array. Whether the methods should instead return the receiver to
support chaining is an open decision recorded in the binding source; do not
rely on the return value.

## No Arithmetic Operators

The arithmetic operators are not bound: `+`, `-`, `*`, `/` and their augmented
forms `+=`, `-=`, `*=`, `/=` all raise `TypeError`:

```python
sarr1 = solvcon.SimpleArrayFloat64(3, value=1.0)
sarr2 = solvcon.SimpleArrayFloat64(3, value=2.0)
sarr1 + sarr2
# TypeError: unsupported operand type(s) for +:
# '_solvcon.SimpleArrayFloat64' and '_solvcon.SimpleArrayFloat64'
```

This diverges from numpy, where the operators are the primary spelling.
Whether the family should bind them is an open decision: unlike the comparison
operators below, no converged target is on record, so the method calls remain
the only spelling.

## The `abs` Method

`abs()` returns a new array of the same class holding the elementwise absolute
value, leaving the receiver untouched. On the signed integer and
floating-point classes it matches `numpy.absolute`; on `SimpleArrayBool` and
the unsigned classes the values are already non-negative and the result is a
plain copy, as in numpy:

```python
sarr = solvcon.SimpleArrayInt64(shape=(3, 2), value=-2)
assert sarr.abs().sum() == 12
```

On the complex classes `abs()` currently returns an unchanged copy, where
`numpy.absolute` computes the elementwise magnitude in the matching
floating-point dtype. Whether complex `abs()` should compute the magnitude,
which would change the result's element type, is an open decision. The
`__abs__` protocol is not bound, so the built-in `abs(sarr)` raises
`TypeError` on every class.

## The SIMD Variants

`add_simd`, `sub_simd`, `mul_simd`, `div_simd` and the in-place `iadd_simd`,
`isub_simd`, `imul_simd`, `idiv_simd` are performance-explicit aliases of the
plain forms: they route through the runtime-dispatched SIMD kernels, and the
desired numerics are identical to the plain spellings. On the numeric classes,
element types without a vector kernel fall back to the generic implementation
with the same results.

On `SimpleArrayBool` the two groups currently differ. The in-place variants
delegate to the plain forms and keep the boolean semantics of the arithmetic
methods above; the copying variants bypass that boolean guard and run the
numeric kernels over the boolean elements, so `add_simd` and `mul_simd` still
land on the logical or and and, but `sub_simd` and `div_simd` return
integer-style values where the plain `sub` and `div` raise `RuntimeError`. The
boolean guard of the plain forms is the evident intent, so extending it to the
copying variants is target behavior.

The operand rules differ from the plain forms in one way: the SIMD variants
take only an array operand, so a scalar raises `TypeError`. The same-class and
same-shape requirements are those of the plain forms, with the variant name in
the mismatch message (`SimpleArray::add_simd(): shape mismatch: ...`), and the
in-place variants return `None` like their plain counterparts:

```python
sarr1 = solvcon.SimpleArrayInt32(array=np.arange(5, dtype='int32'))
sarr2 = solvcon.SimpleArrayInt32(array=np.arange(5, dtype='int32'))
assert sarr1.add_simd(sarr2).ndarray.tolist() == [0, 2, 4, 6, 8]
```

## Comparison

### The Comparison Methods

`eq` and `ne` are bound on every typed class; `lt`, `le`, `gt`, and `ge` are
bound on every non-complex class, including `SimpleArrayBool`. Each takes an
array operand of the same class and shape or a scalar, and returns a
`SimpleArrayBool` of the receiver's shape holding the elementwise result:

```python
narr = np.array([1, 2, 3, 3], dtype='int32')
sarr = solvcon.SimpleArrayInt32(array=narr)
sres = sarr.eq(3)
assert type(sres) is solvcon.SimpleArrayBool
assert sres.ndarray.tolist() == [False, False, True, True]
```

A shape mismatch on the array operand raises `ValueError` with the same
message form as the arithmetic methods
(`SimpleArray::ne(): shape mismatch: this=(8) other=(3)`).

### The Bound Operators

The rich-comparison operators map to the methods, so `==`, `!=`, `<`, `<=`,
`>`, and `>=` compare elementwise and return a `SimpleArrayBool`, like numpy
and unlike the Python default of identity comparison:

```python
narr1 = np.array([1, 2, 3, 3], dtype='int32')
narr2 = np.array([1, 0, 3, 9], dtype='int32')
sarr1 = solvcon.SimpleArrayInt32(array=narr1)
sarr2 = solvcon.SimpleArrayInt32(array=narr2)

sres = sarr1 == sarr2
assert sres.ndarray.tolist() == [True, False, True, False]
sres = sarr1 < 3
assert sres.ndarray.tolist() == [True, True, False, False]
```

The elementwise semantics match numpy for the supported operands. The operand
scope is what still separates the two: the array operand must be the same
class and shape, with no promotion and no broadcasting.

### Complex Arrays Leave the Ordering Unbound

Ordering is undefined for complex numbers, so the complex classes bind neither
the ordering methods nor the ordering operators; numpy raises `TypeError` for
`<`, `<=`, `>`, `>=` on a complex ndarray, and the unbound operators make the
classes behave the same way. Equality stays elementwise:

```python
narr = np.array([1 + 2j, 3 + 4j], dtype='complex128')
sarr = solvcon.SimpleArrayComplex128(array=narr)
assert not hasattr(sarr, 'lt')
sarr < sarr
# TypeError: '<' not supported between instances of ...
assert (sarr == sarr).ndarray.tolist() == [True, True]
```

### NaN Comparison

NaN follows the IEEE 754 rules, matching numpy: the ordering comparisons and
`eq` report `False` against NaN in every direction, including NaN against NaN,
and `ne` reports `True`:

```python
narr = np.array([np.nan, 1.0], dtype='float64')
sarr = solvcon.SimpleArrayFloat64(array=narr)
assert sarr.eq(sarr).ndarray.tolist() == [False, True]
assert sarr.ne(sarr).ndarray.tolist() == [True, False]
```

### Unsupported Operands

When the right-hand side of `==` or `!=` is neither a same-class array nor a
convertible scalar, the binding reports the comparison as not implemented and
Python falls back to its default identity semantics, returning a plain `bool`
instead of an array:

```python
sarr = solvcon.SimpleArrayInt32(array=np.arange(3, dtype='int32'))
assert (sarr == None) is False        # the identity fallback
assert (sarr != "not an array") is True
```

A SimpleArray of a different class is such an operand, so comparing
mixed-dtype arrays with `==` yields the identity `False` where numpy would
promote and compare elementwise; the convergence tag covers this gap. The
ordering operators have no identity fallback in Python, so the same
unsupported operands raise `TypeError` there.

## The Spelling Table

The table gives one row per operation family: the member-call form, the
operator form, and the numpy counterpart. "none" marks a spelling that does
not exist; the SIMD row has no numpy counterpart at all.

```{list-table}
:header-rows: 1
:widths: 25 29 17 29

* - Family
  - Member call
  - Operator
  - Numpy form
* - Arithmetic
  - `a.add(b)`, `a.add(s)`
  - none
  - `a + b`, `a + s`
* - In-place arithmetic
  - `a.iadd(b)`, `a.iadd(s)`
  - none
  - `a += b`, `a += s`
* - SIMD arithmetic
  - `a.add_simd(b)`
  - none
  - none
* - Absolute value
  - `a.abs()`
  - none
  - `np.absolute(a)`
* - Equality
  - `a.eq(b)`, `a.ne(b)`
  - `a == b`, `a != b`
  - `a == b`, `a != b`
* - Ordering
  - `a.lt(b)`, ..., `a.ge(b)`
  - `a < b`, ..., `a >= b`
  - `a < b`, ..., `a >= b`
* - Index selection
  - `barr.argwhere()`
  - none
  - `np.argwhere(a == v)`
* - Value selection
  - `cond.where(x, y)`
  - none
  - `np.where(cond, x, y)`
```

The arithmetic rows differ from numpy in the spelling and the operand rules,
and no target is committed for them, per the open decisions above. The
absolute-value row agrees except for the complex open decision. For comparison
and selection the elementwise semantics already agree on same-dtype operands;
what still differs is the operand scope, promotion, and broadcasting.

## Selection

Comparison produces a `SimpleArrayBool`, and the two selection operations
consume it: `argwhere` maps the true elements to their indices, and `where`
merges two arrays under the condition. Both are converging to the numpy usage,
with the method-on-the-condition spelling standing in for the numpy free
functions.

### The `argwhere` Method

`argwhere()` returns a `SimpleArrayUint64` of shape `(count, ndim)` holding
one row of indices per selected element, in row-major order, equal to the
numpy `argwhere` result:

```python
narr = np.array([[1, 3, 5, 7, 9],
                 [2, 4, 6, 8, 10],
                 [1, 10, 1, 10, 1]], dtype='float64')
sarr = solvcon.SimpleArrayFloat64(array=narr)
ret = sarr.eq(10).argwhere()
assert (ret.ndarray == np.argwhere(narr == 10)).all()
```

The method is bound on every typed class and selects the nonzero elements, so
the boolean array from a comparison is the intended, tested condition form:
`sarr.eq(10).argwhere()` is the counterpart of `np.argwhere(narr == 10)`.
{doc}`Reductions, Statistics, Sorting, and Searching <reduce>` covers the
method beside `argmin` and `argmax`; this page fixes only its role as the
index-selection half of comparison.

### The `where` Method

`where(x, y)` exists only on `SimpleArrayBool`; on the other classes the
attribute does not exist and Python raises `AttributeError`. The receiver is
the condition, and the result takes `x` where the condition is true and `y`
elsewhere, equal to `np.where(cond, x, y)`:

```python
narr = np.arange(12, dtype='float64').reshape(3, 4)
cond = solvcon.SimpleArrayBool(array=(narr < 6))
x = solvcon.SimpleArrayFloat64(array=(narr + 1))
y = solvcon.SimpleArrayFloat64(array=(narr * 10))
ret = cond.where(x, y)
assert (ret.ndarray == np.where(narr < 6, narr + 1, narr * 10)).all()
```

`where` follows the logical indices of its operands, so it is defined and
verified on non-contiguous and mixed-layout operands, and it composes with the
ghost partition: the condition and both operands must carry the same `nghost`,
and the result preserves it.

The operand rules are strict, diverging from the numpy scalar and broadcast
operands until the convergence completes. `x` and `y` must be SimpleArray
objects of the same dtype and the condition's shape, drawn from the integer
and floating-point classes; boolean and complex operands are not supported.
Each violation raises `ValueError`:

```python
cond = solvcon.SimpleArrayBool((1, 3), value=True)
cond.where(1.0, 2.0)
# ValueError: SimpleArray::where(): x and y must be SimpleArray
x64 = solvcon.SimpleArrayFloat64((1, 3), value=1)
x32 = solvcon.SimpleArrayFloat32((1, 3), value=2)
cond.where(x64, x32)
# ValueError: SimpleArray::where(): x and y must have the same dtype
xs = solvcon.SimpleArrayInt32((1, 2), value=2)
cond.where(xs, xs)
# ValueError: SimpleArray::where(): shape mismatch: condition=(1, 3)
# x=(1, 2) y=(1, 2)
cond.where(cond, cond)
# ValueError: SimpleArray::where(): unsupported dtype
```

Except for `where`, the elementwise kernels of this page address their
elements through the linear storage; the verified scope is contiguous
operands, which every array allocated by the constructors of
{doc}`Construction and Data Types <construct>` satisfies.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
