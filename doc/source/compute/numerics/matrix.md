# Matrix Operations

The SimpleArray family carries a matrix layer on top of the array operations
({doc}`Buffers and Arrays <../buffer/index>`).

## Matrix Multiplication

`matmul(other)` multiplies one- and two-dimensional operands of the same
class: matrix-matrix, matrix-vector, vector-matrix, and vector-vector, with
the operand shapes chained as in numpy. The `__matmul__` protocol is bound to
it, so the `@` operator is the equivalent spelling. The results are verified
equal to `numpy.matmul` with one divergence: the vector-vector product returns
a one-element array of shape `(1,)` where numpy returns a scalar:

```python
a = solvcon.SimpleArrayFloat64(array=np.array([[1., 2.], [3., 4.]]))
b = solvcon.SimpleArrayFloat64(array=np.array([[5., 6.], [7., 8.]]))
assert ((a @ b).ndarray == np.array([[19., 22.], [43., 50.]])).all()

v = solvcon.SimpleArrayFloat64(array=np.array([1., 2.]))
w = solvcon.SimpleArrayFloat64(array=np.array([3., 4.]))
assert v.matmul(w).shape == (1,) and v.matmul(w)[0] == 11.0
```

A mismatched inner dimension and an operand of more than two dimensions each
raise `IndexError`; the shape text uses no spaces:

```python
a.matmul(solvcon.SimpleArrayFloat64((3, 3), value=0.0))
# IndexError: SimpleArray::matmul(): shape mismatch: this=(2,2)
# other=(3,3)
c = solvcon.SimpleArrayFloat64((2, 2, 2), value=0.0)
c.matmul(c)
# IndexError: SimpleArray::matmul(): unsupported dimensions:
# this=(2,2,2) other=(2,2,2). SimpleArray must be 1D or 2D.
```

### In-Place Matrix Multiplication

`imatmul` computes the product and replaces the receiver's content, reshaping
it to the result. It returns `None`, like the in-place arithmetic of
{doc}`the elementwise page <../buffer/elementwise>`, under the same open
decision on returning the receiver.

The `__imatmul__` protocol is bound so that `a @= b` computes, but the binding
returns the updated receiver by value: the statement mutates the original
storage in place and then rebinds `a` to a fresh copy, so a prior alias of `a`
sees the product but no later change made through the rebound name. The
binding's own comment records that the `__i*__` protocols must return the
receiver itself, as the Python data model expects, so doing so is target
behavior; until then, do not rely on `a` and its old aliases staying the same
object across `@=`.

## Constructors and Transforms

`eye(n)` and `scaled_eye(n, scale)` are static methods constructing an `n` by
`n` identity and scaled identity; `n` must be positive or `ValueError` is
raised (`SimpleArray::eye(): size must be greater than 0, but got 0`). `eye`
matches `numpy.eye(n)` in the class dtype; `scaled_eye` has no direct numpy
spelling.

`pow(n)` raises a square matrix to a non-negative integer power by squaring,
with `pow(0)` the identity, matching `numpy.linalg.matrix_power` on that
domain. A negative exponent raises `ValueError` instead of computing the numpy
matrix inverse:

```python
m = solvcon.SimpleArrayFloat64(array=np.array([[1., 2.], [3., 4.]]))
assert (m.pow(2).ndarray == np.linalg.matrix_power(m.ndarray,
                                                   2)).all()
m.pow(-1)
# ValueError: SimpleArray::pow(): exponent must be non-negative, but
# got -1
```

`hermitian()` returns the conjugate transpose as a copy of a two-dimensional
array, equal to `narr.conj().T`; on the non-complex classes it is the
transpose copy. `symmetrize()` averages a square matrix with its (conjugate)
transpose. `trace()` sums the diagonal of a square matrix into a scalar;
numpy's `trace` also accepts non-square input, which these methods reject. A
wrong rank or a non-square shape raises `RuntimeError` naming the requirement:

```python
solvcon.SimpleArrayFloat64(5).trace()
# RuntimeError: SimpleArray::trace(): operation requires 2D
# SimpleArray, but got 1D SimpleArray
solvcon.SimpleArrayFloat64((3, 4), value=0.0).symmetrize()
# RuntimeError: SimpleArray::symmetrize(): operation requires square
# SimpleArray, but got 3x4 shape
```

The bindings register the whole matrix family on every typed class. The tests
exercise `matmul`, `pow`, `eye`, and `scaled_eye` on the floating-point
classes, `hermitian` and `symmetrize` on `complex128`, and `trace` on one
class from each of the floating-point, integer, and complex groups; the other
class-operation combinations follow the same kernels but are unverified.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
