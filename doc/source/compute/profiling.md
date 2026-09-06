# Profiling

## Benchmark Inspector

In Pilot, select **Profiling > Benchmark Inspector** to open its MDI window.
Reopening the menu returns to the same inspector and its last result.

Choose Matmul, a dtype, and at least one kernel. Enter shapes and element strides
exactly, including batch axes; changing a shape does not recompute its strides.

Warmups are untimed calls. Each round times the requested calls per kernel,
including NumPy. Worker threads requests BLAS/OpenMP threads only in the worker;
libraries may use fewer. Unavailable kernels are recorded as `ineligible`.

**Run** validates inputs and runs the comparison. The activity bar shows work
is running; it does not estimate a percentage. **Save result...** exports the
completed result as JSON. A new run replaces the result available for export.
Completion, failure and **Stop** unlock the inputs. Closing the inspector or
Pilot stops its worker.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
