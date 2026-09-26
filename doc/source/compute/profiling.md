# Profiling

## Benchmark Inspector

In Pilot, select **Profiling > Benchmark Inspector** to open its MDI window.
Reopening the menu returns to the same inspector and its last result.

Choose Matmul, a dtype, and at least one kernel. Enter shapes and element strides
exactly, including batch axes; changing a shape does not recompute its strides.

Warmups are untimed calls. Each round times the requested calls per kernel,
including NumPy. Worker threads requests BLAS/OpenMP threads only in the worker;
libraries may use fewer. Unavailable kernels are recorded as `ineligible`.

**Run** validates inputs and runs the comparison. The progress bar counts
completed work units, not remaining time, and hides when the worker finishes.
Elapsed times of a minute or longer use hours:minutes:seconds.

The chart shows median bars and p5-p95 ranges on a time axis with evenly spaced,
rounded ticks. Hover for all percentiles, including p25 and p75. Each sample is
a round's time divided by its call count, not an individual call's latency.
The summary shows the sample count; few rounds give coarse tail estimates.
Chart and table timings share an automatically selected unit. Narrow windows
allow scrolling to keep table values readable.

**Save result...** exports the completed result as JSON. A new run replaces
the result available for export. Completion, failure and **Stop** unlock the
inputs. Closing the inspector or Pilot stops its worker.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
