# General rules

These rules apply to every task in this project unless explicitly overridden.
Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

## Rule 1 -- Think Before Coding
State assumptions explicitly. If uncertain, ask rather than guess. Present
multiple interpretations when ambiguity exists. Push back when a simpler
approach exists. Stop when confused. Name what's unclear.

## Rule 2 -- Simplicity First
Minimum code that solves the problem. Nothing speculative. No features beyond
what was asked. No abstractions for single-use code. Test: would a senior
engineer say this is overcomplicated? If yes, simplify.

## Rule 3 -- Surgical Changes
Touch only what you must. Clean up only your own mess. Don't "improve" adjacent
code, comments, or formatting. Don't refactor what isn't broken. Match existing
style.

## Rule 4 -- Goal-Driven Execution
Define success criteria. Loop until verified. Don't follow steps. Define
success and iterate. Strong success criteria let you loop independently.

## Rule 5 -- Use the model only for judgment calls
Use Claude for: classification, drafting, summarization, extraction. Do NOT use
Claude for: routing, retries, deterministic transforms. If code can answer,
code answers. In this repo, hooks under `.claude/hooks/` own the deterministic
checks (ASCII, trailing whitespace, modeline, line length); skills own the
judgment calls.

## Rule 6 -- Token budgets surfaced, not hidden
Watch context usage (visible in the status line as a percentage). If
approaching the budget, summarize and start fresh. Surface the breach. Do not
silently overrun.

## Rule 7 -- Surface conflicts, don't average them
If two patterns contradict, pick one (more recent / more tested). Explain why.
Flag the other for cleanup. Don't blend conflicting patterns.

## Rule 8 -- Read before you write
Before adding code, read exports, immediate callers, shared utilities.
Specifically for solvcon: before editing a `wrap_*.cpp` read its corresponding
C++ source; before editing `SimpleArray` / `ConcreteBuffer` / `BufferExpander`
check their `buffer/pymod/` wrappers; before changing a pybind11 binding check
the Python tests in `tests/` that exercise it. "Looks orthogonal" is dangerous.
If unsure why code is structured a way, ask.

## Rule 9 -- Tests verify intent, not just behavior
Tests must encode WHY behavior matters, not just WHAT it does. A test that
can't fail when business logic changes is wrong.

## Rule 10 -- Checkpoint after every significant step
Summarize what was done, what's verified, what's left. Don't continue from a
state you can't describe back. If you lose track, stop and restate. After
running `make` targets (`build`, `pytest`, `gtest`, `lint`), state an
explicit verdict (clean / issues found / blocking) and treat it as a
checkpoint.

## Rule 11 -- Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase. If you genuinely think a convention is
harmful, surface it. Don't fork silently. `STYLE.md` is canonical for style;
`CLAUDE.md` summarizes. If they disagree, prefer `STYLE.md` and flag the drift
in your verdict.

## Rule 12 -- Fail loud
"Completed" is wrong if anything was skipped silently. "Tests pass" is wrong if
any were skipped. Default to surfacing uncertainty, not hiding it.

## Rule 13 -- Verify against ground truth
A claim is a hypothesis until checked against the actual codebase, the live
system, or rendered output. Do not act on another agent's report, a stale
summary, or a single suspicious tool reading (e.g. version-skewed CLI output);
re-check with a direct command first. Before declaring a task blocked or
impossible, try at least two different approaches and report exactly what was
tried and the errors seen.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
