---
name: simplify
description: Review recently changed code for reuse, simplicity, efficiency, and appropriate abstraction, then apply safe behavior-preserving cleanup. Use when the user invokes $simplify or asks to simplify or refine recent changes. Do not use for correctness review.
---

# Simplify

Review the target the user names. Otherwise inspect tracked and untracked
changes with `git status --short --untracked-files=all`, then read the relevant
diffs and new files. If there is no changed code, report that and stop.

Check four things:

1. Reuse existing helpers instead of duplicating logic.
2. Reduce unnecessary control flow, state, indirection, and comments.
3. Remove avoidable work or allocation when the benefit is clear.
4. Keep behavior at the right abstraction level for the surrounding code.

Use code graph tools for relationships and reuse discovery. Read the diff and
the minimum surrounding code needed to validate each cleanup. Do not broaden
the task into bug hunting, feature work, or unrelated refactoring.

Apply only changes that preserve intended behavior and make the code plainly
simpler. Leave uncertain or subjective suggestions unapplied and report them.
Keep edits inside changed code unless a directly reused helper requires a
small adjacent update.

Run focused tests and relevant lint checks after editing. Invoke
`$cpp-style-review` after C++ changes and `$python-style-review` after Python
changes. Review the final diff and revert any cleanup that adds complexity or
changes behavior. Do not commit or open a pull request unless asked.

Report the cleanups applied, verification results, and any suggestions left
for the user.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
