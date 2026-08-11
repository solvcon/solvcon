---
name: code-review
description: Review a diff, branch, path, commit, or pull request for actionable correctness defects. Use when the user invokes $code-review or asks for a code review. Keep the review read-only unless the user explicitly requests fixes or comments.
---

# Code Review

Resolve the exact review target first. For current work, include staged,
unstaged, and untracked changes. For a pull request, inspect its real base,
head, diff, and relevant discussion.

Use code graph tools to trace changed symbols, callers, and invariants. Read the
minimum surrounding code and tests needed to prove each finding. Focus on bugs,
regressions, unsafe behavior, missing validation, and tests that fail to cover
a material risk. Do not report style preferences or speculative concerns.

Rank findings by severity. For every finding, cite exact file and line evidence,
describe the failing scenario, and explain the impact. If no actionable issue
is found, say so explicitly and note any verification gap.

Remain read-only by default. Apply fixes or post review comments only when the
user explicitly asks, then re-review the resulting live diff.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
