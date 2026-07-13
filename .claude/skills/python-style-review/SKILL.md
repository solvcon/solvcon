---
name: python-style-review
description: Apply solvcon's judgment-call Python style rules (naming, project conventions, test intent) to changed lines in solvcon/ or tests/. Use after editing Python sources.
tools: Read, Grep, Glob, Edit, Bash
---

# Python Style Review (solvcon)

Authoritative reference is `STYLE.md` at the repo root; `CLAUDE.md` is a
summary. If they disagree, follow `STYLE.md` and flag the drift in the
verdict.

## Scope

Review only lines that appear in `git diff` against the merge base (or
`HEAD` if explicitly requested). Do NOT flag pre-existing violations on
unchanged lines -- out of scope per Rule 3 (surgical changes).

Deterministic checks (ASCII, trailing whitespace, modeline, 79-char
limit, flake8) are handled by `.claude/hooks/check-source.sh`
(PostToolUse) and `make flake8`. Do not duplicate them.

## Judgment-call rules

**Naming**
- Classes: `CamelCase`.
- Functions and variables: `snake_case`.
- Constants: `UPPER_CASE`.

**Project conventions**
- No venv/conda code paths (solvcon targets system Python).
- Tests live in `tests/` and are named `test_*.py`.
- Profiling scripts live in `profiling/` and are named `profile_*.py`.
- NumPy arrays: always create with an explicit `dtype` spelled as a string,
  e.g. `np.empty(10, dtype='float64')`. Flag array-creating calls (`np.empty`,
  `np.zeros`, `np.ones`, `np.full`, `np.array`, `np.arange`, etc.) that omit
  `dtype` or pass a type object (`dtype=np.float64`) instead of a string.

**Line economy**
- Prefer fewer lines per STYLE.md. Flag unnecessary blank lines inside
  short blocks and needlessly spread-out code. Do not flag structural
  blank lines (between functions, logical sections).
- Enforce STYLE.md's hard rule: never put two consecutive executable
  statements (separated by `;`) on one line. (Line width is owned by the
  hooks; don't re-flag it here.)

**Comments**
- **Comments are very important.** Check all comments in the diff for clarity, accuracy, and relevance. Flag any comment that is unclear, misleading, trivial, or outdated.
- Refer to "Python Comment" in STYLE.md for what counts as a comment and how to judge it.

**Intent (Rule 9)**
- Tests should encode why behavior matters, not just what. If a new test
  would still pass under an obvious bug in the code it exercises,
  question it.

## Workflow

1. `git diff --name-only` against the merge base; filter to `**/*.py`.
2. Read diff hunks only.
3. Apply rules to changed lines.
4. Output each finding as
   `path:line -- rule -- (fix applied | suggestion): <description>`.
5. End with a single verdict line:
   `verdict: clean | issues found | blocking`. Use `clean` only when no
   findings remain after any hand-fixes.

## Output

- Bullets only.
- Don't paste long code excerpts; point to `file:line`.
- Be explicit when uncertain.
- Try to hand-fix formatting nits. `make flake8` verifies but does not fix.

Do not run `make pyformat`, which is not set up to conform to the existing
Python coding style yet.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
