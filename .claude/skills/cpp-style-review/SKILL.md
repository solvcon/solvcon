---
name: cpp-style-review
description: Apply solvcon's judgment-call C++ style rules (m_ prefix, class/struct/namespace ending marks, function-body placement, SimpleCollector preference, pybind11 binding split, const_cast) to changed lines in cpp/ or gtests/. Use after editing C++ sources.
tools: Read, Grep, Glob, Edit, Bash
---

# C++ Style Review (solvcon)

Authoritative reference is `STYLE.md` at the repo root; `CLAUDE.md` is a
summary. If they disagree, follow `STYLE.md` and flag the drift in the
verdict.

## Scope

Review only lines that appear in `git diff` against the merge base (or `HEAD`
if explicitly requested). Do NOT flag pre-existing violations on unchanged
lines -- they are out of scope per Rule 3 (surgical changes).

Deterministic checks (ASCII bytes, trailing whitespace, modeline at EOF,
include-style, line length) are handled by `.claude/hooks/check-source.sh`
(PostToolUse). Do not duplicate them. If the hook somehow missed one, mention
it briefly but don't re-implement the check here.

## Judgment-call rules

**Naming**
- Classes / structs: `CamelCase`.
- Functions and variables: `snake_case`.
- Member variables: `m_snake_case`. Flag any class member without the `m_`
  prefix.
- Constants: `UPPER_CASE`, or `snake_case` when interoping with foreign code
  (the rationale should be evident from context; question it when it isn't).
- Type aliases: `snake_case_t` or `snake_case_type`.

**Type casting**
- `const_cast` is suspect. If introduced in the diff, ask whether it can be
  removed.

**Containers**
- Prefer `SimpleCollector` over `std::vector` when `value_type` is a
  fundamental type.
- Prefer `small_vector` for small data.
- STL containers in non-prototype member data require a `TODO` comment plus a
  follow-up PR/issue link.
- STL in local variables is tolerated but discouraged.

**Ending marks (easy to forget -- check every class, struct, and namespace)**
- Per STYLE.md "C++ Ending Mark", every class, struct, and namespace closing
  brace carries a trailing comment naming what it closes:
  `}; /* end class MyClass */`, `}; /* end struct MyStruct */`,
  `} /* end namespace solvcon */` (anonymous: `} /* end namespace */`).
- These are routinely omitted (by humans and by AI). If the diff adds or
  changes a class, struct, or namespace, verify the closing brace has the mark
  and that the name matches the declaration.
- Flag: a missing mark; a wrong name; the wrong keyword (`class` vs `struct`
  vs `namespace`); a `//` mark instead of `/* ... */`; and stale wording. The
  form is exactly `end <keyword> <name>` -- not `end of ...`, and not
  `/* namespace X */` with the word "end" dropped.
- A single-line definition (open and close on one line, e.g.
  `struct is_complex : std::false_type {};`) needs no mark.

**Function-body placement**
- Move non-accessor function bodies outside the class declaration when the body
  is more than ~2-3x the size of an accessor.
- Keep short accessors inline.
- Trivial bodies (single `return`, single assignment) as one-liners.

**Line economy**
- Prefer fewer lines per STYLE.md. Flag unnecessary blank lines inside
  short blocks and needlessly spread-out code. Do not flag structural
  blank lines (between functions, logical sections, access specifiers).
- Enforce STYLE.md's two hard rules: never trade line-width conformance
  for fewer lines, and never put two consecutive executable statements
  (separated by `;`) on one line. A single-statement inline accessor body
  is one statement, not two, and stays the preferred form.

**Comments**
- **Comments are very important.** Check all comments in the diff for clarity, accuracy, and relevance. Flag any comment that is unclear, misleading, trivial, or outdated.
- **Doxygen markers (check every changed comment, not just its content).**
  Per STYLE.md, use `/** ... */` for any block of two or more lines, `///`
  for a one-line brief on a declaration, and `///<` for a trailing member or
  enumerator brief. Flag a multi-line `///` block (two or more consecutive
  `///` lines forming one brief) and tell the author to convert it to
  `/** */`. Judging a comment's wording is not enough; verify the marker too.
- Refer to "C++ Comment" in STYLE.md for what counts as a comment and how to judge it.

**pybind11**
- Split constructors from other bindings (methods, properties) into two
  distinct `(*this)` sections.

## Workflow

1. `git diff --name-only` against the merge base; filter to
   `cpp/**/*.{cpp,hpp,c,h}` and `gtests/**/*.cpp`.
2. For each file, read only the diff hunks (use `git diff` output).
3. Apply the rules above to changed lines.
4. Output each finding as `path:line -- rule -- (fix applied | suggestion):
   <description>`.
5. End with a single verdict line: `verdict: clean | issues found | blocking`.
   Use `clean` only when no findings remain after any hand-fixes.

`blocking` is reserved for things `make lint` would reject (which the hooks
already cover). Findings from this skill are typically `issues found`.

## Output

- Bullets only. No prose summaries.
- Don't paste long code excerpts; point to `file:line`.
- Be explicit when uncertain ("not sure whether X is intentional -- please
  confirm").
- For clang-format violations, don't hand-fix -- suggest
  `make FORCE_CLANG_FORMAT=inplace cformat` to auto-fix. For `cinclude`
  findings (include ordering, angle brackets) and other non-auto-fixable
  nits, try to hand-fix.

Do not run `make pyformat` or `make format`. They are still work in progress.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
