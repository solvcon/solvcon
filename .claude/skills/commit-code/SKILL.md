---
name: commit-code
description: Commits changes the solvcon way -- splits work into one-concern commits, lints first, stages exact paths, and writes STYLE.md-conformant messages (imperative subject, no semantic prefixes, no closing keywords, human-authored). Use when the user asks to commit, split work into commits, organize or rewrite commit history, or write a commit message.
---

# Commit Code (solvcon)

The "Commit Log" section of `STYLE.md` at the repo root is canonical for
the message format; `CLAUDE.md` summarizes it and `create-pr` covers
pull requests. This skill adds the surrounding workflow and the project
conventions that differ from default git habits.

## Workflow

1. **Survey.** Run `git status --porcelain` and `git diff` (with
   `--stat`) to see what changed. Group the changes into one-concern
   commits. Ask before staging when a file's grouping is unclear or it
   looks unrelated to the task.
2. **Lint.** Run `make lint` (or the subset for the touched language).
   Fix every report and re-run. Never commit code that fails the
   linters.
3. **Commit each concern.** Stage the exact paths (`git add <path>`),
   confirm `git diff --cached --stat` shows only that concern, draft the
   message, present the subject (and body, when non-trivial) for review,
   then commit with a quoted heredoc so backticks, `$`, and quotes
   survive unescaped:

   ```bash
   git commit -F - <<'MSG'
   <approved subject>

   <approved body wrapped at 72, if any>
   MSG
   ```
4. **Verify.** Run `git log --oneline` and report the commits created.

## Message format

Follow STYLE.md "Commit Log" -- do not restate it here. In brief:
imperative subject (capitalized, no period, aim 50 / 72 hard limit), one
blank line, then a body wrapped at 72 that explains what changed and
why, not how. A one-line subject suffices for a trivial change.

## Conventions that differ from default git habits

These contradict common defaults, so apply them deliberately:

- **Human-authored.** Do not append `Co-Authored-By:` or "Generated
  with ..." trailers; this project keeps commits human-authored (see
  `create-pr`).
- **No semantic prefixes.** Never use `feat:`, `fix:`, `docs:`, etc.
- **Backtick every code token.** Paths, `make` and CMake targets,
  identifiers, macros, variables, flags, commands, and literal values
  are quoted in the subject as well as the body.
- **No closing keywords.** Never use `close`/`fixes`/`resolves #n`. Do
  not reference an issue unless explicitly instructed; when you are, end
  the body with "Related to #xxx" or "For issue #xxx".
- **One concern per commit.** Each commit stands on its own. Do not pad:
  trivial, tightly-coupled edits belong together.
- **Stage exact paths.** Never `git add -A` / `git add .`; never
  `--no-verify`. Stray files (local settings, build artifacts) must not
  ride along.
- **Topic branch only.** Never commit directly on `master`/`main`;
  branch first.

## Rewriting history

When asked to organize or clean up commits, recreate them rather than
patching messages one by one:

1. Soft-reset to the merge base: `git reset --soft <base>`, then
   `git restore --staged .`.
2. Recommit by concern as in the workflow above, in an order that reads
   as a coherent progression.
3. Before pushing, confirm the rewritten tree is identical to the prior
   tip: `git diff <old-tip> HEAD` must be empty.
4. Push with `git push --force-with-lease`, never a bare `--force`.

## Guardrail: closing keywords

Before each commit, scan the drafted message and rewrite any hit:

```bash
printf '%s\n' "$msg" \
    | grep -iEn '\b(close[sd]?|fix(e[sd])?|resolve[sd]?)[[:space:]]+#[0-9]+'
```

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
