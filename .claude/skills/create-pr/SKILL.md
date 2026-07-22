---
name: create-pr
description: Open a solvcon pull request that follows the project PR protocol (concise subject, clear description, "related to #xxx" wording, draft by default, ready-for-review via global comment). Use when the user asks to create, open, or draft a pull request.
---

# Create Pull Request (solvcon)

This file is the authoritative reference for the solvcon PR protocol.
The "Pull Request Guidelines" section of `CLAUDE.md` is the project-wide
cross-reference; flag any drift between the two.

## Protocol the PR must satisfy

1. **Subject** -- concise and informative.
2. **Description** -- clear, short, human-readable global description.
   Write it as prose paragraphs that a person would actually want to
   read. Avoid bullet lists; only fall back to a short bulleted list or
   a table when prose would genuinely be unreadable (e.g. a benchmark
   matrix with many rows). State *what* changed and *why*.
   **Do not hard-wrap paragraphs.** Each paragraph is a single
   unbroken line; separate paragraphs with one blank line. GitHub
   reflows the text to the viewer's width, and mid-sentence line
   breaks at 79/80 columns render as ragged prose in the PR view.
   The 79-char source-code limit does not apply to PR descriptions.
3. **Issue reference** -- end with "Related to #xxx" or "For issue #xxx".
   **Never** use "close #xxx", "closes #xxx", "fixes #xxx", or any closing
   keyword. We do not let PR/commit text drive issue management.
   *Exception:* a private or fork-based prototype PR (see
   `prototype-with-devplan`) omits the issue reference and any upstream link
   entirely, so the draft does not spray backlinks onto the upstream issue.
4. **Draft by default** -- open as draft unless the user explicitly says it
   is ready for review. Pushing the "ready for review" button alone is
   *not* a review request in this project.
5. **Request review with a global comment** -- when the PR is ready, the
   author posts a global PR comment that explicitly asks for review. The
   button push is not quotable in follow-ups, so the comment is required.
6. **Inline annotations** -- the author is expected to add inline review
   annotations to guide the reviewer, unless the diff is one-liner-ish.
   The skill should remind the user; it does not write the annotations.
7. **Human authorship** -- all PR comments are written by humans. Tool
   assistance is OK, but the user must know what the text says. When the
   skill drafts text, present it for the user's review and edits before
   posting.
8. **Skip CI for agent-only changes.** When the diff touches *only*
   agent tooling (the `.claude/` and `.cursor/` trees, root `CLAUDE.md` /
   `AGENTS.md`, and `contrib/prompt/`), end the body with `[skip-ci]` on
   its own line so the `check_skip_ci` workflow skips the heavy CI jobs.
   Omit it if the diff touches any other file. The control string works
   only on its own line and only for a PR opened by a repository owner,
   member, or collaborator.

## Workflow

1. **Confirm scope with the user.** Ask directly when unclear from
   context:
   - Which issue does this PR relate to? (issue number)
   - Is this ready for review, or should it be opened as draft?
   - One-line gist of the change.

2. **Verify branch state.** solvcon's main branch is `master`. Run in
   parallel:
   - `git status --porcelain` -- check for staged or unstaged changes
     and untracked files.
   - `git log --oneline origin/master..HEAD` -- list the commits the PR
     will carry.
   - `git diff --stat origin/master...HEAD` -- summarize the diff
     (three-dot uses the merge base).
   - `git rev-parse --abbrev-ref --symbolic-full-name @{u}` -- check if
     the branch tracks a remote.

   If `git status --porcelain` shows any output, the working tree is
   not clean. Show the user the list (staged, unstaged, and untracked
   separately) and ask explicitly how to proceed: stage and commit
   selected files, stash, or abort. Never run `git add -A` or
   `git add .` without confirmation -- pick specific paths so that
   stray files (local settings, generated artifacts) are not pulled
   into the PR.

   If the branch has no commits ahead of `origin/master`, abort -- the
   PR would be empty.

   If the branch is not pushed (or is behind its remote), push after
   confirming with the user.

   For a private/fork prototype PR (see `prototype-with-devplan`), the base
   is the fork's own default branch and the tracked remote is the fork, not
   `origin`/upstream; compare against and open the PR on that repository.

3. **Draft the subject and body.** Inspect the diff and the user's
   gist, then propose a subject and a body. The body should be **short
   prose** -- a person reading it should understand what changed and
   why without scanning a checklist. Prefer one to three paragraphs.
   Reserve bullets for cases where prose would genuinely be unreadable
   (long enumerations, benchmark matrices). End with the closing line
   `Related to #xxx.` or `For issue #xxx.` -- but omit that line, and any
   upstream reference, for a private/fork prototype PR (see
   `prototype-with-devplan`).

   If the diff is agent-tooling-only (protocol item 8), add `[skip-ci]`
   on its own line at the very end of the body, one blank line below the
   `Related to #xxx.` closing line.

   **Write each paragraph as one continuous line.** Do not insert
   hard line breaks inside a paragraph -- not at 79 columns, not at
   any column. Paragraphs are separated by exactly one blank line.
   GitHub wraps to the viewer's width; pre-wrapped prose looks
   ragged in the rendered PR. This applies to the draft you show
   the user and to the text written into `$body_file` in step 4.

   Present the draft to the user (subject and body) and wait for edits
   or approval before opening the PR. Do not post Claude-Code
   attribution trailers in the PR body -- this project treats PR text
   as human-authored.

4. **Open the PR.** Load the approved title and body into shell
   variables using **quoted heredocs** -- the single-quoted delimiter
   suppresses every form of shell expansion, so the placeholder text
   can contain backticks, `$`, `"`, or apostrophes without escaping.
   Then pass the title as a double-quoted variable expansion (safe
   because the value is already in memory) and the body via a temp
   file:

   ```bash
   title=$(cat <<'TITLE'
   <approved subject>
   TITLE
   )

   body_file=$(mktemp)
   trap 'rm -f "$body_file"' EXIT
   cat >"$body_file" <<'BODY'
   <approved body, already ending with "Related to #xxx." or
   "For issue #xxx." from step 3, plus a trailing "[skip-ci]" line for
   agent-only changes>
   BODY

   gh pr create --draft \
     --title "$title" \
     --body-file "$body_file"
   ```

   Drop `--draft` only when the user has explicitly said the PR is
   ready for review. Do not append a second issue-reference footer here;
   the body from step 3 already has one. The `trap` ensures the temp
   file is removed on any exit path. If the approved body would itself
   contain the literal token `BODY` on its own line, swap the delimiter
   to something unique (e.g. `BODY_EOF_811`).

   For a fork prototype PR, pass `--repo <fork> --base <fork-default>` so the
   PR opens on the fork; `gh pr create` otherwise defaults to the parent
   (upstream) repository.

5. **After creation.** Report the PR URL back to the user. Then:
   - Remind the user that the global review-request comment is the
     author's responsibility -- the skill does **not** post it. When
     the PR is ready (now, or later after leaving draft), the author
     must open a global PR comment that explicitly asks the
     maintainer to review. Point the user at the PR URL and let them
     write and post the comment themselves; do not draft text for it
     and do not call `gh pr comment`.
   - Remind the user to add inline annotations on the diff (the skill
     does not write them). Recommend the user focus on the points
     that help the reviewer most:
     - non-obvious design choices and the alternatives considered;
     - subtle logic, invariants, or ordering constraints a reader
       might overlook;
     - changes that look like dead code or accidental edits but are
       intentional, so the reviewer does not "fix" them;
     - known limitations, follow-up work, and test-coverage gaps;
     - tricky diffs (large reformatting next to substantive edits,
       cross-file renames) where the reviewer needs guidance on what
       to inspect carefully versus what is mechanical.
     Skip the reminder when the diff is genuinely one-liner-ish.

## Guardrails

- **Closing keywords.** After loading `$title` and writing `$body_file`
  in step 4, but **before** the `gh pr create` call, scan title and
  body together with a single case-insensitive check:

  ```bash
  { printf '%s\n' "$title"; cat "$body_file"; } \
      | grep -iEn '\b(close[sd]?|fix(e[sd])?|resolve[sd]?)[[:space:]]+#[0-9]+'
  ```

  Any hit means the text uses a banned closing keyword (close/closes/
  closed/fix/fixes/fixed/resolve/resolves/resolved followed by `#nnn`).
  Rewrite the offending line to "Related to #xxx" or "For issue #xxx"
  and re-confirm with the user before retrying.
- **Hard-wrapped prose.** The 79-char source limit does NOT apply to a
  PR body; each paragraph must be one unbroken line. This is easy to
  violate by reflex after editing wrapped source all session, so treat
  it as a mechanical gate, not a preference. After writing `$body_file`
  in step 4, but **before** `gh pr create`, scan for a paragraph split
  across lines -- any two consecutive non-blank prose lines:

  ```bash
  awk '
    /^```/            { fence = !fence; prev = 0; next }
    fence             { next }
    /^[[:space:]]*$/  { prev = 0; next }
    /^[[:space:]]*([-*+>|]|[0-9]+\.)/ { prev = 0; next }
    { if (prev) { print "hard-wrap at line " NR ": " $0; hit = 1 }
      prev = 1 }
    END { exit hit }
  ' "$body_file"
  ```

  A non-zero exit means a paragraph was wrapped. Rejoin it into a single
  line (fenced code blocks, list items, and table rows are exempt and
  the check skips them) and re-run before creating the PR. Apply the
  same one-line-per-paragraph rule to the draft shown in step 3, not
  only to `$body_file`.
- **Branch protection.** Never push directly to `master`/`main`, never
  `--no-verify`. If `gh pr create` fails, surface the error and stop --
  do not work around it.
- **No fabricated context.** Do not invent benchmark numbers, test
  results, or verification claims. Only include what the user has stated
  or what is visible in the diff/commits.
- **Diff accuracy.** Before presenting the draft, re-read
  `git diff origin/master...HEAD` and confirm every claim in the subject
  and body corresponds to a hunk in the diff. Drop claims about behavior
  that already exists upstream or that the diff does not change.
- **Trailers.** Do not append `Co-Authored-By:` or "Generated with Claude
  Code" trailers to the PR body. Commits in this project are
  human-authored by convention.

## Output

- Show the draft subject and body to the user before calling `gh pr
  create`. Use a fenced block so it is easy to edit.
- After creation, output a single line: `opened: <PR URL> (draft|ready)`.
- If guardrails block the action (closing keyword, dirty tree, unpushed
  branch), output `blocked: <reason>` and stop. Do not retry silently.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
