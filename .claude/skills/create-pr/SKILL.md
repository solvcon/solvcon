---
name: create-pr
description: Open a solvcon pull request that follows the project PR protocol. Use when the user asks to create, open, or draft a pull request.
---

# Create Pull Request (solvcon)

Authoritative reference for the solvcon PR protocol. The "Pull Request
Guidelines" section of `AGENTS.md` is the cross-reference; flag drift.

## Protocol

1. **Subject** -- imperative, concrete, informative.
2. **Body** -- the shortest prose that covers, in this order:
   what the change does (one or two sentences, present tense); why
   (the problem or issue); how, only when the diff does not make it
   obvious; what was tested and the actual result, when tests were
   run. Describe only what the diff contains. Do not pad with file
   lists the diff already shows. Prefer one to three short paragraphs;
   bullets only when prose is genuinely unreadable (e.g. a benchmark
   matrix).
3. **Sentence style** -- a non-native reader must parse each sentence
   exactly one way:
   - Under 25 words, one claim per sentence, at most 6 sentences per
     paragraph, one topic per paragraph.
   - Active voice and simple tenses. "The hook rejects the commit",
     not "the commit is rejected"; "adds", not "has added".
   - Short common words: "use" not "utilize", "start" not "initiate".
     Plain verbs, no metaphors and no phrasal-verb idioms.
   - One term per thing, everywhere. Do not rotate synonyms.
   - Name the observable effect ("closes the interval", "returns the
     pybind11 type"), not a vague verb ("handles", "manages").
   - No hedges. State what the change does; drop "should", "might",
     "essentially", "some", "several" when a fact or number is known.
   - Keep the article and the "that"; every "it"/"this" needs an
     adjacent referent.
4. **No hard wrap** -- each paragraph is one unbroken line; blank line
   between paragraphs. GitHub reflows; the 79-char source limit does
   not apply.
5. **Issue reference** -- end with "Related to #xxx." or
   "For issue #xxx.". **Never** "close/closes/fixes/resolves #xxx";
   PR text does not drive issue management. *Exception:* a fork
   prototype PR (see `prototype-with-devplan`) omits the reference
   and any upstream link.
6. **Draft by default** -- open as draft unless the user says it is
   ready for review.
7. **Review request is a global comment** -- when ready, the author
   posts a PR comment asking for review. The ready-for-review button
   alone is not a request.
8. **Human authorship** -- present drafted text for the user's review
   before posting. No `Co-Authored-By:` or "Generated with Claude
   Code" trailers.
9. **`[skip-ci]` for agent-only diffs** -- when the diff touches only
   `.claude/`, `.cursor/`, root `CLAUDE.md`/`AGENTS.md`, or
   `contrib/prompt/`, end the body with `[skip-ci]` on its own line
   (works only on its own line, only for repo members). Omit
   otherwise.

## Workflow

1. **Confirm scope** when unclear: issue number, draft or ready,
   one-line gist.

2. **Verify branch state** (main branch is `master`). Run in parallel:
   `git status --porcelain`, `git log --oneline origin/master..HEAD`,
   `git diff --stat origin/master...HEAD`, and
   `git rev-parse --abbrev-ref --symbolic-full-name @{u}`.
   - Dirty tree: show staged/unstaged/untracked. Commit and push
     clearly in-scope files without another staging question. Stop and
     ask about unrelated or ambiguous files. Never `git add -A` or
     `git add .`; stage exact paths.
   - After handling the dirty tree, no commits ahead: abort, the PR
     would be empty.
   - Unpushed: push without repeating confirmation. If behind its
     remote, stop and reconcile before pushing.
   - Fork prototype PR: base and remote are the fork, not `origin`.

3. **Draft subject and body** from the diff and the gist, following
   the protocol above. Cut every sentence the reader does not need.
   End with the issue reference (omit for a fork prototype PR), then
   `[skip-ci]` if protocol item 9 applies. Show the draft in a fenced
   block and wait for approval.

4. **Open the PR.** Quoted heredocs suppress shell expansion, so the
   text needs no escaping:

   ```bash
   title=$(cat <<'TITLE'
   <approved subject>
   TITLE
   )

   body_file=$(mktemp)
   trap 'rm -f "$body_file"' EXIT
   cat >"$body_file" <<'BODY'
   <approved body>
   BODY

   gh pr create --draft \
     --title "$title" \
     --body-file "$body_file"
   ```

   Drop `--draft` only when the user said ready. If the body contains
   a literal `BODY` line, use a unique delimiter. For a fork prototype
   PR, add `--repo <fork> --base <fork-default>`.

5. **After creation.** Report the URL. Remind the user:
   - The global review-request comment is theirs to write and post;
     do not draft it and do not call `gh pr comment`.
   - Add inline annotations on the diff (skip when one-liner-ish):
     non-obvious choices, subtle invariants, intentional edits that
     look accidental, known limitations, and where a tricky diff needs
     careful reading versus mechanical.

## Guardrails

Run both checks after writing `$body_file`, before `gh pr create`.

- **Closing keywords:**

  ```bash
  { printf '%s\n' "$title"; cat "$body_file"; } \
      | grep -iEn '\b(close[sd]?|fix(e[sd])?|resolve[sd]?)[[:space:]]+#[0-9]+'
  ```

  Any hit: rewrite to "Related to #xxx" and re-confirm with the user.

- **Hard-wrapped prose** (easy to violate by reflex after editing
  wrapped source; treat as a mechanical gate):

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

  Non-zero exit: rejoin the paragraph into one line and re-run.
  Code fences, list items, and table rows are exempt.

- **Branch protection.** Never push to `master`/`main`, never
  `--no-verify`. If `gh pr create` fails, surface the error and stop.
- **No fabricated context.** No invented benchmarks, test results, or
  verification claims; only what the user stated or the diff shows.
- **Diff accuracy.** Re-read `git diff origin/master...HEAD` and
  confirm every claim in subject and body matches a hunk.

## Output

- Show the draft in a fenced block before `gh pr create`.
- After creation: `opened: <PR URL> (draft|ready)`.
- If blocked (closing keyword, dirty tree, unpushed branch):
  `blocked: <reason>` and stop.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
