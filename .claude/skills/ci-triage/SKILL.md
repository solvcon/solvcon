---
name: ci-triage
description: Triage a failed solvcon GitHub Actions run into a fix branch and a draft PR on the developer's fork that records the analysis for review before anything goes upstream, and decide whether the draft needs CI or the skip-ci label. Use when the user hands over a failing run or job URL, asks why CI is red, or asks for a fix PR for a CI failure.
---

# CI Triage (solvcon)

Turn a red run into an explained fix. The analysis is the deliverable as much
as the diff: the draft PR must let a reader check every claim against the
log.

Open a draft PR on the developer's fork, not on upstream, for verified fixes,
but leave saved analysis when blocked. The developer reviews the draft, then
submits the final PR to the upstream project (section 8). This skill owns the
triage. It hands off to `worktree` for isolation, `commit-code` for the commit,
and `create-pr` for the PR text.

## 1. Isolate

Use `worktree` first. Keep scripts and ongoing investigation records in a
scratchpad outside the checkout; report its path upfront.

## 2. Read the failure

```bash
gh run view <run-id> -R <owner>/<repo>
gh run view -R <owner>/<repo> --job <job-id> --log-failed > "$out/fail.log"
grep -nE 'Fatal Python|FAILED|Error [0-9]+|Traceback|short test summary' \
    "$out/fail.log" | head -n 40
```

- The first command lists the jobs and marks the failing step; take the job
  id from it. `--log-failed` carries only the failing steps.
- A `make` failure shows up as a chain of `Error 2` lines. Scroll up to the
  first real error, not the last line.
- A segfault prints a faulthandler dump. Read "Current thread" top-down. When
  the top Python frame is an event-loop wait (`QTest.qWait`,
  `processEvents`), the fault is in C++ during event dispatch, not in a
  Python slot, because a slot would add its own frame.
- Record the Python, pytest, and Qt/PySide versions from the log
  (`platform linux -- Python`, the `pip install pyside6==` line). A local
  version mismatch weakens a failed reproduction.

## 3. Classify before reading code

```bash
gh run list -R <owner>/<repo> --workflow <workflow> --limit 25 \
    --json databaseId,event,headBranch,headSha,conclusion,displayTitle
```

Decide which of these it is, and say so in the draft PR:

- **Regression**: fails on every run since one commit. Find that commit in
  the list and read its diff first.
- **Lane gap**: the PR run passed, but the push run covers more. A pull
  request runs `make pytest-fast`, so `tests/gui` runs only inside the pilot
  there, while a push runs it in plain Python as well. Some lanes (the lint
  pilot lane, `devbuild_windows`) never run on a pull request. A master-only
  failure is often this, not a flake.
- **Flake**: the same commit, or code that has not changed, passes elsewhere
  in the same lane. Fetch the logs of earlier failures of the same test with
  the same grep; a different signature is a different bug.
- **Infrastructure**: runner outage, download or cache failure, timeout
  with no test output. Fix the workflow or retry; do not touch the code.

## 4. Reproduce, time-boxed

Build what the failing lane uses. `make buildext` provides the `_solvcon`
module that plain `python3 -m pytest` imports; `make pilot` builds the pilot
binary but does not provide that module.

Use `stress.sh` beside this file to repeat a selection. Point
`CI_TRIAGE_OUT` at the scratchpad to keep its per-run logs there:

```bash
.claude/skills/ci-triage/stress.sh --runs 20 --xcb tests/test_x.py
.claude/skills/ci-triage/stress.sh --runs 5 --pilot -k test_x
.claude/skills/ci-triage/stress.sh --runs 5 --pilot --gc --poison -k test_x
```

- `--pilot` matches `make run_pilot_pytest`. The pilot always appends
  `tests/` to the pytest arguments, so select with `-k`; a file path runs the
  whole suite as well.
- `--xcb` matches the Linux CI lane (xcb under Xvfb). The default offscreen
  platform hides window-system timing.
- `--gc` and `--poison` turn wrapper-lifetime and use-after-free bugs from
  rare into frequent.
- Run the helper as a plain command. In a Claude Code worktree session the
  isolation guard refuses commands it cannot prove stay in the worktree: an
  inline `PYTHONPATH=... python3` prefix, `xargs git`, a computed `sed`
  program, or a long compound chain. Put anything longer in a script in the
  scratchpad and run that.

Stop after about 30 clean runs across the modes above. Report the exact
counts and move on to reading code with the log as evidence. Do not keep
theorizing about binding internals before the cheap runs are done.

## 5. Fix and prove it

- Fix the cause when it is known. When the crash does not reproduce, remove
  the fragile pattern the evidence points at, and call the change a
  mitigation, not a root-cause fix.
- Add a test that fails without the fix. Prove it: stash only the source
  change with a unique tag (`git stash push -m <tag> -- <path>`), run the
  test, then `git stash apply <sha>` and drop that entry by its tag. The
  stash stack is shared with other sessions, so never pop.
- Run the affected tests in the failing lane (inside the pilot for a
  `run_pilot_pytest` failure), `make lint` or the targets it covers, and the
  matching style-review skill.

## 6. Commit and open the draft PR

Commit with `commit-code`. The upstream PR later reuses these commits, so
write their messages for upstream from the start.

Push the branch to the developer's fork and open the draft there, never on
the upstream repository:

```bash
git push -u <fork-remote> <branch>
gh pr create --draft --repo <owner>/<fork> --base master --head <branch> \
    --title "<subject>" --body-file "$body_file"
```

The draft is the developer's review of the fix and its analysis, so it
stands in for the text approval that `create-pr` asks for: open it without
waiting, and let the developer review it there. Follow `create-pr` for the
title, the sentence style, and the guardrails.

The draft carries no issue reference, no `#NNN`, and no link to an upstream
issue or pull request, because GitHub would render each one as a backlink
upstream. A link to the failed Actions run creates no backlink, so always
include it. The body holds the full analysis for the developer, in these
sections, one line per paragraph:

- **Failure**: the URL of the failed run, the job, step, and test, and what
  the stack shows. Quote the error from the log verbatim in a fenced block:
  the error line and the stack frames or the few lines that locate it, with
  the log's timestamp prefix removed.
- **Analysis**: the classification from section 3 with its evidence, the
  mechanism, and what was ruled out.
- **Reproduction**: each mode and its count, including zero-failure results.
- **Change**: what the diff does and the new test.
- **Verification**: commands run and their actual results.
- **CI**: the decision from section 7 and its reason.

## 7. Decide on CI for the draft

This decides whether the fork runs the matrix on the draft before the
developer submits upstream. Run it when CI is the only place the fix can be
verified:

- The diff touches `.github/`, `contrib/ci/`, CMake, the Makefile, or
  dependency pins.
- The failure is deterministic but reproduces only on a platform you cannot
  run locally (Windows, macOS cocoa, CUDA).
- The fix changes platform-specific code.

Add the `skip-ci` label (`gh pr create --label skip-ci`; see
`.github/workflows/check_skip_ci.yml` for what it skips) when a run adds no
evidence:

- The failure reproduced locally in the failing lane, and the fix made it
  pass there.
- The failure is a rare flake. One green run proves nothing when the unfixed
  commit also passed. The master runs after the upstream merge are the real
  evidence; say so in the draft.

## 8. Hand off to upstream

The developer reviews the draft and decides whether the fix goes upstream.
Prepare the upstream PR only when asked, and only after the developer
approves its text as `create-pr` requires:

- Open it against the upstream `master` from the same fork branch, so the
  reviewed commits go up unchanged.
- Add the issue reference that the draft left out, as "Related to #NNN"; a
  failed master run often has one that `open_issue_on_failure` opened.
  Never use a closing keyword.
- Shorten the body to the form `create-pr` asks for: what the change does,
  why, and what was tested. Keep the run URL, and the quoted error when it is
  short. The full reproduction record stays in the draft.
- The upstream PR runs CI by default. Decide on `skip-ci` there separately,
  with the developer.

Leave closing the draft to the developer.

## Output

- Run, job, step, and the failing test or command.
- Classification with its evidence.
- Reproduction counts per mode.
- `opened: <draft PR URL> (draft on <fork>)` and the CI decision with its
  reason.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
