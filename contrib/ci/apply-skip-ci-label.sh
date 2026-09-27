#!/bin/bash
#
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING
#
# Carry the "skip-ci" label to the runs a pull request already has: cancel
# what is still running when the label arrives, and re-run what the label
# stopped when it goes away. Driven by .github/workflows/skip_ci_label.yml,
# and runnable by hand against any repository the local `gh` can reach:
#
#   GITHUB_REPOSITORY=solvcon/solvcon LABEL_NAME=skip-ci \
#     LABEL_ACTION=labeled HEAD_SHA=<sha> contrib/ci/apply-skip-ci-label.sh

set -euo pipefail

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${LABEL_NAME:?LABEL_NAME is required}"
: "${LABEL_ACTION:?LABEL_ACTION is required}"
: "${HEAD_SHA:?HEAD_SHA is required}"
: "${HEAD_REF:?HEAD_REF is required}"

# The workflows whose heavy jobs check_skip_ci gates.
WORKFLOWS="devbuild.yml devbuild_windows.yml lint.yml"

if [ "$LABEL_NAME" != "skip-ci" ]; then
  echo "::notice::The \"$LABEL_NAME\" label does not gate CI."
  exit 0
fi

# Pull request runs only. check_skip_ci reports a skip for a pull request and
# nothing for a push, so the label governs no push run. The branch narrows the
# commit further, because one commit can head more than one pull request. A
# run raised by a fork carries no pull request number to key on, so the branch
# is as close as the listing gets.
runs() {
  gh api --paginate \
    "repos/$GITHUB_REPOSITORY/actions/workflows/$1/runs?event=pull_request&head_sha=$HEAD_SHA&branch=$HEAD_REF&per_page=100" \
    --jq "$2"
}

for workflow in $WORKFLOWS; do
  if [ "$LABEL_ACTION" = "labeled" ]; then
    for id in $(runs "$workflow" \
        '.workflow_runs[] | select(.status != "completed") | .id'); do
      if error="$(gh api -X POST \
          "repos/$GITHUB_REPOSITORY/actions/runs/$id/cancel" 2>&1 >/dev/null)"
      then
        echo "::notice::Cancelled $workflow run $id."
        continue
      fi
      # A run that ended between the listing and the request answers 409,
      # which is the label arriving a moment too late, not a failure. Any
      # other answer leaves the run going and must not read as success.
      case "$error" in
        *"HTTP 409"*)
          echo "::notice::$workflow run $id was no longer running."
          ;;
        *)
          echo "::error::Could not cancel $workflow run $id: $error"
          exit 1
          ;;
      esac
    done
    continue
  fi

  # A cancelled run is one this script stopped, and it reported nothing.
  for id in $(runs "$workflow" \
      '.workflow_runs[] | select(.conclusion == "cancelled") | .id'); do
    gh api -X POST "repos/$GITHUB_REPOSITORY/actions/runs/$id/rerun" >/dev/null
    echo "::notice::Re-ran $workflow run $id."
  done

  for id in $(runs "$workflow" \
      '.workflow_runs[] | select(.status == "completed" and
                                 .conclusion != "cancelled") | .id'); do
    # A run that finished with the label on skipped every gated job. One that
    # built, failed, or timed out keeps the result it reported, so a broken
    # build does not spend the matrix again over a label. The gate itself
    # always succeeds and says nothing about whether the run did any work.
    reported="$(gh api --paginate \
      "repos/$GITHUB_REPOSITORY/actions/runs/$id/jobs?per_page=100" \
      --jq '[.jobs[]
             | select(.name | endswith("check_skip_ci") | not)
             | select(.conclusion != "skipped")] | length')"
    if [ "$reported" -eq 0 ]; then
      gh api -X POST "repos/$GITHUB_REPOSITORY/actions/runs/$id/rerun" \
        >/dev/null
      echo "::notice::Re-ran $workflow run $id."
    fi
  done
done

# vim: set ff=unix fenc=utf8 et sw=2 ts=2 sts=2:
