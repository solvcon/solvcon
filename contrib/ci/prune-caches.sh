#!/bin/bash
#
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING
#
# Delete the cache generations that GitHub keeps but no run can ever restore.
# Driven by .github/workflows/cache_cleanup.yml, and runnable by hand against
# any repository the local `gh` is authenticated for:
#
#   GITHUB_REPOSITORY=solvcon/solvcon DRY_RUN=true contrib/ci/prune-caches.sh
#
# GITHUB_REPOSITORY selects the repository, KEEP is the number of generations
# to keep per lineage (default 1), and DRY_RUN=true lists what would go
# without deleting it.

set -euo pipefail

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
KEEP="${KEEP:-1}"
DRY_RUN="${DRY_RUN:-false}"

# The dispatch input is free text and lands in awk unchecked, where 0 deletes
# every cache in the repository and a non-number deletes none while still
# reporting success. Neither is worth discovering from the summary line.
case "$KEEP" in
  '' | *[!0-9]*)
    echo "::error::KEEP must be a positive integer, got '$KEEP'"
    exit 1
    ;;
esac
if [ "$KEEP" -lt 1 ]; then
  echo "::error::KEEP must be at least 1, got '$KEEP'"
  exit 1
fi

# Outside Actions there is no step summary to append to, so send the totals to
# stdout and keep the script usable from a terminal.
summary="${GITHUB_STEP_SUMMARY:-/dev/stdout}"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

# ccache-action stamps every save with a timestamp and a GitHub cache entry is
# immutable, so a lineage such as ccache-Linux-Release gains one entry per save
# and only the newest can ever be restored. The rest hold the repository over
# its 10 GB quota, where GitHub evicts by last use across every lineage at
# once, which is blind to what the next run needs: a Linux save can drop the
# macOS cache.
gh api --paginate "repos/$GITHUB_REPOSITORY/actions/caches?per_page=100" \
  --jq '.actions_caches[]
        | [ (.key | sub("-[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]+Z$"; "")),
            .ref,
            .created_at,
            (.id | tostring),
            (.size_in_bytes | tostring),
            .key ]
        | @tsv' > "$workdir/caches.tsv"

# A silent no-op and a clean sweep look the same from the log, so refuse to
# report success when the timestamp above stopped matching and every entry
# became a lineage of one.
if [ -s "$workdir/caches.tsv" ] \
    && ! awk -F'\t' '$1 != $6 { found = 1 } END { exit !found }' "$workdir/caches.tsv"; then
  echo "::error::no cache key carries the expected timestamp suffix; the key format changed"
  exit 1
fi

# A cache is scoped to the ref that saved it, and the same key lives
# independently on master, on a branch, and on a pull request merge ref. Group
# by ref as well, or the newest entry across all of them wins and a branch push
# deletes the master generation that every other run restores from. Given
#
#   ccache-Linux-Release   master    10:00
#   ccache-Linux-Release   master    11:00
#   ccache-Linux-Release   feature   12:00
#
# grouping by lineage and ref drops only the superseded master 10:00. Grouping
# by lineage alone ranks all three together, so feature 12:00 takes first place
# and master 11:00 is dropped as merely second newest, leaving every run that
# reads master cold.
sort -t"$(printf '\t')" -k1,1 -k2,2 -k3,3r "$workdir/caches.tsv" \
  | awk -F'\t' -v keep="$KEEP" '
      { group = $1 SUBSEP $2 }
      { if (group == prev) { n += 1 } else { n = 1; prev = group } }
      n > keep { print }
    ' > "$workdir/stale.tsv"

deleted=0
failed=0
total=0
while IFS="$(printf '\t')" read -r lineage ref created id size key; do
  if [ "$DRY_RUN" = 'true' ]; then
    echo "would delete $key ($ref, $((size / 1048576)) MB)"
  elif gh api -X DELETE "repos/$GITHUB_REPOSITORY/actions/caches/$id" >/dev/null; then
    echo "deleted $key ($ref, $((size / 1048576)) MB)"
  else
    echo "::warning::failed to delete $key ($ref)"
    failed=$((failed + 1))
    continue
  fi
  deleted=$((deleted + 1))
  total=$((total + size))
done < "$workdir/stale.tsv"

if [ "$DRY_RUN" = 'true' ]; then
  summary_line="dry run: $deleted superseded generations, $((total / 1048576)) MB"
else
  summary_line="$deleted superseded generations deleted, $((total / 1048576)) MB reclaimed"
fi
echo "$summary_line" >> "$summary"

if [ "$failed" -gt 0 ]; then
  echo "::error::$failed cache deletions failed"
  exit 1
fi

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
