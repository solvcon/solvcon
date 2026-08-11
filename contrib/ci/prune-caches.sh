#!/bin/bash
#
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING
#
# Delete the caches that GitHub keeps but no run can ever restore: the
# generations a newer save superseded, and everything saved on a ref that is
# gone. Driven by .github/workflows/cache_cleanup.yml, and runnable by hand
# against any repository the local `gh` is authenticated for:
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

# Fail the sweep when ccache-action stopped timestamping the keys it saves,
# which would leave every generation a lineage of one, superseding nothing,
# so the run would delete nothing and still report success. Field 1 is the
# key with the timestamp stripped and field 6 the raw key, so equal fields
# mark such a key. awk exits zero when it printed one, hence the branch below.
if awk -F'\t' '
    $6 ~ /^s?ccache-/ && $1 == $6 {
      print "::error::" $6 " carries no timestamp suffix; the key format changed"
      found = 1
    }
    END { exit !found }
  ' "$workdir/caches.tsv"; then
  exit 1
fi

# `gh` says HTTP 404 for a branch that is gone and for a repository it cannot
# read, and the second answers 404 for every branch alike. Fail early on a
# repository that cannot be reached at all, so the common misconfiguration
# stops here with something a reader can act on.
if ! default_branch="$(gh api "repos/$GITHUB_REPOSITORY" --jq '.default_branch' \
    2>"$workdir/probe.err")"; then
  echo "::error::cannot read repository $GITHUB_REPOSITORY"
  cat "$workdir/probe.err" >&2
  exit 1
fi

# Checking that once would only cover the state before the loop, and access
# can be lost while it runs: a repository made private, renamed, or a token
# rotated away turns every later probe into a 404 and would condemn every
# branch in one pass. So a 404 is read as death only while the default
# branch, which cannot be missing, still answers for itself.
probe_branch() {
  if gh api "repos/$GITHUB_REPOSITORY/branches/$1" \
      >/dev/null 2>"$workdir/probe.err"; then
    echo live
  elif ! grep -q 'HTTP 404' "$workdir/probe.err"; then
    echo unknown
  elif gh api "repos/$GITHUB_REPOSITORY/branches/$default_branch" >/dev/null 2>&1; then
    echo dead
  else
    echo unknown
  fi
}

# Death is established per ref rather than inferred from a listing. A
# paginated listing that comes back short, whether truncated or reshuffled by
# a concurrent push, drops a live ref, and an absence read as death takes
# every cache on it at once. Asking about one ref answers about that ref, and
# a ref the probe cannot settle stays live: it is recorded and reported, and
# the sweep goes on to the generations that need no probe at all.
: > "$workdir/dead-refs.txt"
probed=0
unjudged=0
while read -r ref; do
  case "$ref" in
    refs/heads/*)
      branch="${ref#refs/heads/}"
      case "$branch" in
        *['#%']*)
          # A branch may legally carry these, and they do not survive the trip
          # through the URL path: the fragment is cut and a percent escape is
          # decoded, so the answer would be about some other name. Nothing is
          # going to change that, so warn and leave the caches where they are
          # rather than fail a daily job over it for good.
          echo "::warning::cannot probe $ref, its name does not survive a URL path"
          continue
          ;;
      esac
      probed=$((probed + 1))
      case "$(probe_branch "$branch")" in
        dead)
          echo "$ref" >> "$workdir/dead-refs.txt"
          ;;
        unknown)
          echo "::warning::cannot determine whether $ref still exists"
          cat "$workdir/probe.err" >&2
          unjudged=$((unjudged + 1))
          ;;
      esac
      ;;
    refs/pull/*/merge)
      number="${ref#refs/pull/}"
      number="${number%/merge}"
      # A closed pull request is the only death here. GitHub does not delete
      # pull requests, and this ref is proof number existed, so a 404 says the
      # probe cannot see them, which a token without pull-requests read also
      # answers. Reading that as death would sweep every merge ref at once.
      probed=$((probed + 1))
      if state="$(gh api "repos/$GITHUB_REPOSITORY/pulls/$number" --jq '.state' \
          2>"$workdir/probe.err")"; then
        if [ "$state" = 'closed' ]; then
          echo "$ref" >> "$workdir/dead-refs.txt"
        fi
      else
        echo "::warning::cannot determine the state of pull request $number"
        cat "$workdir/probe.err" >&2
        unjudged=$((unjudged + 1))
      fi
      ;;
  esac
done < <(cut -f2 "$workdir/caches.tsv" | sort -u)

# A cache on a deleted branch or a closed pull request can never be restored,
# whatever its generation, so those go whole rather than by the keep rule
# below. A ref the loop above did not judge, a tag say, stays live: it is not
# something this script knows how to call dead.
: > "$workdir/dead-entries.tsv"
: > "$workdir/live-entries.tsv"
# Keyed on FILENAME rather than NR == FNR, which reads the second file as the
# first when no ref turned out to be dead and would classify every entry as a
# ref name, sweeping nothing at all.
awk -F'\t' -v refs="$workdir/dead-refs.txt" \
    -v dead="$workdir/dead-entries.tsv" -v live="$workdir/live-entries.tsv" '
    FILENAME == refs { gone[$0] = 1; next }
    $2 in gone { print > dead; next }
    { print > live }
  ' "$workdir/dead-refs.txt" "$workdir/caches.tsv"

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
{
  awk -F'\t' '{ print "dead ref\t" $0 }' "$workdir/dead-entries.tsv"
  sort -t"$(printf '\t')" -k1,1 -k2,2 -k3,3r "$workdir/live-entries.tsv" \
    | awk -F'\t' -v keep="$KEEP" '
        { group = $1 SUBSEP $2 }
        { if (group == prev) { n += 1 } else { n = 1; prev = group } }
        n > keep { print "superseded\t" $0 }
      '
} > "$workdir/stale.tsv"

superseded=0
dead=0
failed=0
total=0
while IFS="$(printf '\t')" read -r reason lineage ref created id size key; do
  if [ "$DRY_RUN" = 'true' ]; then
    echo "would delete $key ($ref, $reason, $((size / 1048576)) MB)"
  elif gh api -X DELETE "repos/$GITHUB_REPOSITORY/actions/caches/$id" >/dev/null; then
    echo "deleted $key ($ref, $reason, $((size / 1048576)) MB)"
  else
    echo "::warning::failed to delete $key ($ref, $reason)"
    failed=$((failed + 1))
    continue
  fi
  if [ "$reason" = 'dead ref' ]; then
    dead=$((dead + 1))
  else
    superseded=$((superseded + 1))
  fi
  total=$((total + size))
done < "$workdir/stale.tsv"

counts="$superseded superseded, $dead on dead refs, $((total / 1048576)) MB"
if [ "$DRY_RUN" = 'true' ]; then
  summary_line="dry run: $counts"
else
  summary_line="deleted $counts reclaimed"
fi
echo "$summary_line" >> "$summary"

if [ "$failed" -gt 0 ]; then
  echo "::error::$failed cache deletions failed"
fi
# One ref going unjudged costs a little space and has already been warned
# about per ref. Every one of them going unjudged is a different thing, a
# repository the run could not read at all, and that is worth waking someone
# for. Failing on the first blip instead would leave the daily job red often
# enough that nobody reads it.
if [ "$unjudged" -gt 0 ] && [ "$unjudged" -eq "$probed" ]; then
  echo "::error::none of the $probed refs could be judged; nothing was swept on liveness"
  exit 1
fi
if [ "$failed" -gt 0 ]; then
  exit 1
fi

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
