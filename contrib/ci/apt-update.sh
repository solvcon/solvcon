#!/bin/bash
#
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING
#
# Run `apt-get update` under a wall-clock cap and retry it. apt applies its
# acquire timeout to one read, not to the whole transfer, so a mirror that
# stays connected and trickles data never trips it, and the update runs until
# the CI job hits its time limit. A retry opens fresh connections, which a
# round-robin archive host can answer from a healthy backend. Driven by
# .github/actions/setup_linux/action.yml, and runnable by hand:
#
#   sudo contrib/ci/apt-update.sh -qq
#
# Every argument goes to `apt-get` ahead of the `update` sub-command. TIMEOUT
# is the cap on one attempt in seconds, and ATTEMPTS is how many attempts to
# make before the script gives up.

set -euo pipefail

TIMEOUT="${TIMEOUT:-120}"
ATTEMPTS="${ATTEMPTS:-3}"

for ((attempt = 1; attempt <= ATTEMPTS; attempt++)) ; do
    if [ "$attempt" -gt 1 ] ; then
        sleep $((30 * (attempt - 1)))
    fi
    # Fail the attempt on an index that did not download, e.g. a PPA 503.
    if timeout --kill-after=10 "$TIMEOUT" \
            apt-get -o APT::Update::Error-Mode=any "$@" update ; then
        exit 0
    fi
    echo "apt-get update attempt $attempt of $ATTEMPTS failed or timed out"
done

exit 1

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
