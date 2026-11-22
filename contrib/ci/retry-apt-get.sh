#!/bin/bash
#
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING
#
# Run `apt-get update` or `apt-get install` with its network transfer under a
# wall-clock cap, and retry it. apt applies its acquire timeout to one read,
# not to the whole transfer, so a mirror that stays connected and trickles
# data never trips it, and the command runs until the CI job hits its time
# limit. A retry opens fresh connections, which a round-robin archive host can
# answer from a healthy backend. Driven by .github/actions/setup_linux and
# .github/workflows/doc.yml, and runnable by hand:
#
#   sudo contrib/ci/retry-apt-get.sh -qq update
#   sudo contrib/ci/retry-apt-get.sh -qy install doxygen
#
# Every argument goes to `apt-get` verbatim. TIMEOUT is the cap on one attempt
# in seconds, and ATTEMPTS is how many attempts to make before the script
# gives up.

set -euo pipefail

TIMEOUT="${TIMEOUT:-120}"
ATTEMPTS="${ATTEMPTS:-3}"

retry() {
    local attempt
    for ((attempt = 1; attempt <= ATTEMPTS; attempt++)) ; do
        if [ "$attempt" -gt 1 ] ; then
            sleep $((30 * (attempt - 1)))
        fi
        if timeout --kill-after=10 "$TIMEOUT" apt-get "$@" ; then
            return 0
        fi
        echo "apt-get attempt $attempt of $ATTEMPTS failed or timed out"
    done
    return 1
}

for arg in "$@" ; do
    case "$arg" in
    # A separate option value ahead of the sub-command could read as one,
    # e.g. the config file in `-c update install`.
    -c|-o|-t|-a|--config-file|--option|--target-release|--host-architecture)
        echo "$0: attach the value to $arg, e.g. -oKey=value" >&2
        exit 2
        ;;
    update)
        # Fail the attempt on an index that did not download, e.g. a PPA 503.
        retry -o APT::Update::Error-Mode=any "$@"
        exit 0
        ;;
    install)
        # Cap only the download. Killing dpkg mid-unpack would break the
        # package database. A retry resumes from the files that an earlier
        # attempt left in the archive cache, and the install reads them there.
        # apt applies options in order, so the last one cannot be overridden.
        retry "$@" --download-only
        exec apt-get "$@"
        ;;
    esac
done

echo "$0: only the update and install sub-commands are supported" >&2
exit 2

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
