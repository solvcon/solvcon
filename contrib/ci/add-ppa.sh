#!/bin/bash
#
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING
#
# Add a PPA with `add-apt-repository` and retry it. add-apt-repository asks
# Launchpad for the PPA signing key over its REST API, and Launchpad answers
# with HTTP 500 GPGKeyTemporarilyNotFoundError now and then. A retry sends a
# new request, which Launchpad answers once the key is available again.
# Driven by .github/actions/setup_linux/action.yml, and runnable by hand:
#
#   sudo contrib/ci/add-ppa.sh ppa:ubuntu-toolchain-r/test
#
# Every argument goes to `add-apt-repository` after `-y -n`. ATTEMPTS is how
# many attempts to make before the script gives up.

set -euo pipefail

ATTEMPTS="${ATTEMPTS:-3}"

for ((attempt = 1; attempt <= ATTEMPTS; attempt++)) ; do
    if [ "$attempt" -gt 1 ] ; then
        sleep 10
    fi
    if add-apt-repository -y -n "$@" ; then
        exit 0
    fi
    echo "add-apt-repository attempt $attempt of $ATTEMPTS failed"
done

exit 1

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
