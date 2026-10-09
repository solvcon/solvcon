#!/bin/bash
#
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING
#
# Repeat a pytest selection to catch a flaky failure, optionally under the
# conditions that make one more likely. Run it from the checkout (or worktree)
# to test, after `make buildext` (plain pytest) or `make pilot` (--pilot):
#
#   .claude/skills/ci-triage/stress.sh --runs 20 --xcb tests/test_x.py
#   .claude/skills/ci-triage/stress.sh --runs 5 --pilot --gc -k test_x
#
# Options:
#   --runs N   repeat N times (default 10)
#   --pilot    run inside the pilot binary (`pilot --mode=pytest`); the pilot
#              always appends tests/, so select with -k, not a file path
#   --xcb      use the xcb platform under xvfb-run, like the Linux CI lane;
#              the default is the offscreen platform
#   --gc       run the cyclic garbage collector on nearly every allocation
#   --poison   fill freed memory with a byte pattern (glibc malloc.perturb,
#              with PYTHONMALLOC=malloc) so a use-after-free faults at once
#
# Every run's log lands in $CI_TRIAGE_OUT (default: a new temporary
# directory). The script prints each failing run and a final tally.

set -u

here=$(cd "$(dirname "$0")" && pwd)
root=$(git rev-parse --show-toplevel) || exit 1
runs=10
pilot=0
xcb=0
gc=0
poison=0
while [ $# -gt 0 ]; do
    case $1 in
        --runs) runs=$2; shift 2 ;;
        --pilot) pilot=1; shift ;;
        --xcb) xcb=1; shift ;;
        --gc) gc=1; shift ;;
        --poison) poison=1; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done

out=${CI_TRIAGE_OUT:-$(mktemp -d)}
mkdir -p "$out"
cd "$root" || exit 1

export PYTHONPATH=$root${PYTHONPATH:+:$PYTHONPATH}
opts="-q -x -p no:cacheprovider"
if [ $gc -eq 1 ]; then
    export PYTHONPATH=$PYTHONPATH:$here
    opts="$opts -p gcstress"
fi
if [ $poison -eq 1 ]; then
    export GLIBC_TUNABLES=glibc.malloc.perturb=165 PYTHONMALLOC=malloc
fi
if [ $xcb -eq 1 ]; then
    export QT_QPA_PLATFORM=xcb
    wrap="xvfb-run -a"
else
    export QT_QPA_PLATFORM=offscreen
    wrap=""
fi

if [ $pilot -eq 1 ]; then
    binary=$(ls -t "$root"/build/*/cpp/binary/pilot/pilot 2>/dev/null \
        | head -n 1)
    if [ -z "$binary" ]; then
        echo "no pilot binary under build/; run make pilot first" >&2
        exit 1
    fi
    export PYTEST_OPTS="$opts $*"
    command="$binary --mode=pytest"
else
    command="${PYTHON:-python3} -X faulthandler -m pytest $opts $*"
fi

fails=0
for i in $(seq 1 "$runs"); do
    $wrap $command > "$out/run$i.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        fails=$((fails + 1))
        echo "run $i: exit $rc ($out/run$i.log)"
        grep -E 'Fatal Python|FAILED|File ".*/tests/' "$out/run$i.log" \
            | head -n 4
    fi
done
echo "failed $fails of $runs runs; last summary:"
tail -n 1 "$out/run$runs.log"
echo "logs: $out"

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
