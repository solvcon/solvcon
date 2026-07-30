# GUI tests

A test belongs here when it needs a real top-level window surface, which is
what `NO_LIVE_WINDOW` in each file guards against. That is a property of the
test rather than a judgement about it: the offscreen Qt platform and the
headless Windows CI runner cannot back a live window, so these tests skip
there, and they are the slow half of the suite because driving real windows
costs real time.

Everything else, including the pilot tests that build widgets without mapping
a window, stays in `tests/`.

Run one side or the other:

    make pytest-fast    # everything except this directory
    make pytest-gui     # only this directory
    make pytest         # both, and what CI runs on master and nightly

A pull request runs `pytest-fast` under python plus PySide6, because the same
tests run seconds later inside the pilot binary in the same job, where the
embedded interpreter is the environment nothing else exercises. A push and the
nightly schedule run the whole suite in both hosts.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4: -->
