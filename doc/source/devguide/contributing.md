# Contributing

Contributions are made through [pull
requests](https://github.com/solvcon/solvcon/pulls). Build and test locally,
keep the source clean, and follow the project conventions below.

## Before You Commit

Run `make lint` to check the source.  It bundles seven checks: `cformat` (C++
formatting), `cinclude` (`#include` ordering), `flake8` (Python style and the
79-character line limit), `checkascii` (ASCII-only bytes), `checktws` (no
trailing whitespace), `checktests` (a window test lives under `tests/gui/`),
and `checkapps` (nothing outside an application imports it).  All sources are
UTF-8, Unix LF, ASCII-only, and carry a modeline at the end of file.  The
{doc}`style` page is the canonical reference.

Add tests for new behavior.  Python tests in `tests/` are preferred; add a C++
gtest only when the behaviour cannot or should not be exercised from Python.
See {doc}`testing` for how to run them.

## Pull Requests

Reference the related issue without closing keywords -- write "Related to #725"
rather than "fixes #725", so that issue management is not driven by commit or PR
text.  Keep the subject concise and describe the change clearly.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
