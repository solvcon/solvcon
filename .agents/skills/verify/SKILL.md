---
name: verify
description: Verify that a code change works through the real running application and observable acceptance criteria. Use when the user invokes $verify or asks for end-to-end confirmation beyond unit tests, lint, or type checks.
---

# Verify

Translate the request and diff into concrete observable acceptance criteria.
Read repository instructions and a matching `run-*` skill if one exists.

Build and launch the real application using the supported workflow. Exercise
the changed behavior through its actual user or integration surface and collect
evidence for every acceptance criterion. Include an important negative or
boundary case when it is safe and relevant.

Tests, lint, logs, and internal state may support the result, but they do not
replace direct observation of the running application. Do not claim success
when an environment limitation prevented the relevant behavior from running.

Stop processes and remove temporary state created for verification. Report each
criterion as passed, failed, or blocked, with the evidence and commands used.
Do not fix a failure unless the user also requested implementation.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
