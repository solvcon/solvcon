---
name: serve-docs
description: Build the solvcon Sphinx docs and serve them on specified network with guaranteed teardown when the Claude session ends. Use when the user asks to build and serve the documentation, preview the docs, or share the rendered docs online.
---

# Serve Docs (solvcon)

Build `doc/build/html` and serve it on the network interface specified by the
user so they can open it from another machine, usually on a trusted
network, with a watchdog that shuts the server down within a few seconds of the
Claude session ending. CI builds the docs in the nightly run, and on a pull
request only when the `SCGH_FORCE_DOC` variable is `enable`, so always build
before serving.

## When to use

The user wants to see the rendered docs, preview a doc change, or reach the
docs from another machine on the trusted network. This skill owns the
build-serve-teardown flow; pair it with `commit-code` and `create-pr` when the
change is also being landed.

## 1. Build locally

Doxygen feeds the C++ API page through breathe, so run it before Sphinx;
without it the C++ API page renders empty (a warning, not an error). From
`doc/`, run the Doxygen target (`make doxygen`, which writes the C++ API XML
to `build/doxygen/xml` and needs a system doxygen), then the HTML target
(`make html`, which writes the site to `build/html/index.html`).

Treat a non-empty `make html` warning list as a defect to fix. The
pre-existing LaTeX `siunitx`/EPS image warnings from the CESE math pages
are environment noise and can be ignored; a warning naming a file you
edited cannot. For a faithful rebuild after only comment or XML changes,
prefer `make clean html` (or `make doxygen` again): Sphinx does not track
the Doxygen XML as a dependency and an incremental build can serve stale
content.

Whenever you rebuild during a session, report the document address to the
user again (section 2): the server keeps serving the same URL, now with the
refreshed content.

## 2. Serve on the specified network, with teardown tied to the session

`doc/contrib/serve-docs.sh` does the serving. Derive the trusted-network
address the user named at run time (never hardcode it, never `0.0.0.0`) and
pass it as the first argument; an optional second argument pins the port,
otherwise the script takes the first free port in 8765-8769.

The script ties the server's lifetime to a launcher PID and tears the server
down when that process exits. Its default launcher is its own parent, which
is useless here because each command runs in a throwaway shell, so pass the
real session with `--launcher-pid`: the launcher is the nearest ancestor
process named `claude`, found by climbing `PPid` in `/proc/<pid>/status` from
the shell's parent. Invoke it as
`doc/contrib/serve-docs.sh <bind-ip> --launcher-pid <pid>` (add the port as a
second positional argument to pin it).

The script binds `python3 -m http.server` to that address, starts a watchdog
that kills the server within about half a minute of the launcher exiting,
probes the URL with curl, and prints it. Report that
`http://<trusted-network-ip>:<port>/` URL to the user.

Report the URL again after every rebuild: the server serves straight from
`build/html`, so a fresh `make html` updates the live site at the same
address without a restart, no need to rerun the script. If the address has a
DNS record, use it.

## 3. Teardown

The watchdog guarantees teardown: the server is killed within about half a
minute of the session process exiting, so nothing is left listening after
the session ends. To stop it earlier in the same session, kill the server
PID the script printed (or `pkill -f "http.server <port>"`). A `SessionEnd`
hook is a weaker backstop only, because a mid-session settings edit may not
reload; the watchdog is the primary mechanism.

## Gotchas

- **Keep the serving logic in the script.** `doc/contrib/serve-docs.sh` is
  reviewable, checked-in code; do not inline a copy of it back into this
  skill. Read it before trusting it.
- **Do not switch the script to `setsid`.** With job control setsid forks, so
  `$!` is the dead parent, not the python PID, which breaks both the watchdog
  and any explicit kill. The script uses `nohup ... &` then `disown` on
  purpose.
- **Pass `--launcher-pid`.** The script's default launcher is its parent
  shell, a throwaway per-command shell here; without the real session PID the
  watchdog would kill the server almost immediately. Pass the nearest
  `claude` ancestor's PID.
- **Bind to the specified network IP, not an all-interfaces address.**
  Binding to all interfaces exposes the server on the local LAN.
  `http.server` is unauthenticated, so anyone who can reach the bound address
  can read everything under `build/html`; keep it on the trusted network. The
  script refuses all-interfaces binds (`0.0.0.0`, `::`) and validates the
  port and launcher PID, but it cannot tell a trusted address from a public
  one, so deriving the right address is still the caller's job.
- **Port collisions.** Sibling worktrees serving their own docs may hold 8000
  or 8765; the script skips busy ports.
- **No doc CI.** The local `make html` is the only doc validation, so a clean
  build matters. A docs-only branch pushed to `pg1` auto-skips the heavy CI
  matrix.

<!-- vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79: -->
