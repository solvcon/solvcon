# Application

An application owns everything specific to solving one problem set. It can be
headless, GUI-heavy, or both, and it can use anything in solvcon. It is not a
library: the pilot may launch it, but no other code builds on it, and code
that others need is factored out of it. Applications live in the `solvcon.apps` package.

```{toctree}
:maxdepth: 2

euler
obsrefl
ising
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
