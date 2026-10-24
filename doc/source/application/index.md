# Application

An application is a Python package for solving a problem or a set of problems.
It can be headless, GUI-heavy, or both.  One application should not
depend on (access) another (it is not a library), but can use everything in
solvcon.

solvcon provides built-in applications in the `solvcon.apps` package.  Users
may create their own solvcon applications outside the `solvcon` namespace.

```{toctree}
:maxdepth: 2

euler
obsrefl
ising
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
