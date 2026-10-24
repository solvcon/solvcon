#!/usr/bin/env python3
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
App boundary checker for solvcon.

An application owns everything specific to solving one problem set. See
doc/source/application/index.md for the definition. The built-in
applications are the sub-packages of solvcon/apps. Every .py file under
solvcon is parsed, its imports (relative ones resolved, function-local ones
included) are mapped to absolute module names, and these are enforced:

- A module inside an app may import itself and anything that is not another
  app.
- The collection package solvcon/apps/__init__.py may import its apps.
- Every other module must not import an app or anything below it.
- Importing the bare package solvcon.apps is allowed only in BARE_ALLOWED.

The tests are out of scope, because a test may import app internals.
"""

import argparse
import ast
import pathlib
import sys

PACKAGE = 'solvcon'
APPS = f'{PACKAGE}.apps'

# The program entry point loads the apps so that they register with the
# pilot, which is the one place outside the apps that needs the bare package.
BARE_ALLOWED = ('solvcon/system.py',)


def find_apps(root):
    apps_dir = root / PACKAGE / 'apps'
    if not apps_dir.is_dir():
        return set()
    return {p.name for p in apps_dir.iterdir()
            if p.is_dir() and (p / '__init__.py').exists()}


def module_parts(path, root):
    parts = list(path.relative_to(root).with_suffix('').parts)
    is_pkg = parts[-1] == '__init__'
    if is_pkg:
        parts.pop()
    return parts, is_pkg


def classify(name, apps):
    """Return an app name, '' for the bare package, or None."""
    if name == APPS:
        return ''
    if name.startswith(APPS + '.'):
        head = name[len(APPS) + 1:].split('.')[0]
        return head if head in apps else ''
    return None


def imported_targets(node, parts, is_pkg, apps):
    """Yield (classification, absolute name) for an import node."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            target = classify(alias.name, apps)
            if target is not None:
                yield target, alias.name
        return

    base = []
    if node.level:
        pkg = parts if is_pkg else parts[:-1]
        keep = len(pkg) - (node.level - 1)
        if keep < 0:
            return
        base = pkg[:keep]
    if node.module:
        base = base + node.module.split('.')
    modname = '.'.join(base)

    target = classify(modname, apps)
    if target:
        yield target, modname
        return
    for alias in node.names:
        name = modname if alias.name == '*' else f'{modname}.{alias.name}'
        target = classify(name, apps)
        if target is not None:
            yield target, name


def check_file(path, root, apps):
    parts, is_pkg = module_parts(path, root)
    rel = path.relative_to(root).as_posix()
    in_app = len(parts) >= 3 and '.'.join(parts[:2]) == APPS
    owner = parts[2] if in_app else None
    is_collection = parts == APPS.split('.')

    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for target, name in imported_targets(node, parts, is_pkg, apps):
            if target == '':
                if in_app or is_collection or rel in BARE_ALLOWED:
                    continue
                found.append((node.lineno, f'imports the app package {name}',
                              'factor the shared code out of the apps, or '
                              'add the file to BARE_ALLOWED in '
                              'contrib/lint/check_app_imports.py if it must '
                              'load the apps'))
            elif target == owner or is_collection:
                continue
            else:
                found.append((node.lineno, f'imports app code {name}',
                              'move the shared code out of the app into a '
                              'common module'))
    return [(rel, *item) for item in found]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--root', type=pathlib.Path,
                        default=pathlib.Path('.'),
                        help='repository root (default: current directory)')
    root = parser.parse_args().root

    if not (root / PACKAGE).is_dir():
        print(f'FAILED: {root / PACKAGE} not found, run from the repository '
              f'root', file=sys.stderr)
        return 1

    apps = find_apps(root)
    files = sorted((root / PACKAGE).rglob('*.py'))
    print(f'Checking {len(files)} files for imports across the app boundary '
          f'(apps: {", ".join(sorted(apps)) or "none"})')

    violations = []
    for path in files:
        violations += check_file(path, root, apps)

    if violations:
        for rel, line, msg, hint in violations:
            print(f'{rel}:{line} -- {msg} -- {hint}')
        print(f'\nFAILED: {len(violations)} imports cross the app boundary.')
        return 1

    print(f'SUCCESS: No import crosses the app boundary in {len(files)} '
          f'files.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
