# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Test lane checker for solvcon.

A test that needs a real top-level window surface belongs under tests/gui,
and everything else belongs outside it. See tests/gui/README.md for the rule
and the reasoning.

Both directions are checked. The second one, a tests/gui file that has
stopped being a window test, is the one that rots without help, because a
file keeps its place long after its contents have moved on.
"""

import ast
import pathlib
import sys

GUARD = 'NO_LIVE_WINDOW'
TESTS_DIR = pathlib.Path('tests')
GUI_DIR = TESTS_DIR / 'gui'


def needs_live_window(path):
    text = path.read_text(encoding='utf-8')
    # The token appears in prose too, so confirm it is really referenced
    # before believing it. Parsing every file instead would cost more than
    # the rest of this script put together.
    if GUARD not in text:
        return False
    tree = ast.parse(text, filename=str(path))
    return any(isinstance(node, ast.Name) and GUARD == node.id
               for node in ast.walk(tree))


def main():
    if not TESTS_DIR.is_dir():
        print(f'FAILED: {TESTS_DIR} not found, run from the repository root',
              file=sys.stderr)
        return 1

    gui, plain = [], []
    for path in sorted(TESTS_DIR.rglob('test_*.py')):
        (gui if GUI_DIR in path.parents else plain).append(path)

    print(f'Checking {len(gui) + len(plain)} test files for lane placement')

    failed = [f'{p}: window test outside {GUI_DIR}'
              for p in plain if needs_live_window(p)]
    failed += [f'{p}: no window test left'
               for p in gui if not needs_live_window(p)]

    if failed:
        print(f'\nFAILED: {len(failed)} test files are in the wrong lane, '
              f'see {GUI_DIR}/README.md:')
        for line in failed:
            print(f'  - {line}')
        print(f'\nChecked {len(gui) + len(plain)} files total.')
        return 1

    print(f'SUCCESS: All {len(gui) + len(plain)} test files are in the '
          f'right lane.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
