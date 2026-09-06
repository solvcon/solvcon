#!/usr/bin/env python3
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Install an IDE's CMakeUserPresets.json from the checked-in template.

``CMakeUserPresets.json`` is gitignored and holds one machine's literal
paths.  Its presets are named ``ide-*`` to keep them apart from the shared
presets in ``CMakePresets.json``, which is where a build belongs.

The scdv templates beside this script name the prefix once, as
``$penv{SCDV_USRDIR}`` in the preset's ``environment`` map, and the cache
variables read it back as ``$env{SCDV_USRDIR}``.  Installing is a copy with
that single occurrence replaced by a resolved prefix.  The substitution is
the point: an IDE started from the desktop inherits the session
environment, not the shell where an scdv was activated, so the unexpanded
form would configure the project against empty paths.

The ``example`` templates beside the scdv ones are for a prefix that is not
an scdv.  They are hand-edited and are not installed from here.

Usage:
    contrib/cmake/install-user-presets.py
    contrib/cmake/install-user-presets.py --prefix ~/var/scdv/main/usr
    contrib/cmake/install-user-presets.py --check
"""

import os
import sys
import json
import shutil
import difflib
import pathlib
import argparse
import subprocess

PLACEHOLDER = '$penv{SCDV_USRDIR}'
TARGET = 'CMakeUserPresets.json'
TEMPLATES = {
    'nt': 'CMakeUserPresets.win-scdv.json',
    'posix': 'CMakeUserPresets.scdv.json',
}
EXPECTED_PRESETS = (
    ('configure', ('ide-scdv-reldbg',)),
    ('build', ('ide-scdv-reldbg', 'ide-scdv-reldbg-module',
               'ide-scdv-reldbg-gtest')),
)


def repo_root():
    """Return the checkout holding this script.

    A worktree gets its own, not the main checkout that spawned it.
    """
    return pathlib.Path(__file__).resolve().parents[2]


def template_path(root):
    return root / 'contrib' / 'cmake' / TEMPLATES[os.name]


def resolve_prefix(explicit):
    """Resolve the scdv prefix that the template's placeholder names."""
    if explicit:
        prefix = pathlib.Path(explicit).expanduser()
    elif os.environ.get('SCDV_USRDIR'):
        prefix = pathlib.Path(os.environ['SCDV_USRDIR'])
    elif os.environ.get('SCDV_BASE'):
        prefix = pathlib.Path(os.environ['SCDV_BASE']) / 'usr'
    else:
        raise SystemExit(
            'error: no scdv prefix resolved. Activate an scdv environment, '
            'or pass --prefix naming the usr directory of one.')
    prefix = prefix.expanduser().resolve()
    if not prefix.is_dir():
        raise SystemExit('error: prefix %s is not a directory' % prefix)
    return prefix


def render(root, prefix):
    """Return the template text with the prefix substituted in."""
    path = template_path(root)
    text = path.read_text(encoding='utf-8')
    if PLACEHOLDER not in text:
        raise SystemExit('error: %s holds no %s to substitute'
                         % (path, PLACEHOLDER))
    # A Windows prefix goes in with forward slashes: a backslash would not
    # survive the JSON string, and CMake accepts either separator.
    return text.replace(PLACEHOLDER, prefix.as_posix())


def diff(installed, rendered):
    return ''.join(difflib.unified_diff(
        installed.splitlines(keepends=True),
        rendered.splitlines(keepends=True),
        fromfile='%s (installed)' % TARGET,
        tofile='%s (rendered)' % TARGET))


def verify(root, target):
    """Check that the installed file names a real prefix and lists presets."""
    presets = json.loads(target.read_text(encoding='utf-8'))
    named = presets['configurePresets'][0]['environment']['SCDV_USRDIR']
    if not os.path.isdir(named):
        print('error: SCDV_USRDIR "%s" is not a directory' % named,
              file=sys.stderr)
        return False
    if shutil.which('cmake') is None:
        print('note: cmake is not on PATH; presets were not listed')
        return True
    # An IDE sees the environment with no scdv activated, which is what the
    # substitution exists to survive, so list the presets without it.
    env = dict(os.environ)
    env.pop('SCDV_USRDIR', None)
    for kind, wanted in EXPECTED_PRESETS:
        proc = subprocess.run(['cmake', '--list-presets=' + kind],
                              cwd=str(root), env=env,
                              capture_output=True, text=True)
        missing = [name for name in wanted
                   if '"%s"' % name not in proc.stdout]
        if proc.returncode != 0 or missing:
            sys.stderr.write(proc.stdout + proc.stderr)
            print('error: cmake did not list %s preset %s'
                  % (kind, ', '.join(missing or wanted)), file=sys.stderr)
            return False
        print('%s presets: %s' % (kind, ', '.join(wanted)))
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Install CMakeUserPresets.json from the checked-in '
        'scdv template, substituting the resolved prefix.'
    )
    parser.add_argument(
        '--prefix',
        help='scdv usr directory to substitute '
        '(default: $SCDV_USRDIR, else $SCDV_BASE/usr)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='report whether the installed copy is current; write nothing'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='overwrite an installed copy that differs from the template'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    root = repo_root()
    target = root / TARGET
    prefix = resolve_prefix(args.prefix)
    rendered = render(root, prefix)
    installed = (target.read_text(encoding='utf-8')
                 if target.exists() else None)
    current = installed == rendered

    if args.check:
        if current:
            print('%s is current (prefix %s)' % (TARGET, prefix))
            return 0
        state = 'missing' if installed is None else 'differs'
        print('%s %s (prefix %s)' % (TARGET, state, prefix))
        if installed is not None:
            sys.stdout.write(diff(installed, rendered))
        return 1

    if current:
        print('%s is current (prefix %s)' % (TARGET, prefix))
    elif installed is not None and not args.force:
        sys.stdout.write(diff(installed, rendered))
        sys.stdout.flush()
        print('error: %s differs from the template. Review the diff, then '
              'rerun with --force to overwrite.' % TARGET, file=sys.stderr)
        return 1
    else:
        target.write_text(rendered, encoding='utf-8')
        print('wrote %s (prefix %s)' % (target, prefix))

    return 0 if verify(root, target) else 1


if __name__ == '__main__':
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
