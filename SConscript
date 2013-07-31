"""
SConscript: The defined rules.
"""

import os
Import('targets', 'env')

# documents.
targets['scepydoc'] = env.BuildEpydoc('solvcon/__init__.py')
targets['scsphinx'] = env.BuildSphinx(
    Glob('doc/source/*.rst')+Glob('doc/source/*.py'))

# vim: set ff=unix ft=python fenc=utf8 ai et sw=4 ts=4 tw=79:
