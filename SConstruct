"""
SConstruct: The settings.
"""

import sys, os
import numpy as np
from solvcon import __version__

# compiler options.
AddOption('--debug-build', dest='debug_build', action='store_true',
    default=False, help='Make debugging build; '
    'this option will add ".dbg" to build directory')
AddOption('--disable-openmp', dest='openmp',
    action='store_false', default=True, help='Disable OpenMP.')
AddOption('--ctool', dest='ctool', type='string', action='store',
    default='gcc',
    help='SCons C compiler tool, e.g., gcc or intelc; default is "%default".')
AddOption('--optlevel', dest='optlevel', type=str, action='store', default='2',
    help='Optimization level; default is "%default".',)
AddOption('--sm', action='store', default='20', dest='sm',
    help='Compute capability for CUDA; '
    '13=1.3 and 20=2.0 are currently supported; default is "%default".')
# paths.
AddOption('--python-path', dest='pythonpath', type='string', action='store',
    default=os.path.join(sys.prefix, 'include',
        'python%d.%d'%tuple(sys.version_info[:2])),
    help='Python include path; default is %default.')
# generated files.
AddOption('--library-prefix', dest='libprefix', type='string', action='store',
    default='sc_', help='Prefix for compiled libraries; default is "%default".')
AddOption('--library-dir', dest='libdir', type='string', action='store',
    default='lib',
    help='Directory for compiled libraries; default is "%default".')
AddOption('--build-dir', dest='builddir', type='string', action='store',
    default='build', help='Build directory; default is "%default".')
# miscellaneous.
AddOption('--get-scdata', dest='get_scdata',
    action='store_true', default=False, help='Clone/pull example data.')
AddOption('--list-aliases', dest='list_aliases',
    action='store_true', default=False,
    help='List all target aliases and build nothing.')

# solvcon environment.
env = Environment(ENV=os.environ,
                  SCLIBPREFIX=GetOption('libprefix'),
                  SCLIBDIR=os.path.abspath(GetOption('libdir')),
                  SCBUILDDIR=os.path.abspath(GetOption('builddir')),
                  SCPROJDIR=os.path.abspath('.'))
# tools.
env.Tool('mingw' if sys.platform.startswith('win') else 'default')
env.Tool(GetOption('ctool'))
env.Tool('cython')
env.Tool('pyext')
env.Tool('solvcon')
env.Tool('sphinx')
env.Tool('scons_epydoc')
# allow using alternative command for CC.
env.Replace(CC=os.environ.get('CC', env['CC']))
# SCOTCH.
env.Append(LIBS=['scotchmetis', 'scotch', 'scotcherr', 'scotcherrexit'])
# Intel C runtime library.
if GetOption('ctool') == 'intelc':
    env.Append(LIBS='irc_s')
# debugging.
if GetOption('debug_build'):
    env['SCBUILDDIR'] += '.dbg'
    env.Append(CFLAGS='-g')
# optimization level.
env.Append(CFLAGS='-O%s'%GetOption('optlevel'))
# SSE4.
if env.HasSse4() and GetOption('ctool') == 'gcc':
    env.Append(CFLAGS='-msse4')
    env.Append(CFLAGS='-mfpmath=sse')
# OpenMP.
if GetOption('openmp'):
    if GetOption('ctool') == 'gcc':
        env.Append(CFLAGS='-fopenmp')
        env.Append(LINKFLAGS='-fopenmp')
    elif GetOption('ctool') == 'intelc':
        env.Append(CFLAGS='-openmp')
        env.Append(LINKFLAGS='-openmp')
# include paths.
env.Append(CPPPATH=map(os.path.abspath, ['include', 'solvcon']))
for path in [GetOption('pythonpath'), np.get_include()]:
    if path not in env['CPPPATH']:
        env.Append(CPPPATH=[path])
# library paths.
if 'SCROOT' in os.environ:
    env.Append(LIBPATH=[os.path.join(os.environ['SCROOT'], 'lib')])
env.Append(LIBPATH=[GetOption('libdir')])
# CUDA.
env.Tool('cuda')
env.Append(NVCCFLAGS='-arch=sm_%s'%GetOption('sm'))
env.Append(NVCCINC=' -I include -I solvcon')

# get example data.
if GetOption('get_scdata'):
    if __version__.endswith('+'):
        env.GetScdata('https://bitbucket.org/solvcon/scdata', 'scdata')
    else:
        raise RuntimeError('released tarball shouldn\'t use this option')

# invoke rules set in SConscript.
targets = {}
Export('targets', 'env')
SConscript(['SConscript', 'SConscript.legacy', 'solvcon/SConscript'])

# set alias and default targets.
for key in targets:
    Alias(key, targets[key])
Alias('scdocs', [targets['sc'+key] for key in 'epydoc', 'sphinx'])
Alias('sclibs', [targets['sc'+key] for key in 'main', 'test', 'kp', 'kpcu'])
Default(['sclibs', 'scmods'])

# show target aliases without doing anything else.
if GetOption('list_aliases'):
    for key in sorted(targets.keys()):
        sys.stdout.write('First-level targets: %s\n' % key)
    for key in 'scdocs', 'sclibs':
        sys.stdout.write('Second-level targets: %s\n' % key)
    sys.exit()

# vim: set ff=unix ft=python fenc=utf8 ai et sw=4 ts=4 tw=79:
