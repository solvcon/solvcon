import os
import sys
from solvcon import __version__

# compilation.
AddOption('--disable-openmp', dest='use_openmp',
    action='store_false', default=True,
    help='Disable OpenMP.')
AddOption('--cc', dest='cc', type='string', action='store', default='gcc',
    help='C compiler (SCons tool): gcc, intelc.',)
AddOption('--optlevel', dest='optlevel', type=int, action='store', default=2,
    help='Optimization level; default is 2.',)
AddOption('--cmpvsn', action='store', default='', dest='cmpvsn',
    help='Compiler version; for gcc-4.5 it\'s --cmpvsn=-4.5',
)
AddOption('--sm', action='store', default='20', dest='sm',
    help='Compute capability; 13=1.3 and 20=2.0 are currently supported.',
)

AddOption('--get-scdata', dest='get_scdata',
    action='store_true', default=False,
    help='Flag to clone/pull example data.')

# solvcon environment.
env = Environment(ENV=os.environ)
# tools.
env.Tool('mingw' if sys.platform.startswith('win') else 'default')
env.Tool(GetOption('cc'))
env.Tool('solvcon')
env.Tool('sphinx')
env.Tool('scons_epydoc')
# Intel C runtime library.
if GetOption('cc') == 'intelc':
    env.Append(LIBS='irc_s')
# optimization level.
env.Append(CFLAGS='-O%d'%GetOption('optlevel'))
# SSE4.
if env.HasSse4() and GetOption('cc') == 'gcc':
    env.Append(CFLAGS='-msse4')
    env.Append(CFLAGS='-mfpmath=sse')
# OpenMP.
if GetOption('use_openmp'):
    if GetOption('cc') == 'gcc':
        env.Append(CFLAGS='-fopenmp')
        env.Append(LINKFLAGS='-fopenmp')
    elif GetOption('cc') == 'intelc':
        env.Append(CFLAGS='-openmp')
        env.Append(LINKFLAGS='-openmp')
# include paths.
env.Append(CPPPATH='include')
# CUDA.
env.Tool('cuda')
env.Append(NVCCFLAGS='-arch=sm_%s'%GetOption('sm'))
env.Append(NVCCINC=' -I include')

# replace gcc with a certain version.
if GetOption('cc') == 'gcc':
    env.Replace(CC='gcc%s'%GetOption('cmpvsn'))

# get example data.
if GetOption('get_scdata'):
    if __version__.endswith('+'):
        env.GetScdata('https://bitbucket.org/solvcon/scdata', 'scdata')
    else:
        raise RuntimeError('released tarball shouldn\'t use this option')

everything = []
Export('everything', 'env')

SConscript(['SConscript'])
Default(everything)

# vim: set ft=python ff=unix:
