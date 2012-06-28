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

AddOption('--count', dest='count',
    action='store_true', default=False,
    help='Count line of sources.')

class LineCounter(object):
    """
    Walk given directory to count lines in source files.
    """

    def __init__(self, *args, **kw):
        self.exts = args
        self.counter = dict()
        self.testdir = kw.pop('testdir', ['tests'])
        self.testcounter = 0
        self.corecounter = 0

    def __call__(self, path):
        import os
        from os.path import join, splitext
        for root, dirs, files in os.walk(path):
            for fname in files:
                mainfn, extfn = splitext(fname)
                if extfn not in self.exts:
                    continue
                if os.path.islink(join(root, fname)):
                    continue
                nline = len(open(join(root, fname)).readlines())
                self.counter[extfn] = self.counter.get(extfn, 0) + nline
                if os.path.basename(root) in self.testdir:
                    self.testcounter += nline
                else:
                    if extfn == '.py' and os.path.basename(root) == 'solvcon':
                        self.corecounter += nline

    def __str__(self):
        keylenmax = max([len(key) for key in self.counter])
        tmpl = "%%-%ds = %%d" % keylenmax
        all = 0
        ret = list()
        for extfn in sorted(self.counter.keys()):
            ret.append(tmpl % (extfn, self.counter[extfn]))
            all += self.counter[extfn]
        ret.append(tmpl % ('All', all))
        ret.append('%d are for unittest.' % self.testcounter)
        ret.append('%d are for core (only .py directly in solvcon/).' % \
            self.corecounter)
        return '\n'.join(ret)

if GetOption('count'):
    counter = LineCounter('.py', '.c', '.h', '.cu')
    paths = ('solvcon', 'src', 'include', 'test')
    for path in paths:
        counter(path)
    sys.stdout.write('In directories %s:\n' % ', '.join(paths))
    sys.stdout.write(str(counter)+'\n')
    sys.exit(0)

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
