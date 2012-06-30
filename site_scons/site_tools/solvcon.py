"""
SOLVCON Tool for SCons
"""

import sys
import os

def has_sse4(env):
    if not sys.platform.startswith('linux'):
        return False
    entries = [line.split(':') for line in
        open('/proc/cpuinfo').read().strip().split('\n') if len(line) > 0]
    cpuinfo = dict([(entry[0].strip(), entry[1].strip()) for entry in entries])
    if 'sse4' in cpuinfo['flags']:
        return True
    return False

def get_scdata(env, url, datapath):
    if os.path.exists(datapath):
        orig = os.getcwd()
        os.chdir(datapath)
        os.system('hg pull -u')
        os.chdir(orig)
    else:
        os.system('hg clone %s %s' % (url, datapath))

LIBPREFIX = 'sc'
LIBDIR = 'lib'
BUILDDIR = 'build'

def solvcon_shared(env, sdirs, libname, ndim=None, ext=None, fptype=None,
        srcdir='src', prepends=None):
    # clone the environment to avoid polution.
    env = env.Clone()
    # prepend custom environment variables.
    prepends = {} if prepends is None else prepends
    for key in prepends:
        env.Prepend(key=prepends[key])
    # prepare file lists.
    ddsts = list()
    for dsrc in sdirs:
        # craft source directory name.
        dsrc = '%s/%s' % (srcdir, str(dsrc))
        # skip non-directory.
        if not os.path.isdir(dsrc):
            continue
        # craft destination directory name.
        ddst = '%s/%s' % (BUILDDIR, os.path.basename(dsrc))
        if ndim is not None:
            ddst += '%dd' % ndim
        if ext is not None:
            ddst += '_%s' % ext
        if fptype is not None:
            ddst += '_%s' % {'float': 's', 'double': 'd'}[fptype]
        # copy source.
        env.VariantDir(ddst, dsrc, duplicate=1)
        # collect source files.
        ddsts.extend(env.Glob('%s/*.%s' % (ddst, 'c' if ext is None else ext)))
    ddsts = env.Flatten(ddsts)
    # craft library file name.
    filename = '%s/%s_%s' % (LIBDIR, LIBPREFIX, libname)
    if ndim is not None:
        env.Prepend(CCFLAGS=['-DNDIM=%d'%ndim])
        env.Prepend(NVCCFLAGS=['-DNDIM=%d'%ndim])
        filename += '%dd' % ndim
    if ext is not None:
        filename += '_%s' % ext
    if fptype is not None:
        env.Prepend(CCFLAGS=['-DFPTYPE=%s'%fptype])
        env.Prepend(NVCCFLAGS=['-DFPTYPE=%s'%fptype])
        filename += '_%s' % {'float': 's', 'double': 'd'}[fptype]
    # make the library.
    return env.SharedLibrary(filename, ddsts)

def generate(env):
    env.AddMethod(has_sse4, 'HasSse4')
    env.AddMethod(get_scdata, 'GetScdata')
    env.AddMethod(solvcon_shared, 'SolvconShared')

def exists(env):
    return env.Detect('solvcon')

# vim: set ff=unix ft=python fenc=utf8 ai et sw=4 ts=4 tw=79:
