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

def prepare_files(env, sdirs, libname, ndim=None, ext=None, fptype=None,
        srcdir='src', prepends=None, sclibprefix=None):
    """
    I need SCBUILDDIR, SCLIBDIR, and SCLIBPREFIX set in env.
    """
    # clone the environment to avoid polution.
    env = env.Clone()
    # prepend custom environment variables.
    prepends = {} if prepends is None else prepends
    for key in prepends:
        if prepends[key] is not None:
            env.Prepend(**{key: prepends[key]}) # weird treatment for Prepend.
    # determine library prefix.
    sclibprefix = env['SCLIBPREFIX'] if sclibprefix is None else sclibprefix
    # prepare file lists.
    ddsts = list()
    for dsrc in sdirs:
        # craft source directory name.
        dsrc = '%s/%s' % (srcdir, str(dsrc))
        # skip non-directory.
        if not os.path.isdir(dsrc):
            continue
        # craft destination directory name.
        ddst = '%s/%s' % (env['SCBUILDDIR'], os.path.basename(dsrc))
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
    filename = '%s/%s%s' % (env['SCLIBDIR'], sclibprefix, libname)
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
    return env, filename, ddsts

def solvcon_static(env, *args, **kw):
    if '-fPIC' not in env['CFLAGS']:
        env = env.Clone()
        env.Append(CFLAGS=['-fPIC'])
    env, filename, ddsts = prepare_files(env, *args, **kw)
    return env.StaticLibrary(filename, ddsts)

def solvcon_shared(env, *args, **kw):
    env, filename, ddsts = prepare_files(env, *args, **kw)
    return env.SharedLibrary(filename, ddsts)

def prepare_module_files(env, sdirs, libname, ndim=None, ext=None, fptype=None,
        srcdir='src', prepends=None, sclibprefix=None):
    """
    I need SCBUILDDIR, SCLIBDIR, and SCLIBPREFIX set in env.
    """
    # clone the environment to avoid polution.
    env = env.Clone()
    # prepend custom environment variables.
    prepends = {} if prepends is None else prepends
    for key in prepends:
        if prepends[key] is not None:
            env.Prepend(**{key: prepends[key]}) # weird treatment for Prepend.
    # determine library prefix.
    sclibprefix = env['SCLIBPREFIX'] if sclibprefix is None else sclibprefix
    # prepare file lists.
    ddsts = list()
    for dsrc in sdirs:
        # craft source directory name.
        dsrc = '%s/%s' % (srcdir, str(dsrc))
        # skip non-directory.
        if not os.path.isdir(dsrc):
            continue
        # craft destination directory name.
        ddst = '%s/%s' % (env['SCBUILDDIR'], os.path.basename(dsrc))
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
    filename = '%s/%s%s' % (env['SCLIBDIR'], sclibprefix, libname)
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
    return env, filename, ddsts

def solvcon_module(env, srcs, prepends=None, pkgroot='solvcon'):
    from SCons.Defaults import Copy
    env = env.Clone()
    stenv = env.Clone()
    dsts = list()
    # prepend custom environment variables.
    prepends = {} if prepends is None else prepends
    for key in prepends:
        if prepends[key] is not None:
            env.Prepend(**{key: prepends[key]}) # weird treatment for Prepend.
    # make sure of input list.
    if isinstance(srcs, basestring):
        srcs = [srcs]
    # cython C files.
    cyfiles = []
    for src in srcs:
        mainfn = os.path.splitext(os.path.basename(str(src)))[0]
        cyfiles.append(env.Cython(src))
    cyfiles = env.Flatten(cyfiles)
    dsts.extend(cyfiles)
    # static library.
    sdirs = [os.path.splitext(os.path.basename(str(src)))[0] for src in srcs]
    stenv, filename, ddsts = prepare_module_files(
        stenv, sdirs, 'solvcon', sclibprefix='')
    if '-fPIC' not in stenv['CFLAGS']:
        stenv.Append(CFLAGS=['-fPIC'])
    sclib = stenv.StaticLibrary(filename, ddsts)
    dsts.append(sclib)
    # make all sources.
    for cyfile in cyfiles:
        mainfn = os.path.splitext(os.path.basename(str(src)))[0]
        pymod = env.PythonExtension(cyfile)[0]
        env.Depends(pymod, sclib)
        dsts.append(pymod)
    return dsts

def generate(env):
    env.AddMethod(has_sse4, 'HasSse4')
    env.AddMethod(get_scdata, 'GetScdata')
    env.AddMethod(solvcon_static, 'SolvconStatic')
    env.AddMethod(solvcon_shared, 'SolvconShared')
    env.AddMethod(solvcon_module, 'SolvconModule')

def exists(env):
    return env.Detect('solvcon')

# vim: set ff=unix ft=python fenc=utf8 ai et sw=4 ts=4 tw=79:
