import os, sys
Import('env')

lpre = 'sc'
ldir = 'lib'
sdir = 'src'
bdir = 'build'
libs = list()

# lib_solvcon.
for fpmark, fptype in [('s', 'float'), ('d', 'double'),]:
    ddsts = list()
    for dsrc in ['block', 'domain', 'partition', 'solve']:
        dsrc = '%s/%s' % (sdir, dsrc)
        if not os.path.isdir(dsrc): continue
        ddst = '%s/%s_%s' % (bdir, os.path.basename(dsrc), fpmark)
        VariantDir(ddst, dsrc, duplicate=0)
        ddsts.append(ddst)
    envm = env.Clone()
    envm.Prepend(CCFLAGS=['-DFPTYPE=%s'%fptype])
    libs.append(envm.SharedLibrary(
        '%s/%s_solvcon_%s' % (ldir, lpre, fpmark),
        [Glob('%s/*.c'%ddst) for ddst in ddsts],
    ))

# lib_solvcontest.
VariantDir('%s/test' % bdir, 'test/src', duplicate=0)
libs.append(env.SharedLibrary('%s/%s_solvcontest' % (ldir, lpre),
    Glob('%s/test/*.c'%bdir)))
# lib_cuda13test.
if FindFile('nvcc', os.environ['PATH'].split(':')):
    envm = env.Clone()
    envm['NVCCFLAGS'] = ['-arch=sm_13', '-Xcompiler', '-fPIC']
    envm.Append(LIBS=['cudart'])
    VariantDir('%s/cuda13test' % bdir, 'test/cudasrc', duplicate=0)
    libs.append(envm.SharedLibrary('%s/%s_cuda13test' % (ldir, lpre),
        Glob('%s/cuda13test/*.cu'%bdir)))
# lib_cuda20test.
if FindFile('nvcc', os.environ['PATH'].split(':')):
    envm = env.Clone()
    envm['NVCCFLAGS'] = ['-arch=sm_20', '-Xcompiler', '-fPIC']
    envm.Append(LIBS=['cudart'])
    VariantDir('%s/cuda20test' % bdir, 'test/cudasrc', duplicate=0)
    libs.append(envm.SharedLibrary('%s/%s_cuda20test' % (ldir, lpre),
        Glob('%s/cuda20test/*.cu'%bdir)))

# kerpak libraries.
def make_kplib(lname, lpre, ldir, sdir, bdir, env, extra_links=None):
    """
    Make kerpak libraries.
    """
    libs = []
    for ndim in 2, 3:
        VariantDir('%s/%s%dd' % (bdir, lname, ndim), sdir, duplicate=0)
        envm = env.Clone()
        envm.Prepend(CCFLAGS=['-DNDIM=%d'%ndim])
        if extra_links is not None:
            envm.Prepend(LIBS=extra_links)
        libs.append(envm.SharedLibrary(
            '%s/%s_%s%dd' % (ldir, lpre, lname, ndim),
            Glob('%s/%s%dd/*.c' % (bdir, lname, ndim))))
    return libs
kplibs = [
    ('cese', None), ('ceseb', None),    # solvcon.kerpak.cese
    ('elaslin', None), ('elaslinb', None),  # solvcon.kerpak.elaslin
    ('euler', None), ('eulerb', None),  # solvcon.kerpak.euler
    ('lincese', ['lapack']),    # solvcon.kerpak.lincese
]
for lname, extra_links in kplibs:
    libs.extend(make_kplib(lname, lpre, ldir, '%s/%s'%(sdir, lname),
        bdir, env, extra_links=extra_links))

# TODO: OBSELETE
if GetOption('enable_f90'):
    # lib_solvcon in FORTRAN.
    for fpmark, fptype in [('s', 4), ('d', 8),]:
        ddsts = list()
        for dsrc in ['block', 'domain', 'partition', 'solve']:
            dsrc = '%sf/%s' % (sdir, dsrc)
            if not os.path.isdir(dsrc): continue
            ddst = '%s/%s_%s' % (bdir, os.path.basename(dsrc), fpmark)
            VariantDir(ddst, dsrc, duplicate=0)
            ddsts.append(ddst)
        envm = env.Clone()
        envm.Prepend(F90FLAGS=['-DFPKIND=%d'%fptype])
        libs.append(envm.SharedLibrary(
            '%s/%s_solvcon_%s' % (ldir, lpre, fpmark),
            [Glob('%s/*.f90'%ddst) for ddst in ddsts],
        ))
    # lib_solvcontest.
    lib_solvcontest = env.SharedLibrary('%s/%s_solvcontest' % (ldir, lpre),
        Glob('test/src/*.f90'))

# METIS.
src = 'dep/metis-4.0/Lib'
VariantDir('%s/metis' % bdir, src, duplicate=0)
ccflags = list()
if sys.platform.startswith('win'):
    ccflags.append('-D__VC__')
envm = env.Clone()
envm['CCFLAGS'] = ' '.join(ccflags)
envm['CPPPATH'] = '-I%s' % src
lib_metis = envm.SharedLibrary('%s/%s_metis' % (ldir, lpre),
    Glob('%s/metis/*.c' % bdir))

# documents.
epydoc = env.Epydoc('solvcon/__init__.py')
sphinx = env.Sphinx('doc/source/conf.py')

# name targets and set default.
solvcon = Alias('solvcon', libs)
metis = Alias('metis', [lib_metis])
all = Alias('all', [metis, solvcon])
Default(all)
Alias('epydoc', epydoc)
Alias('sphinx', sphinx)
doc = Alias('doc', [epydoc, sphinx])

# vim: set ft=python ff=unix:
