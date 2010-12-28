import os, sys
Import('env')

lpre = 'sc'
sdir = 'src'
bdir = 'build'
ldir = 'lib'
libs = list()

# lib_solvcon.
for fpmark, fptype in [('s', 'float'), ('d', 'double'),]:
    ddsts = list()
    for dsrc in ['block', 'domain', 'partition', 'solve']:
        dsrc = '%s/%sc' % (sdir, dsrc)
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

# kerpak libraries.
dimlibs = [ 
    'cese', 'ceseb',    # solvcon.kerpak.cese
    'euler', 'eulerb',  # solvcon.kerpak.euler
    'elasticb', # solvcon.kerpak.elastic
]
for ndim in 2, 3:
    for lkey in dimlibs:
        VariantDir('%s/%s%dd' % (bdir, lkey, ndim),
            '%s/%s' % (sdir, lkey), duplicate=0)
    envm = env.Clone()
    envm.Prepend(CCFLAGS=['-DNDIM=%d'%ndim])
    for lkey in dimlibs:
        libs.append(envm.SharedLibrary(
            '%s/%s_%s%dd' % (ldir, lpre, lkey, ndim),
            Glob('%s/%s%dd/*.c' % (bdir, lkey, ndim)),
        ))
# elastic solver needs lapack.
dimlibs = [
    'elastic',  # solvcon.kerpak.elastic
]
for ndim in 2, 3:
    for lkey in dimlibs:
        VariantDir('%s/%s%dd' % (bdir, lkey, ndim),
            '%s/%s' % (sdir, lkey), duplicate=0)
    envm = env.Clone()
    envm.Prepend(CCFLAGS=['-DNDIM=%d'%ndim])
    envm.Prepend(LIBS=['lapack'])
    for lkey in dimlibs:
        libs.append(envm.SharedLibrary(
            '%s/%s_%s%dd' % (ldir, lpre, lkey, ndim),
            Glob('%s/%s%dd/*.c' % (bdir, lkey, ndim)),
        ))

if GetOption('enable_f90'): # TODO: OBSELETE
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
ccflags.append('-O3')
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
