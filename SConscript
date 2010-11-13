import os
Import('env')

libs = list()
if not GetOption('enable_f90'):
    # lib_solvcon in C.
    for fpmark, fptype in [('s', 'float'), ('d', 'double'),]:
        ddsts = list()
        for dsrc in ['block', 'domain', 'partition', 'solve']:
            dsrc = 'src/%sc' % dsrc
            if not os.path.isdir(dsrc): continue
            ddst = 'lib/%s_%s' % (os.path.basename(dsrc), fpmark)
            VariantDir(ddst, dsrc, duplicate=0)
            ddsts.append(ddst)
        envm = env.Clone()
        envm.Prepend(CCFLAGS=['-DFPTYPE=%s'%fptype])
        libs.append(envm.SharedLibrary(
            'lib/_clib_solvcon_%s' % fpmark,
            [Glob('%s/*.c'%ddst) for ddst in ddsts],
        ))
    # lib_cese.
    dimlibs = ['cese', 'ceseb']
    for ndim in 2, 3:
        for lkey in dimlibs:
            VariantDir('lib/%s%dd'%(lkey, ndim), 'src/%s'%lkey, duplicate=0)
        envm = env.Clone()
        envm.Prepend(CCFLAGS=['-DNDIM=%d'%ndim])
        for lkey in dimlibs:
            libs.append(envm.SharedLibrary(
                'lib/_clib_%s%dd'%(lkey, ndim),
                Glob('lib/%s%dd/*.c'%(lkey, ndim)),
            ))
    # lib_solvcontest in C.
    lib_solvcontest = env.SharedLibrary('lib/_clib_solvcontest',
        Glob('test/src/*.c'))
else:
    # lib_solvcon in FORTRAN.
    for fpmark, fptype in [('s', 4), ('d', 8),]:
        ddsts = list()
        for dsrc in ['block', 'domain', 'partition', 'solve']:
            dsrc = 'src/%sf' % dsrc
            if not os.path.isdir(dsrc): continue
            ddst = 'lib/%s_%s' % (os.path.basename(dsrc), fpmark)
            VariantDir(ddst, dsrc, duplicate=0)
            ddsts.append(ddst)
        envm = env.Clone()
        envm.Prepend(F90FLAGS=['-DFPKIND=%d'%fptype])
        libs.append(envm.SharedLibrary(
            'lib/_clib_solvcon_%s' % fpmark,
            [Glob('%s/*.f90'%ddst) for ddst in ddsts],
        ))
    # lib_solvcontest.
    lib_solvcontest = env.SharedLibrary('lib/_clib_solvcontest',
        Glob('test/src/*.f90'))

# METIS.
Import('metisenv', 'metissrc')
lib_metis = metisenv.SharedLibrary('lib/_clib_metis',
    Glob('%s/*.c'%metissrc))

# documents.
epydoc = env.Epydoc('solvcon/__init__.py')
sphinx = env.Sphinx('doc/source/conf.py')

# name targets and set default.
solvcon = Alias('solvcon', libs + [lib_solvcontest])
metis = Alias('metis', [lib_metis])
all = Alias('all', [metis, solvcon])
Default(all)
Alias('epydoc', epydoc)
Alias('sphinx', sphinx)
doc = Alias('doc', [epydoc, sphinx])

# vim: set ft=python ff=unix:
