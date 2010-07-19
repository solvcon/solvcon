import os
Import('env')


# lib_solvconc.
clibs = list()
for fpmark, fptype in [('s', 'float'), ('d', 'double'),]:
    ddsts = list()
    for dsrc in Glob('src/*'):
        dsrc = str(dsrc)
        if not os.path.isdir(dsrc): continue
        ddst = 'lib/%s_%s' % (os.path.basename(dsrc), fpmark)
        VariantDir(ddst, dsrc, duplicate=0)
        ddsts.append(ddst)
    envm = env.Clone()
    envm.Prepend(CCFLAGS=['-DFPTYPE=%s'%fptype])
    clibs.append(envm.SharedLibrary(
        'lib/_clib_solvconc_%s' % fpmark,
        [Glob('%s/*.c'%ddst) for ddst in ddsts],
    ))

# lib_solvcon.
flibs = list()
for fpmark, fptype in [('s', 4), ('d', 8),]:
    ddsts = list()
    for dsrc in Glob('src/*'):
        dsrc = str(dsrc)
        if not os.path.isdir(dsrc): continue
        ddst = 'lib/%s_%s' % (os.path.basename(dsrc), fpmark)
        VariantDir(ddst, dsrc, duplicate=0)
        ddsts.append(ddst)
    envm = env.Clone()
    envm.Prepend(F90FLAGS=['-DFPKIND=%d'%fptype])
    flibs.append(envm.SharedLibrary(
        'lib/_clib_solvcon_%s' % fpmark,
        [Glob('%s/*.f90'%ddst) for ddst in ddsts],
    ))

# lib_solvcontest.
lib_solvcontest = env.SharedLibrary('lib/_clib_solvcontest',
    Glob('test/src/*.f90'))
epydoc = env.Epydoc('solvcon/__init__.py')
sphinx = env.Sphinx('doc/source/conf.py')

Import('metisenv', 'metissrc')
lib_metis = metisenv.SharedLibrary('lib/_clib_metis',
    Glob('%s/*.c'%metissrc))

# name targets and set default.
solvcon = Alias('solvcon', clibs + flibs + [lib_solvcontest])
metis = Alias('metis', [lib_metis])
all = Alias('all', [metis, solvcon])
Default(all)
Alias('epydoc', epydoc)
Alias('sphinx', sphinx)
doc = Alias('doc', [epydoc, sphinx])

# vim: set ft=python ff=unix:
