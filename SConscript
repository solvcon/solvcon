import os
Import('env')

# lib_solvcon.
def compile_objects(srcs, ext, F90FLAGS):
    ext = '.' + ext if ext[0] != '.' else ext
    objs = []
    for src in srcs:
        src = str(src)
        dst = os.path.splitext(src)[0] + ext
        objs.append(env.SharedObject(dst, src,
            F90FLAGS=' '.join(F90FLAGS),
        ))
    return objs
F90FLAGS = env['F90FLAGS']
SRCS = Glob('src/*/*.f90')
lib_solvcon_d = env.SharedLibrary('lib/_clib_solvcon_d',
    compile_objects(SRCS, '.dos', ['-DFPKIND=8', F90FLAGS]),
)
lib_solvcon_s = env.SharedLibrary('lib/_clib_solvcon_s',
    compile_objects(SRCS, '.sos', ['-DFPKIND=4', F90FLAGS]),
)

# lib_solvcontest.
lib_solvcontest = env.SharedLibrary('lib/_clib_solvcontest',
    Glob('test/src/*.f90'))
epydoc = env.Epydoc('solvcon/__init__.py')
sphinx = env.Sphinx('doc/source/conf.py')

Import('metisenv', 'metissrc')
lib_metis = metisenv.SharedLibrary('lib/_clib_metis',
    Glob('%s/*.c'%metissrc))

# name targets and set default.
solvcon = Alias('solvcon', [lib_solvcon_d, lib_solvcon_s, lib_solvcontest])
metis = Alias('metis', [lib_metis])
all = Alias('all', [metis, solvcon])
Default(all)
Alias('epydoc', epydoc)
Alias('sphinx', sphinx)
doc = Alias('doc', [epydoc, sphinx])

# vim: set ft=python ff=unix:
