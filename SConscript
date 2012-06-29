import os, sys
Import('everything', 'env')

lpre = 'sc'
ldir = 'lib'
sdir = 'src'
bdir = 'build'
libs = list()

# lib_solvcon.
for fptype in ['float', 'double']:
    libs.append(env.SolvconShared(
        ['block', 'domain', 'partition', 'solve'], 'solvcon', fptype))

# lib_solvcontest.
libs.append(env.SolvconShared(['src'], 'solvcontest', srcroot='test'))
# lib_cuda13test.
if FindFile('nvcc', os.environ['PATH'].split(':')):
    envm = env.Clone()
    envm['NVCCFLAGS'] = ['-arch=sm_13']
    envm.Append(LIBS=['cudart'])
    VariantDir('%s/cuda13test' % bdir, 'test/cudasrc', duplicate=0)
    libs.append(envm.SharedLibrary('%s/%s_cuda13test' % (ldir, lpre),
        Glob('%s/cuda13test/*.cu'%bdir)))
# lib_cuda20test.
if FindFile('nvcc', os.environ['PATH'].split(':')):
    envm = env.Clone()
    envm['NVCCFLAGS'] = ['-arch=sm_20']
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

# kerpak libraries with CUDA.
def make_kpculib(lname, lpre, ldir, sdir, bdir, env, extra_links=None):
    """
    Make kerpak libraries with CUDA.
    """
    libs = []
    for ndim in 2, 3:
        for ext in ('c', 'cu'):
            if ext == 'cu':
                if not FindFile('nvcc', os.environ['PATH'].split(':')):
                    continue
            VariantDir('%s/%s%dd_%s' % (bdir, lname, ndim, ext),
                sdir, duplicate=0)
            envm = env.Clone()
            envm.Prepend(CCFLAGS=['-DNDIM=%d'%ndim])
            envm.Prepend(NVCCFLAGS=['-DNDIM=%d'%ndim])
            envm.Append(NVCCINC=' -I src/cuse')
            if ext == 'cu': envm.Append(LIBS=['cudart'])
            if extra_links is not None:
                envm.Prepend(LIBS=extra_links)
            libs.append(envm.SharedLibrary(
                '%s/%s_%s%dd_%s' % (ldir, lpre, lname, ndim, ext),
                Glob('%s/%s%dd_%s/*.%s' % (bdir, lname, ndim, ext, ext))))
    return libs
kpculibs = [
    ('cuse', None), ('cuseb', None),    # solvcon.kerpak.cuse
    ('gasdyn', None), ('gasdynb', None),    # solvcon.kerpak.gasdyn
    ('lincuse', ['lapack']),    # solvcon.kerpak.lincese
    ('vslin', None), ('vslinb', None),  # solvcon.kerpak.vslin
]
for lname, extra_links in kpculibs:
    libs.extend(make_kpculib(lname, lpre, ldir, '%s/%s'%(sdir, lname),
        bdir, env, extra_links=extra_links))

# documents.
epydoc = env.BuildEpydoc('solvcon/__init__.py')
sphinx = env.BuildSphinx(Glob('doc/source/*.rst')+Glob('doc/source/*.py'))

# name targets.
solvcon = Alias('solvcon', libs)
everything.append(solvcon)
Alias('epydoc', epydoc)
Alias('sphinx', sphinx)
doc = Alias('doc', [epydoc, sphinx])

# vim: set ff=unix ft=python fenc=utf8 ai et sw=4 ts=4 tw=79:
