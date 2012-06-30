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
        ['block', 'domain', 'partition', 'solve'], 'solvcon', fptype=fptype))

# lib_solvcontest.
libs.append(env.SolvconShared(['src'], 'solvcontest', srcdir='test'))
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
kplibs = [
    ('cese', None), ('ceseb', None),    # solvcon.kerpak.cese
    ('elaslin', None), ('elaslinb', None),  # solvcon.kerpak.elaslin
    ('euler', None), ('eulerb', None),  # solvcon.kerpak.euler
    ('lincese', ['lapack']),    # solvcon.kerpak.lincese
]
for lname, extra_links in kplibs:
    for ndim in 2, 3:
        libs.extend(env.SolvconShared([lname], lname, ndim=ndim,
            prepends={'LIBS': extra_links}))

# kerpak libraries with CUDA.
kpculibs = [
    ('cuse', None), ('cuseb', None),    # solvcon.kerpak.cuse
    ('gasdyn', None), ('gasdynb', None),    # solvcon.kerpak.gasdyn
    ('lincuse', ['lapack']),    # solvcon.kerpak.lincese
    ('vslin', None), ('vslinb', None),  # solvcon.kerpak.vslin
]
for lname, extra_links in kpculibs:
    for ndim in 2, 3:
        for ext in ('c', 'cu'):
            prepends = {'LIBS': extra_links}
            if ext == 'cu':
                if not FindFile('nvcc', os.environ['PATH'].split(':')):
                    continue
                prepends['NVCCINC'] = ' -I src/cuse'
                prepends['LIBS'].append('cudart')
            libs.extend(env.SolvconShared([lname], lname, ndim=ndim, ext=ext,
                prepends=prepends))

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
