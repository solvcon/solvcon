"""
SConscript: The defined rules.
"""

import os, sys
Import('targets', 'env')

# libsolvcon.
sclib = targets.setdefault('sclib', [])
sclib.append(env.SolvconStatic(['mesh', 'fake_algorithm'], 'solvcon', sclibprefix=''))

# lib_solvcon.
scmain = targets.setdefault('scmain', [])
for fptype in ['float', 'double']:
    scmain.append(env.SolvconShared(['solve'], 'solvcon', fptype=fptype))

# lib_solvcontest.
sctest = targets.setdefault('sctest', [])
sctest.append(env.SolvconShared(['src'], 'solvcontest', srcdir='test'))
# lib_cuda13test.
if FindFile('nvcc', os.environ['PATH'].split(':')):
    envm = env.Clone()
    envm['NVCCFLAGS'] = ['-arch=sm_13']
    envm.Append(LIBS=['cudart'])
    VariantDir('%s/cuda13test' % envm['SCBUILDDIR'], 'test/cudasrc',
        duplicate=0)
    sctest.append(envm.SharedLibrary('%s/%s_cuda13test' % (envm['SCLIBDIR'],
        envm['SCLIBPREFIX']), Glob('%s/cuda13test/*.cu'%envm['SCBUILDDIR'])))
# lib_cuda20test.
if FindFile('nvcc', os.environ['PATH'].split(':')):
    envm = env.Clone()
    envm['NVCCFLAGS'] = ['-arch=sm_20']
    envm.Append(LIBS=['cudart'])
    VariantDir('%s/cuda20test' % envm['SCBUILDDIR'], 'test/cudasrc',
        duplicate=0)
    sctest.append(envm.SharedLibrary('%s/%s_cuda20test' % (envm['SCLIBDIR'],
        envm['SCLIBPREFIX']), Glob('%s/cuda20test/*.cu'%envm['SCBUILDDIR'])))

# kerpak libraries.
sckp = targets.setdefault('sckp', [])
for lname, extra_links in [
        ('cese', None), ('ceseb', None),    # solvcon.kerpak.cese
        ('elaslin', None), ('elaslinb', None),  # solvcon.kerpak.elaslin
        ('euler', None), ('eulerb', None),  # solvcon.kerpak.euler
        ('lincese', ['lapack']),    # solvcon.kerpak.lincese
        ]:
    for ndim in 2, 3:
        sckp.extend(env.SolvconShared([lname], lname, ndim=ndim,
            prepends={'LIBS': extra_links}))

# kerpak libraries with CUDA.
sckpcu = targets.setdefault('sckpcu', [])
for lname, extra_links in [
        ('cuse', None), ('cuseb', None),    # solvcon.kerpak.cuse
        ('gasdyn', None), ('gasdynb', None),    # solvcon.kerpak.gasdyn
        ('lincuse', ['lapack']),    # solvcon.kerpak.lincese
        ('vslin', None), ('vslinb', None),  # solvcon.kerpak.vslin
        ]:
    for ndim in 2, 3:
        for ext in ('c', 'cu'):
            prepends = {'LIBS': extra_links}
            if ext == 'cu':
                if not FindFile('nvcc', os.environ['PATH'].split(':')):
                    continue
                prepends['NVCCINC'] = ' -I src/cuse'
                prepends['LIBS'].append('cudart')
            sckpcu.extend(env.SolvconShared([lname], lname, ndim=ndim, ext=ext,
                prepends=prepends))

# cython.
scmods = targets.setdefault('scmods', [])
env.VariantDir('%s/cython'%env['SCBUILDDIR'], 'cython', duplicate=1)
prepends = {'LIBS': ['solvcon']}
scmods.extend(env.SolvconModule(
    env.Glob('%s/cython/*.pyx'%env['SCBUILDDIR']), prepends=prepends))

# documents.
targets['scepydoc'] = env.BuildEpydoc('solvcon/__init__.py')
targets['scsphinx'] = env.BuildSphinx(
    Glob('doc/source/*.rst')+Glob('doc/source/*.py'))

# vim: set ff=unix ft=python fenc=utf8 ai et sw=4 ts=4 tw=79:
