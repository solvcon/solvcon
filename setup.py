# -*- coding: UTF-8 -*-
#
# Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""SOLVCON distribution script."""

from __future__ import absolute_import, division, print_function

CLASSIFIERS = """Development Status :: 3 - Alpha
Intended Audience :: Developers
Intended Audience :: Education
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Programming Language :: C
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Application Frameworks"""


import sys
import os
import glob

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

from distutils.ccompiler import CCompiler
from numpy.distutils.ccompiler import replace_method
from numpy.distutils.ccompiler import CCompiler_customize as numpy_CCompiler_customize
from numpy.distutils import log

from distutils.extension import Extension
from numpy.distutils.core import setup
from numpy.distutils.command import build_ext as np_build_ext
from Cython.Build import cythonize
from Cython.Distutils import Extension as CyExtension

import solvcon as sc


def CCompiler_customize(self, *args, **kw):
    need_cxx = kw.get('need_cxx', 0)

    # list unwanted flags (e.g. '-g') here.
    unwanted = []

    # call the original method.
    numpy_CCompiler_customize(self, *args, **kw)

    # update arguments.
    ccshared = ' '.join(set(self.compiler_so) - set(self.compiler))
    compiler = ' '.join(it for it in self.compiler if it not in unwanted)
    old_compiler = self.compiler
    self.set_executables(
        compiler=compiler,
        compiler_so=compiler + ' ' + ccshared,
    )
    modified = self.compiler != old_compiler
    if modified and need_cxx and hasattr(self, 'compiler'):
        log.warn("#### %s ####### %s removed" % (self.compiler, unwanted))

    return

replace_method(CCompiler, 'customize', CCompiler_customize)


def make_cython_extension(
    name, c_subdirs, include_dirs=None, libraries=None, extra_compile_args=None
):
    pak_dir = os.path.join(*name.split('.')[:-1])
    files = [name.replace('.', os.sep) + '.pyx']
    for c_subdir in sorted(c_subdirs):
        path = os.path.join(pak_dir, c_subdir, '*.c')
        files += glob.glob(path)
    include_dirs = [] if None is include_dirs else include_dirs
    include_dirs.insert(0, 'solvcon')
    include_dirs.insert(0, os.path.join(pak_dir))
    libraries = [] if None is libraries else libraries
    libraries = (['scotchmetis', 'scotch', 'scotcherr', 'scotcherrexit']
                 + libraries)
    rpathflag = '-Wl,-rpath,%s/lib' % sys.exec_prefix
    if extra_compile_args is None: extra_compile_args = []
    extra_compile_args = [
        '-Werror',
        '-Wno-cpp' if sys.platform != 'darwin' else '-Wno-#warnings',
        '-Wno-unused-function',
    ] + extra_compile_args
    return CyExtension(
        name, files,
        include_dirs=include_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=[rpathflag],
    )


class my_build_ext(np_build_ext.build_ext):

    def _copy_cmake_extension(self, ext):
        import shutil
        from distutils.errors import DistutilsSetupError
        from distutils.dep_util import newer_group

        sources = ext.sources
        if sources is None or not isinstance(sources, (list, tuple)):
            raise DistutilsSetupError(
                  "in 'ext_modules' option (extension '%s'), "
                  "'sources' must be present and must be "
                  "a list of source filenames" % ext.name)
        sources = list(sources)

        ext_path = self.get_ext_fullpath(ext.name)
        depends = sources + ext.depends
        if not (self.force or newer_group(depends, ext_path, 'newer')):
            log.debug("skipping '%s' cmake extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' cmake extension", ext.name)

        extdir, extbase =  os.path.split(ext_path)
        if not os.path.exists(extdir):
            os.makedirs(extdir)
        shutil.copyfile(
            os.path.join('solvcon', extbase), ext_path)

    def build_extension(self, ext):
        ''' Copies the already-compiled pyd
        '''
        return np_build_ext.build_ext.build_extension(self, ext)


def main():
    data_files = list()
    # includes.
    data_files.append((os.path.join('include', 'solvcon'),
                       glob.glob(os.path.join('include', '*'))))
    # javascript code.
    lead = os.path.join('solvcon', 'visual', 'js')
    for root, directory, files in os.walk(lead):
        files = [(lead, os.path.join(root, fname)) for fname in files]
        data_files.extend(files)
    # test data.
    lead = os.path.join('share', 'solvcon', 'test')
    data_files.extend([
        (lead, glob.glob(os.path.join('test', 'data', '*.g'))),
        (lead, glob.glob(os.path.join('test', 'data', '*.jou'))),
        (lead, glob.glob(os.path.join('test', 'data', '*.nc'))),
        (lead, glob.glob(os.path.join('test', 'data', '*.neu'))),
        (lead, glob.glob(os.path.join('test', 'data', '*.blk'))),
        (lead, glob.glob(os.path.join('test', 'data', '*.vtk'))),
        (lead, glob.glob(os.path.join('test', 'data', '*.msh.gz'))),
        (lead, glob.glob(os.path.join('test', 'data', '*.geo'))),
        (os.path.join(lead, 'sample.dom'),
         glob.glob(os.path.join('test', 'data', 'sample.dom', '*')))
    ])
    # examples.
    lead = os.path.join('share', 'solvcon')
    for edir in glob.glob(os.path.join('examples', '*', '*')):
        if os.path.isdir(edir):
            data_files.append(
                (os.path.join(lead, edir), [os.path.join(edir, 'go')]))
            for ext in ('tmpl', 'py', 'h'):
                data_files.append((os.path.join(lead, edir),
                    glob.glob(os.path.join(edir, '*.%s'%ext))))

    turn_off_unused_warnings = '-Wno-unused-variable'
    if sys.platform != 'darwin':
        turn_off_unused_warnings += ' -Wno-unused-but-set-variable'
    # set up extension modules.
    lapack_libraries = ['lapack', 'blas']
    if os.environ.get('LAPACK_GFORTRAN'):
        lapack_libraries.append('gfortran')
    ext_modules = [
        make_cython_extension(
            'solvcon._march_bridge', [],
            include_dirs=['libmarch/include']
        ),
        make_cython_extension(
            'solvcon.mesh',
            ['src'],
        ),
        make_cython_extension(
            'solvcon.parcel.fake._algorithm',
            ['src'],
            extra_compile_args=[
                turn_off_unused_warnings,
            ],
        ),
        make_cython_extension(
            'solvcon.parcel.linear._algorithm', ['src'],
            libraries=lapack_libraries,
            extra_compile_args=[
                turn_off_unused_warnings,
                '-Wno-unknown-pragmas',
            ],
        ),
        make_cython_extension(
            'solvcon.parcel.bulk._algorithm',
            ['src'],
            extra_compile_args=[
                turn_off_unused_warnings,
                '-Wno-unknown-pragmas',
                '-Wno-uninitialized',
            ],
        ),
        make_cython_extension(
            'solvcon.parcel.gas._algorithm',
            ['src'],
            extra_compile_args=[
                turn_off_unused_warnings,
                '-Wno-unknown-pragmas',
            ],
        ),
        make_cython_extension(
            'solvcon.parcel.vewave._algorithm', ['src'],
            libraries=['lapack', 'blas'],
            extra_compile_args=[
                turn_off_unused_warnings,
                '-Wno-unknown-pragmas',
            ],
        ),
    ]

    # remove files when cleaning.
    sidx = sys.argv.index('setup.py') if 'setup.py' in sys.argv else -1
    cidx = sys.argv.index('clean') if 'clean' in sys.argv else -1
    if cidx > sidx:
        derived = list()
        for mod in ext_modules:
            pyx = mod.sources[0] # this must be the pyx file.
            mainfn, dotfn = os.path.splitext(pyx)
            if '.pyx' == dotfn:
                derived += ['.'.join((mainfn, ext)) for ext in ('c', 'h')]
            derived += ['%s.so' % mainfn] + glob.glob('%s.*.so' % mainfn)
        derived = [fn for fn in derived if os.path.exists(fn)]
        if derived:
            sys.stdout.write('Removing in-place generated files:')
            for fn in derived:
                os.remove(fn)
                sys.stdout.write('\n %s' % fn)
            sys.stdout.write('\n')
    else:
        if "/home/docs/checkouts/readthedocs.org" in os.getcwd():
            # Do not build extension modules if I am in readthedocs.org,
            # because the dependency cannot be met.
            ext_modules = list()
        else:
            ext_modules = cythonize(ext_modules)

    with open('README.rst') as fobj:
        long_description = ''.join(fobj.read())
    setup(
        name='SOLVCON',
        maintainer='Yung-Yu Chen',
        author='Yung-Yu Chen',
        maintainer_email='yyc@solvcon.net',
        author_email='yyc@solvcon.net',
        description='Solvers of Conservation Laws',
        long_description=long_description,
        license='BSD',
        url='http://solvcon.net/',
        download_url='https://github.com/solvcon/solvcon/releases',
        classifiers=[tok.strip() for tok in CLASSIFIERS.split('\n')],
        platforms=[
            'Linux',
        ],
        version=sc.__version__,
        scripts=[
            'scg',
        ],
        packages=[
            'solvcon',
            'solvcon.io',
            'solvcon.io.tests',
            'solvcon.kerpak',
            'solvcon.parcel',
            'solvcon.parcel.bulk',
            'solvcon.parcel.fake',
            'solvcon.parcel.gas',
            'solvcon.parcel.linear',
            'solvcon.parcel.tests',
            'solvcon.parcel.vewave',
            'solvcon.tests',
            'solvcon.vis',
        ],
        package_data={
            'solvcon.vis': ["js/*"],
        },
        cmdclass={
            'build_ext': my_build_ext,
        },
        ext_modules=ext_modules,
        data_files=data_files,
    )
    return

if __name__ == '__main__':
    main()
