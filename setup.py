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

def make_extension(name, c_subdirs, include_dirs=None, libraries=None):
    import sys, os
    from glob import glob
    from numpy.distutils.core import Extension
    pak_dir = os.path.join(*name.split('.')[:-1])
    files = [name.replace('.', os.sep) + '.c']
    for c_subdir in sorted(c_subdirs):
        path = os.path.join(pak_dir, c_subdir, '*.c')
        files += glob(path)
    include_dirs = [] if None is include_dirs else include_dirs
    include_dirs.insert(0, 'solvcon')
    include_dirs.insert(0, os.path.join(pak_dir))
    libraries = [] if None is libraries else libraries
    libraries = (['scotchmetis', 'scotch', 'scotcherr', 'scotcherrexit']
                 + libraries)
    rpathflag = '-Wl,-rpath,%s/lib' % sys.exec_prefix
    return Extension(name, files,
                     include_dirs=include_dirs, libraries=libraries,
                     extra_link_args=[rpathflag])

def main():
    import os, sys
    # BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
    # update it when the contents of directories change.
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')
    from glob import glob
    from numpy.distutils.core import setup
    import solvcon

    data_files = list()
    # includes.
    data_files.append((os.path.join('include', 'solvcon'),
        glob(os.path.join('include', '*'))))
    # libraries.  NOTE: this directory should be manually removed for sdist.
    data_files.append((os.path.join('lib', 'solvcon'),
        glob(os.path.join('lib', '*'))))
    # test data.
    lead = os.path.join('share', 'solvcon', 'test')
    data_files.extend([
        (lead, glob(os.path.join('test', 'data', '*.g'))),
        (lead, glob(os.path.join('test', 'data', '*.jou'))),
        (lead, glob(os.path.join('test', 'data', '*.nc'))),
        (lead, glob(os.path.join('test', 'data', '*.neu'))),
        (lead, glob(os.path.join('test', 'data', '*.blk'))),
        (lead, glob(os.path.join('test', 'data', '*.vtk'))),
        (os.path.join(lead, 'sample.dom'),
            glob(os.path.join('test', 'data', 'sample.dom', '*')))])
    # examples.
    lead = os.path.join('share', 'solvcon')
    for edir in glob(os.path.join('examples', '*', '*')):
        data_files.append(
            (os.path.join(lead, edir), [os.path.join(edir, 'go')]))
        for ext in ('tmpl', 'py', 'h'):
            data_files.append((os.path.join(lead, edir),
                glob(os.path.join(edir, '*.%s'%ext))))

    setup(
        name='SOLVCON',
        maintainer='Yung-Yu Chen',
        author='Yung-Yu Chen',
        maintainer_email='yyc@solvcon.net',
        author_email='yyc@solvcon.net',
        description='Solvers of Conservation Laws',
        long_description=''.join(open('README.rst').readlines()[4:]),
        license='BSD',
        url='http://solvcon.net/',
        download_url='http://bitbucket.org/yungyuc/solvcon/downloads/',
        classifiers=[tok.strip() for tok in CLASSIFIERS.split('\n')],
        platforms=[
            'Linux',
        ],
        version=solvcon.__version__,
        scripts=[
            'scg',
        ],
        packages=[
            'solvcon',
            'solvcon.io',
            'solvcon.io.tests',
            'solvcon.kerpak',
            'solvcon.parcel',
            'solvcon.tests',
        ],
        ext_modules=[
            make_extension('solvcon.mesh', ['src']),
            make_extension('solvcon.parcel.fake._algorithm', ['src']),
            make_extension('solvcon.parcel.linear._algorithm', ['src'],
                           libraries=['lapack', 'blas']),
            make_extension('solvcon.parcel.bulk._algorithm', ['src']),
            make_extension('solvcon.parcel.gas._algorithm', ['src']),
            make_extension('solvcon.parcel.vewave._algorithm', ['src'],
                           libraries=['lapack', 'blas']),
        ],
        data_files=data_files,
    )
    return

if __name__ == '__main__':
    main()
