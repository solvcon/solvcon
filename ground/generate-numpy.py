#!/usr/bin/env python

site_tmpl = """[DEFAULT]
library_dirs = /usr/local/lib:%(BUILD)s/ATLAS/ATLAS_LINUX/lib:%(BUILD)s/umfpack:%(OPT)s/fftw3/lib
include_dirs = /usr/local/include:%(BUILD)s/ATLAS/ATLAS_LINUX/include:%(BUILD)s/umfpack:%(OPT)s/fftw3/include

[atlas]
atlas_libs = lapack, f77blas, cblas, atlas

[amd]
amd_libs = amd

[umfpack]
umfpack_libs = umfpack

[fftw]
# fftw no longer supported in scipy 0.7.
libraries = fftw3"""

setup_tmpl = """[config_fc]
fcompiler = gnu95
"""

import os

def main():
    f = open("site.cfg", 'w')
    f.write(site_tmpl % dict(
        BUILD=os.environ.get('HOME', '')+'/build/lib',
        OPT=os.environ.get('HOME', '')+'/opt',
    ))
    f.close()

    f = open("setup.cfg", 'w')
    f.write(setup_tmpl)
    f.close()

if __name__ == '__main__':
    main()
