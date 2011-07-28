#!/usr/bin/env python

site_tmpl = """[DEFAULT]
library_dirs = /usr/local/lib:%(SCROOT)s/lib
include_dirs =
/usr/local/include:%(SCROOT)s/include:%(SCROOT)s/include/atlas

[atlas]
atlas_libs = lapack, f77blas, cblas, atlas"""

setup_tmpl = """[config_fc]
fcompiler = gfortran
"""

import os

def main():
    f = open("site.cfg", 'w')
    f.write(site_tmpl % os.environ)
    f.close()

    f = open("setup.cfg", 'w')
    f.write(setup_tmpl)
    f.close()

if __name__ == '__main__':
    main()
