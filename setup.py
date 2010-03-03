"""Unstructured mesh manipulator."""

import os

CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Developers
Intended Audience :: Education
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Programming Language :: Fortran
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Application Frameworks
"""

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

def main():
    import os, sys
    from distutils.core import setup
    import solvcon

    doclines = solvcon.__doc__.split('\n')

    setup(
        name='solvcon',
        maintainer='Yung-Yu Chen',
        author='Yung-Yu Chen',
        maintainer_email='yyc@seety.org',
        author_email='yyc@seety.org',
        description=doclines[0],
        long_description='\n'.join(doclines[2:]),
        license='GPL',
        #url='http://cfd.eng.ohio-state.edu/~yungyuc/solvcon/',
        #download_url='http://cfd.eng.ohio-state.edu/~yungyuc/solvcon/',
        classifiers=[tok.strip() for tok in CLASSIFIERS.split('\n')],
        platforms=[
            'Linux',
            'Windows',
        ],
        version=solvcon.__version__,
        packages=[
            'solvcon',
            'solvcon.io',
            'solvcon.io.tests',
            'solvcon.tests',
        ],
    )
    return

if __name__ == '__main__':
    main()
