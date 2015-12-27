#!/usr/bin/env python

import os
import argparse

from notebook.nbextensions import install_nbextension

def install(user=False, symlink=False, **kw):
    """
    Install SOLVCON's nbextension.
    
    :param user: Install to current user's home.
    :type user: bool
    :param symlink: Do symbolic link instead of copy.
    :type symlink: bool
    """
    directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'nbextension')
    install_nbextension(directory, destination='solvcon',
                        symlink=symlink, user=user, **kw)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Installs SOLVCON widgets")
    parser.add_argument("-u", "--user",
                        help="Install to current user's home",
                        action="store_true")
    parser.add_argument("-s", "--symlink",
                        help="Do symbolic link instead of copy",
                        action="store_true")
    parser.add_argument("-f", "--force",
                        help="Overwrite any previously-installed files "
                             "for this extension",
                        action="store_true")
    args = parser.parse_args()
    install(user=args.user, symlink=args.symlink, overwrite=args.force)
