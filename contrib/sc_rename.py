#!/usr/bin/env python2.7

from __future__ import print_function

import sys
import os
import glob

def replace(pattern, oldname, newname):
    for oldfn in glob.glob(pattern):
        fobj = open(oldfn)
        data = fobj.read().replace(oldname, newname)
        fobj.close()
        newfn = oldfn.replace(oldname, newname)
        os.system('hg move %s %s' % (oldfn, newfn))
        fobj = open(newfn, 'w')
        fobj.write(data)
        fobj.close()
        print(oldfn, '->', newfn)

def main():
    replace(*sys.argv[1:])

if __name__ == '__main__':
    main()
