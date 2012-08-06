"""
Tool to run Cython files (.pyx) into .c and .cpp.

TODO:
 - Add support for dynamically selecting in-process Cython
   through CYTHONINPROCESS variable.
 - Have a CYTHONCPP option which turns on C++ in flags and
   changes output extension at the same time

VARIABLES:
 - CYTHON - The path to the "cython" command line tool.
 - CYTHONFLAGS - Flags to pass to the "cython" command line tool.

AUTHORS:
 - David Cournapeau
 - Dag Sverre Seljebotn

"""
import os
import re
try:
    set
except NameError:
    from sets import Set as set
import itertools

import SCons
from SCons.Builder import Builder
from SCons.Scanner import Scanner
from SCons.Action import Action

#def cython_action(target, source, env):
#    print target, source, env
#    from Cython.Compiler.Main import compile as cython_compile
#    res = cython_compile(str(source[0]))

cythonAction = Action("$CYTHONCOM")

def create_builder(env):
    try:
        cython = env['BUILDERS']['Cython']
    except KeyError:
        cython = SCons.Builder.Builder(
                  action = cythonAction,
                  emitter = {},
                  suffix = cython_suffix_emitter,
                  single_source = 1)
        env['BUILDERS']['Cython'] = cython

    return cython

def cython_suffix_emitter(env, source):
    return "$CYTHONCFILESUFFIX"

def cython_scan(node, env, path, arg=None):
    """
    A simple .pyx scanner for one-line cimports of .pxd files.
    """
    # scan file to extract all possible cimports.
    contents = node.get_text_contents()
    names = [reo.findall(contents) for reo in [
        re.compile(r'^\s*from\s+(.+?)\s+cimport\s.*$', re.M),
        re.compile(r'^\s*cimport\s+(.+?)$', re.M),
    ]]
    names = itertools.chain(*names)
    # keep characters before " as ".
    names = [name.split(' as ')[0] for name in names]
    # split each cimport.
    names = itertools.chain(*[name.split(',') for name in names])
    names = [name.strip() for name in names]
    # remove duplications.
    names = set(names)
    # prepend with the directory of the original pyx file.
    prefix = os.path.dirname(env.GetBuildPath(node))
    names = [os.path.join(prefix, '%s.pxd'%name) for name in names]
    # only take local pxd file and ignore anything unfound.
    names = [name for name in names if os.path.exists(name)]
    return [env.File(name) for name in names]

def generate(env):
    env["CYTHON"] = "cython"
    env["CYTHONCOM"] = "$CYTHON $CYTHONFLAGS -o $TARGET $SOURCE"
    env["CYTHONCFILESUFFIX"] = ".c"

    c_file, cxx_file = SCons.Tool.createCFileBuilders(env)

    c_file.suffix['.pyx'] = cython_suffix_emitter
    c_file.add_action('.pyx', cythonAction)

    c_file.suffix['.py'] = cython_suffix_emitter
    c_file.add_action('.py', cythonAction)

    create_builder(env)

    pyxscanner = Scanner(function=cython_scan, skeys=['.pyx'], name='PYX')
    env.Append(SCANNERS=[pyxscanner])

def exists(env):
    try:
#        import Cython
        return True
    except ImportError:
        return False
