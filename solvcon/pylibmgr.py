# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import sys
import os
import time
import importlib.abc
import importlib.machinery


class SolvconPathFinder(importlib.abc.MetaPathFinder):
    def __init__(self, lib_paths):
        self.lib_paths = lib_paths

    def find_spec(self, lib_name, path, target=None):
        if lib_name in self.lib_paths:
            _ = os.path.abspath(self.lib_paths[lib_name])
            pkg_path = os.path.join(_, lib_name)
            init_path = os.path.join(pkg_path, '__init__.py')

            if not os.path.exists(init_path):
                return None

            # Create a loader instance for the given package name and
            # it's __init__.py
            loader = importlib.machinery.SourceFileLoader(lib_name, init_path)

            # Create a module spec instance for the given module name,
            # specific loader and the path of module file.
            # If the module is a package the argument is_package should be
            # set to True.
            # Ref:
            # https://docs.python.org/3/library/importlib.html#importlib.
            # machinery.ModuleSpec
            spec = importlib.machinery.ModuleSpec(
                    lib_name, loader, origin=init_path, is_package=True)
            # This attribute tells the import system where to look for
            # submodules or subpackages, it should not be set to None
            # for a package modules.
            # Ref:
            # https://docs.python.org/3/library/importlib.html#importlib.
            # machinery.ModuleSpec.submodule_search_locations
            spec.submodule_search_locations = [pkg_path]
            return spec

        return None


def is_solvcon_meta_path_finder_registered():
    return any(isinstance(finder, SolvconPathFinder)
               for finder in sys.meta_path)


def search_library_root(curr_path, lib_root_name, timeout=1.0):
    """
    Walk through the thridparty library and register all third-party
    library folder path into a dictionary with library name as the key,
    user can select which library need to be imported by library name.

    :param curr_path: The path to start searching from.
    :type curr_path: str
    :param lib_root_name: The name of the root folder containing libraries.
    :type lib_root_name: str
    :param timeout: Maximum time for root path searching. Default is 1.0 sec.
    :type timeout: float
    :return: None
    """
    # Try to find the library root, if failed to find it
    # solvcon will raise ImportError and remind the user.
    lib_path = {}
    folder_name = os.path.join(lib_root_name)
    _path = curr_path
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(os.path.join(_path, folder_name)):
            break
        if _path == os.path.dirname(_path):
            _path = None
            break
        else:
            _path = os.path.dirname(_path)
    _pf = os.path.join(_path, folder_name) if _path else None
    if _path is None or not os.path.exists(_pf):
        sys.stderr.write(f'Can not find {lib_root_name}.\n')
        return

    _path = os.path.join(_path, folder_name)

    for item in os.listdir(_path):
        if os.path.isdir(os.path.join(_path, item)):
            lib_path[item] = os.path.join(_path, item)

    if not is_solvcon_meta_path_finder_registered():
        sys.meta_path.append(SolvconPathFinder(lib_path))
# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
