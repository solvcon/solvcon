# Copyright (c) 2020, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import pathlib
import subprocess

import setuptools
from setuptools.command import build_ext


# Taken from https://stackoverflow.com/a/48015772
class CMakeExtension(setuptools.Extension):
    def __init__(self, name, **kwa):
        super().__init__(name, sources=[])


class cmake_build_ext(build_ext.build_ext):
    user_options = build_ext.build_ext.user_options + [
        ('cmake-args=', None, 'arguments to cmake'),
        ('make-args=', None, 'arguments to make'),
    ]

    def initialize_options(self):

        super().initialize_options()
        self.cmake_args = ''
        self.make_args = ''

    def finalize_options(self):

        super().finalize_options()

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):

        cwd = pathlib.Path().absolute()

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name)).parent
        extdir.mkdir(parents=True, exist_ok=True)

        local_cmake_args = '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(
            str(extdir.absolute()))

        cmd = 'cmake {} {} {}'.format(cwd, local_cmake_args, self.cmake_args)
        proc = subprocess.run(cmd, shell=True, cwd=str(build_temp))
        if 0 != proc.returncode:
            raise RuntimeError('{} return {}'.format(cmd, proc.returncode))

        target_name = ext.name.split('.')[-1]
        cmd = 'make {} {}'.format(target_name, self.make_args)
        proc = subprocess.run(cmd, shell=True, cwd=str(build_temp))
        if 0 != proc.returncode:
            raise RuntimeError('{} return {}'.format(cmd, proc.returncode))


def main():
    setuptools.setup(
        name="solvcon",
        version="0.0",
        packages=[
            'solvcon',
            'solvcon.agent',
            'solvcon.multidim',
            'solvcon.multidim.euler',
            'solvcon.onedim',
            'solvcon.pilot',
            'solvcon.pilot.airfoil',
            'solvcon.plot',
            'solvcon.profiling',
            'solvcon.track',
        ],
        ext_modules=[CMakeExtension("_solvcon")],
        cmdclass={'build_ext': cmake_build_ext},
    )


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
