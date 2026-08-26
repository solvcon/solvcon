# Copyright (c) 2020, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import pathlib
import shlex
import subprocess
import sys

import setuptools
from setuptools.command import build_ext


# Taken from https://stackoverflow.com/a/48015772
class CMakeExtension(setuptools.Extension):
    def __init__(self, name, **kwa):
        super().__init__(name, sources=[])


def split_command_args(value):

    if os.name != 'nt':
        return shlex.split(value)

    # A Windows path argument (-DCMAKE_PREFIX_PATH=C:\scdv\usr) would lose
    # its backslashes to shlex escape processing, so disable escapes there.
    lexer = shlex.shlex(value, posix=True)
    # A '#' is a legal character in a cmake argument, not a
    # comment introducer.
    lexer.commenters = ''
    lexer.whitespace_split = True
    lexer.escape = ''
    # cmd.exe has no single-quote quoting, and an apostrophe is
    # ordinary in a Windows path (C:\Users\O'Brien).
    lexer.quotes = '"'
    return list(lexer)


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
        config = 'Debug' if self.debug else 'Release'

        local_cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(config),
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(
                str(extdir.absolute())),
            '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
        ]
        if os.name == 'nt':
            local_cmake_args.append(
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    config.upper(), str(extdir.absolute())))

        cmd = ['cmake', '-S', str(cwd), '-B', '.'] + local_cmake_args
        cmd.extend(split_command_args(self.cmake_args))
        subprocess.run(cmd, check=True, cwd=str(build_temp))

        target_name = ext.name.split('.')[-1]
        cmd = [
            'cmake', '--build', '.', '--config', config,
            '--target', target_name,
        ]
        make_args = split_command_args(self.make_args)
        if make_args:
            cmd.extend(['--'] + make_args)
        subprocess.run(cmd, check=True, cwd=str(build_temp))


def main():
    setuptools.setup(
        name="solvcon",
        version="0.0",
        packages=[
            'solvcon',
            'solvcon.agent',
            'solvcon.agent.draw',
            'solvcon.agent.window',
            'solvcon.mcap',
            'solvcon.multidim',
            'solvcon.multidim.euler',
            'solvcon.onedim',
            'solvcon.pilot',
            'solvcon.pilot.agent',
            'solvcon.pilot.airfoil',
            'solvcon.pilot.apps',
            'solvcon.pilot.apps.obsrefl',
            'solvcon.pilot.base',
            'solvcon.pilot.canvas',
            'solvcon.pilot.onedim',
            'solvcon.pilot.painter',
            'solvcon.pilot.panel',
            'solvcon.pilot.visual',
            'solvcon.plot',
            'solvcon.profiling',
            'solvcon.track',
        ],
        install_requires=['jsonschema>=4'],
        ext_modules=[CMakeExtension("_solvcon")],
        cmdclass={'build_ext': cmake_build_ext},
    )


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
