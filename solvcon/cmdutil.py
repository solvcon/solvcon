# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2010 Yung-Yu Chen.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
Supporting functionalities and structures for UI commands.
"""

from .gendata import SingleAssignDict, AttributeDict

class CommandRegistry(SingleAssignDict, AttributeDict):
    def register(self, cmdtype):
        name = cmdtype.__name__
        if name.islower():
            self[name] = cmdtype
            return cmdtype
        else:
            return None
cmdregy = CommandRegistry() # registry singleton.

class CommandMeta(type):
    def __new__(cls, name, bases, namespace):
        newcls = super(CommandMeta, cls).__new__(cls, name, bases, namespace)
        # register.
        cmdregy.register(newcls)
        return newcls

class Command(object):
    """
    Command line parameters.

    @cvar min_args: minimal length of command line arguments to a command.
    @ctype min_args: int
    @ivar env: environmental object.
    @itype env: solvcon.conf.Solvcon
    @ivar op: overall option parser object.
    @itype op: optparse.OptionParser
    @ivar opg_global: group for global options.
    @itype opg_global: optparse.OptionGroup
    @ivar _opargs: tuple for storing options and arguments.
    @itype _opargs: tuple
    @ivar _usage: generic usage string for the UI.
    @itype _usage: str
    """

    __metaclass__ = CommandMeta

    min_args = 0

    def __init__(self, env):
        from optparse import OptionParser, OptionGroup
        from . import __version__

        self._usage = '%prog command <args> ... <ops> ...'

        op = OptionParser(usage=self._usage, version=__version__)

        opg = OptionGroup(op, 'Global')
        opg.add_option('--print-solvcon', action='store_true',
            dest='print_solvcon', default=False,
            help='Print the solvcon package in use.',
        )
        opg.add_option('--print-project-dir', action='store_true',
            dest='print_project_dir', default=False,
            help='Print the the project directory.',
        )
        op.add_option_group(opg)
        self.opg_global = opg

        # set to self.
        self.env = env
        self.op = op
        self._opargs = None

    @property
    def command_description(self):
        cmdstrs = sorted(cmdregy.keys())
        cmdmaxlen = max(len(c) for c in cmdstrs)
        idt1 = 2
        nsep = 1
        idt2 = idt1 + cmdmaxlen + nsep
        descriptions = []
        for cmdstr in cmdstrs:
            cmdcls = cmdregy[cmdstr]
            desc = cmdcls.__doc__.strip()
            desc = desc.split('\n')[0].strip()
            descriptions.append(''.join([' '*idt1,
                ('%%-%ds'%cmdmaxlen) % cmdstr, ' '*nsep, desc]))
        description = '\n'.join(descriptions)
        return '\n'.join(['Command:', description])

    @property
    def usage(self):
        return self._usage

    @property
    def opargs(self):
        import sys, os
        import solvcon
        from solvcon.helper import info
        # get the options and arguments.
        if not self._opargs:
            self.op.usage = self.usage
            ops, args = self.op.parse_args()
            narg = len(args)
            if narg >= 1:
                args = args[1:]
            else:
                args = []
            self._opargs = ops, args
            # include project path.
            flag_setproj = True
            for path in sys.path:
                if self.env.projdir == os.path.abspath(path):
                    flag_setproj = False
                    break
            if flag_setproj:
                sys.path.insert(0, self.env.projdir)
            # output general information.
            if ops.print_solvcon:
                info('*** Use the solvcon package located at "%s".\n' % \
                    solvcon.__file__)
            if ops.print_project_dir:
                info('*** Project is located at "%s".\n' % self.env.projdir)
        ops, args = self._opargs
        # test for number of arguments.
        if len(args) < self.min_args:
            info('Number of arguments is less than the sufficient: %d\n' % \
                self.min_args)
            sys.exit(0)
        # set self to env.
        self.env.command = self
        # return the options and arguments.
        return self._opargs

    def __call__(self):
        raise NotImplementedError

def go():
    """
    Command runner.
    """
    import sys
    from . import command
    from .conf import env, use_application
    for modname in env.modnames:
        if modname:
            use_application(modname)
    narg = len(sys.argv)
    if narg >= 2 and not sys.argv[1].startswith('-'):
        cmdcls = cmdregy.get(sys.argv[1], None)
    else:
        cmdcls = None
    if cmdcls == None:
        cmdcls = cmdregy['help']
    cmd = cmdcls(env)
    cmd()
