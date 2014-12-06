# -*- coding: UTF-8 -*-
#
# Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Supporting functionalities and structures for UI commands.
"""

from .gendata import TypeNameRegistry

class CommandRegistry(TypeNameRegistry):
    def register(self, cmdtype):
        name = cmdtype.__name__
        if name.islower():
            return super(CommandRegistry, self).register(cmdtype)
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

def test():
    test.__test__ = False
    import os
    import nose
    path = os.path.dirname(__file__)
    nose.run(defaultTest=path)

def go():
    """
    Command runner.
    """
    import sys
    from . import command
    from .conf import env
    env.enable_applications()
    narg = len(sys.argv)
    if narg >= 2 and not sys.argv[1].startswith('-'):
        cmdcls = cmdregy.get(sys.argv[1], None)
    else:
        cmdcls = None
    if cmdcls == None:
        cmdcls = cmdregy['help']
    cmd = cmdcls(env)
    cmd()
