# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014, Yung-Yu Chen <yyc@solvcon.net>
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
Cloud controlling tools.
"""


import os

import boto.ec2
import paramiko as ssh

import helper


__all__ = ['AwsHost']


class OperatingSystem(object):
    FAMILIES = ("RHEL", "Suse", "Debian", "Ubuntu")

    def __init__(self, family):
        assert family in self.FAMILIES
        self.family = family

    @property
    def package_install_command(self):
        if "RHEL" == self.family:
            return "yum install -y"


class AwsHost(object):
    """
    Abstraction for actions toward an AWS EC2 host.
    """

    def __init__(self, instance, username, osfamily):
        #: :py:class:`boto.ec2.instance.Instance` for the host.
        self.instance = instance
        #: :py:class:`str` for the user name to the host.
        self.username = username
        #: :py:class:`OperatingSystem`.
        self.opsys = OperatingSystem(osfamily)
        #: :py:class:`paramiko.client.SSHClient` to command the remote host.
        self.cli = None
        def default_keyfn_getter(keyname):
            keyfn = "aws_%s.pem" % keyname
            return os.path.join(os.environ['HOME'], ".ssh", keyfn)
        #: A callable takes a :py:class:`str` and convert it to a file name for
        #: the key.
        self.keyname2fn = default_keyfn_getter

    def connect(self):
        self.cli = ssh.client.SSHClient()
        self.cli.load_system_host_keys()
        self.cli.set_missing_host_key_policy(ssh.client.AutoAddPolicy())
        self.cli.connect(self.instance.public_dns_name,
                         username=self.username,
                         key_filename=self.keyname2fn(self.instance.key_name),
                         allow_agent=True)
        return self.cli

    def disconnect(self):
        self.cli.disconnect()
        self.cli = None

    def run(self, cmd, sudo=False, wd=None, bufsize=-1, timeout=None):
        if wd is not None:
            cmd = "cd %s; %s" % (wd, cmd)
        helper.info(cmd + "\n")
        chan = self.cli.get_transport().open_session()
        forward = ssh.agent.AgentRequestHandler(chan)
        chan.get_pty()
        chan.set_combine_stderr(True)
        chan.settimeout(timeout)
        if sudo:
            cmd = "sudo %s" % cmd
        chan.exec_command(cmd)
        stdin = chan.makefile('wb', bufsize)
        stdout = chan.makefile('r', bufsize)
        data = stdout.read()
        helper.info(data + "\n")
        return data

    def install(self, packages):
        manager = self.opsys.package_install_command
        if not isinstance(packages, basestring):
            packages = " ".join(packages)
        self.run("%s %s" % (manager, packages), sudo=True)

    def hgclone(self, path, sshcmd="", **kw):
        cmd = "hg clone"
        if path.startswith("ssh") and sshcmd:
            cmd += " --ssh '%s'" % sshcmd
        cmd = "%s %s" % (cmd, path)
        self.run(cmd, **kw)
