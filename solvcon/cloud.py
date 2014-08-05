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
import collections
import functools

import boto.ec2
import paramiko as ssh

import helper


__all__ = ['AwsHost', 'ahsregy']


class OperatingSystem(object):
    FAMILIES = ("RHEL", "Suse", "Debian", "Ubuntu")

    PACKAGE_INSTALL_COMMAND = {
        "RHEL": "yum install -y",
        "Debian": "apt-get install -y",
        "Ubuntu": "apt-get install -y",
    }

    def __init__(self, family):
        assert family in self.FAMILIES
        self.family = family
        self.package_install_command = self.PACKAGE_INSTALL_COMMAND[family]


#: Tuple keys of :py:class:`AwsHostSetting`.
_AWSHOSTINFOKEYS = ("region", "ami", "osfamily", "username",
                    "instance_type", "security_groups")

class AwsHostSetting(collections.namedtuple("AwsHostSetting", _AWSHOSTINFOKEYS)):
    """
    Collection of AWS host information.

    Filling information into the class :py:class:`AwsHostSetting`.  The filled
    data will become read-only.

    >>> info = AwsHostSetting(region="us-west-2", ami="ami-77d7a747",
    ...                       osfamily="RHEL", username="ec2-user")
    >>> info # doctest: +NORMALIZE_WHITESPACE
    AwsHostSetting(region='us-west-2', ami='ami-77d7a747', osfamily='RHEL',
    username='ec2-user', instance_type='t2.micro', security_groups=('default',))
    >>> info.osfamily = "Debian"
    Traceback (most recent call last):
        ...
    AttributeError: can't set attribute

    Positional arguments aren't allowed:

    >>> info = AwsHostSetting("us-west-2", "ami-77d7a747", "RHEL", "ec2-user")
    Traceback (most recent call last):
        ...
    KeyError: "positional arguments aren't allowed"
    """

    #: Allowed OS families.
    _OSFAMILIES = ("RHEL", "Ubuntu", "Debian")

    #: Mapping to the package installation command for each of the OS families.
    _PINSTCMD = {
        "RHEL": "yum install -y",
        "Ubuntu": "apt-get install -y",
        "Debian": "apt-get install -y",
    }

    #: Minimal packages.
    _MINPKGS = {
        "RHEL": "mercurial git vim ctags wget screen bzip2 patch".split(),
        "Ubuntu": "mercurial git vim ctags wget screen bzip2 patch".split(),
        "Debian": "mercurial git vim ctags wget screen bzip2 patch".split(),
    }

    def __new__(cls, *args, **kw):
        # Disallow positional arguments.
        if args:
            raise KeyError("positional arguments aren't allowed")
        # Sanitize osfamily.
        osfamily = kw['osfamily']
        if osfamily not in cls._OSFAMILIES:
            fam = ", ".join("\"%s\"" % fam for fam in cls._OSFAMILIES)
            raise ValueError("osfamily \"%s\" not in %s" % (osfamily, fam))
        # Set default values.
        kw.setdefault("instance_type", "t2.micro")
        kw.setdefault("security_groups", ("default",))
        # Make up arguments.
        args = tuple(kw[key] for key in cls._fields)
        # Create the object.
        obj = super(AwsHostSetting, cls).__new__(cls, *args)
        # Return the object.
        return obj

    @property
    def pinstcmd(self):
        return self._PINSTCMD[self.osfamily]

    @property
    def minpkgs(self):
        return self._MINPKGS[self.osfamily]


class AwsHostSettingRegistry(dict):
    @classmethod
    def populate(cls):
        regy = cls()
        regy["RHEL64"] = AwsHostSetting(
            region="us-west-2", ami="ami-77d7a747",
            osfamily="RHEL", username="ec2-user")
        regy["trusty64"] = AwsHostSetting(
            region="us-west-2", ami="ami-d34032e3",
            osfamily="Ubuntu", username="ubuntu")
        regy[""] = regy["trusty64"]
        return regy

ahsregy = AwsHostSettingRegistry.populate()


class AwsHost(object):
    """
    Abstraction for actions toward an AWS EC2 host.
    """

    MINICONDA_URL = ("http://repo.continuum.io/miniconda/"
                     "Miniconda-3.5.5-Linux-x86_64.sh")

    def __init__(self, instance, setting):
        #: :py:class:`boto.ec2.instance.Instance` for the host.
        self.instance = instance
        #: :py:class:`AwsHostSetting` for read-only AWS EC2 settings.
        self.setting = setting
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
                         username=self.setting.username,
                         key_filename=self.keyname2fn(self.instance.key_name),
                         allow_agent=True)
        return self.cli

    def disconnect(self):
        self.cli.disconnect()
        self.cli = None

    @staticmethod
    def _prepare_command(cmd, sudo=False, env=None, wd=None):
        """
        This is the helper method to make up the command (*cmd*) with different
        settings.

        The optional argument *sudo* will prefix "sudo".  Default is False:

        >>> AwsHost._prepare_command("cmd")
        'cmd'
        >>> AwsHost._prepare_command("cmd", sudo=True)
        'sudo cmd'

        The optional argument *env* will prefix "env" with the given string of
        environment variables.  The default is None:

        >>> AwsHost._prepare_command("cmd", env="PATH=$HOME:$PATH")
        'env PATH=$HOME:$PATH cmd'

        The optional argument *wd* will first change the working directory and
        execute the command.  The default is None:

        >>> AwsHost._prepare_command("cmd", wd="/tmp")
        'cd /tmp; cmd'

        Argument *env* can be used with either *sudo* or *wd*:

        >>> AwsHost._prepare_command("cmd", sudo=True,
        ...                          env="PATH=$HOME:$PATH")
        'sudo env PATH=$HOME:$PATH cmd'
        >>> AwsHost._prepare_command("cmd", env="PATH=$HOME:$PATH", wd="/tmp")
        'cd /tmp; env PATH=$HOME:$PATH cmd'

        However, *sudo* doesn't work with *wd*:

        >>> AwsHost._prepare_command("cmd", sudo=True, wd="/tmp")
        Traceback (most recent call last):
            ...
        ValueError: sudo can't be True with wd set
        """
        if env is not None:
            cmd = "env %s %s" % (env, cmd)
        if wd is not None:
            cmd = "cd %s; %s" % (wd, cmd)
            if sudo:
                raise ValueError("sudo can't be True with wd set")
        if sudo:
            cmd = "sudo %s" % cmd
        return cmd

    @staticmethod
    def _setup_channel(chan, bufsize=-1, timeout=None):
        chan = self.cli.get_transport().open_session()
        forward = ssh.agent.AgentRequestHandler(chan)
        chan.get_pty()
        chan.set_combine_stderr(True)
        chan.settimeout(timeout)
        chan.exec_command(cmd)
        stdin = chan.makefile('wb', bufsize)
        stdout = chan.makefile('r', bufsize)
        data = stdout.read()

    def run(self, cmd, bufsize=-1, timeout=None, **kw):
        # Prepare the command.
        cmd = self._prepare_command(cmd, **kw)
        # Log command information.
        helper.info(cmd + "\n")
        # Open the channel.
        chan = self.cli.get_transport().open_session()
        # Set up SSH authentication agent forwarding.
        forward = ssh.agent.AgentRequestHandler(chan)
        # Get and set up the terminal.
        chan.get_pty()
        chan.set_combine_stderr(True)
        chan.settimeout(timeout)
        # Send the command.
        chan.exec_command(cmd)
        # Use the STD I/O.
        stdin = chan.makefile('wb', bufsize)
        stdout = chan.makefile('r', bufsize)
        # Get the data and report.
        data = stdout.read()
        helper.info(data + "\n")
        return data

    def install(self, packages):
        manager = self.setting.pinstcmd
        if not isinstance(packages, basestring):
            packages = " ".join(packages)
        self.run("%s %s" % (manager, packages), sudo=True)

    def hgclone(self, path, sshcmd="", **kw):
        cmd = "hg clone"
        if path.startswith("ssh") and sshcmd:
            cmd += " --ssh '%s'" % sshcmd
        cmd = "%s %s" % (cmd, path)
        self.run(cmd, **kw)

    def deploy_minimal(self):
        # Use OS package manager to install tools.
        self.install(self.setting.minpkgs)
        # Install miniconda.
        mcurl = self.MINICONDA_URL
        mcfn = mcurl.split("/")[-1]
        run = functools.partial(self.run, wd="/tmp")
        run("rm -f %s" % mcfn)
        run("wget %s" % mcurl)
        run("bash %s -p ~/opt/miniconda -b" % mcfn)
        # Update and install conda packages.
        run = functools.partial(self.run,
                                env="PATH=$HOME/opt/miniconda/bin:$PATH")
        run("conda update --all --yes")
        run("conda install jinja2 --yes")
        run("conda install setuptools mercurial conda-build "
            "scons cython numpy netcdf4 nose sphinx paramiko boto "
            "--yes")
