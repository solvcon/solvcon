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


__all__ = ['Host', 'AwsHost', 'AwsOperator', 'aoregy']


class Host(object):
    """
    Abstraction for actions toward a host.
    """

    @staticmethod
    def _prepare_command(cmd, sudo=False, env=None, wd=None):
        """
        This is the helper method to make up the command (*cmd*) with different
        operators.

        The optional argument *sudo* will prefix "sudo".  Default is False:

        >>> Host._prepare_command("cmd")
        'cmd'
        >>> Host._prepare_command("cmd", sudo=True)
        'sudo cmd'

        The optional argument *env* will prefix "env" with the given string of
        environment variables.  The default is None:

        >>> Host._prepare_command("cmd", env="PATH=$HOME:$PATH")
        'env PATH=$HOME:$PATH cmd'

        The optional argument *wd* will first change the working directory and
        execute the command.  The default is None:

        >>> Host._prepare_command("cmd", wd="/tmp")
        'cd /tmp; cmd'

        Argument *env* can be used with either *sudo* or *wd*:

        >>> Host._prepare_command("cmd", sudo=True,
        ...                          env="PATH=$HOME:$PATH")
        'sudo env PATH=$HOME:$PATH cmd'
        >>> Host._prepare_command("cmd", env="PATH=$HOME:$PATH", wd="/tmp")
        'cd /tmp; env PATH=$HOME:$PATH cmd'

        However, *sudo* doesn't work with *wd*:

        >>> Host._prepare_command("cmd", sudo=True, wd="/tmp")
        Traceback (most recent call last):
            super(self, AwsHost).__init__()
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

    def connect(self, username):
        pass

    def disconnect(self):
        pass

    def run(self, cmd, **kw):
        cmd = self._prepare_command(cmd, **kw)
        return cmd


class AwsHost(Host):
    """
    Abstraction for actions toward an AWS EC2 host.
    """

    def __init__(self, instance):
        super(AwsHost, self).__init__()
        #: :py:class:`boto.ec2.instance.Instance` for the host.
        self.instance = instance
        #: :py:class:`paramiko.client.SSHClient` to command the remote host.
        self.cli = None
        def default_keyfn_getter(keyname):
            keyfn = "aws_%s.pem" % keyname
            return os.path.join(os.environ['HOME'], ".ssh", keyfn)
        #: A callable takes a :py:class:`str` and convert it to a file name for
        #: the key.
        self.keyname2fn = default_keyfn_getter

    def connect(self, username):
        self.cli = ssh.client.SSHClient()
        self.cli.load_system_host_keys()
        self.cli.set_missing_host_key_policy(ssh.client.AutoAddPolicy())
        self.cli.connect(self.instance.public_dns_name,
                         username=username,
                         key_filename=self.keyname2fn(self.instance.key_name),
                         allow_agent=True)
        return self.cli

    def disconnect(self):
        self.cli.disconnect()
        self.cli = None

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


#: Tuple keys of :py:class:`AwsOperator`.
_AWSHOSTINFOKEYS = ("region", "ami", "osfamily", "username",
                    "instance_type", "security_groups")

class AwsOperator(collections.namedtuple("AwsOperator", _AWSHOSTINFOKEYS)):
    """
    Collection of AWS host information.

    Filling information into the class :py:class:`AwsOperator`.  The filled
    data will become read-only.

    >>> info = AwsOperator(region="us-west-2", ami="ami-77d7a747",
    ...                    osfamily="RHEL", username="ec2-user")
    >>> info # doctest: +NORMALIZE_WHITESPACE
    AwsOperator(region='us-west-2', ami='ami-77d7a747', osfamily='RHEL',
    username='ec2-user', instance_type='t2.micro', security_groups=('default',))
    >>> info.osfamily = "Debian"
    Traceback (most recent call last):
        ...
    AttributeError: can't set attribute

    Positional arguments aren't allowed:

    >>> info = AwsOperator("us-west-2", "ami-77d7a747", "RHEL", "ec2-user")
    Traceback (most recent call last):
        ...
    KeyError: "positional arguments aren't allowed"
    """

    #: Allowed OS families.
    _OSFAMILIES = ("RHEL", "Ubuntu", "Debian")

    #: Mapping to the package metadata updating command for each of the OS
    #: families.
    _PMETACMD = {
        "RHEL": "yum makecache -y",
        "Ubuntu": "apt-get update -y",
        "Debian": "apt-get update -y",
    }

    #: Mapping to the package installation command for each of the OS families.
    _PINSTCMD = {
        "RHEL": "yum install -y",
        "Ubuntu": "apt-get install -y",
        "Debian": "apt-get install -y",
    }

    _MINPKGS_COMMON = (
        "vim ctags wget screen bzip2 patch mercurial git gcc gfortran".split())
    _MINPKGS_DEBBUILD = ("build-essential zlib1g "
                         "liblapack-dev liblapack-pic".split())
    #: Minimal packages.
    _MINPKGS = {
        "RHEL": _MINPKGS_COMMON,
        "Ubuntu": _MINPKGS_COMMON + _MINPKGS_DEBBUILD,
        "Debian": _MINPKGS_COMMON + _MINPKGS_DEBBUILD,
    }

    #: The downloading URL for conda installer.
    MINICONDA_URL = ("http://repo.continuum.io/miniconda/"
                     "Miniconda-3.5.5-Linux-x86_64.sh")

    #: Where to find conda on the destination box.
    MINICONDA_PATH = "$HOME/opt/miniconda/bin"

    #: Where to find SOLVCON on the destination box.
    SOLVCON_PATH = "$HOME/sc/solvcon"

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
        obj = super(AwsOperator, cls).__new__(cls, *args)
        #: The commanding :py:class:`Host` object.
        obj.host = Host()
        # Return the object.
        return obj

    def connect(self, *args, **kw):
        return self.host.connect(self.username, *args, **kw)

    def disconnect(self, *args, **kw):
        return self.host.disconnect(*args, **kw)

    def run(self, *args, **kw):
        """
        >>> info = AwsOperator(region="us-west-2", ami="ami-77d7a747",
        ...                    osfamily="RHEL", username="ec2-user")
        >>> info.run("command")
        'command'
        """
        return self.host.run(*args, **kw)

    @property
    def pmetacmd(self):
        return self._PMETACMD[self.osfamily]

    @property
    def pinstcmd(self):
        return self._PINSTCMD[self.osfamily]

    @property
    def minpkgs(self):
        return self._MINPKGS[self.osfamily]

    def update_package_metadata(self):
        self.run(self.pmetacmd, sudo=True)

    def install(self, packages):
        manager = self.pinstcmd
        if not isinstance(packages, basestring):
            packages = " ".join(packages)
        self.run("%s %s" % (manager, packages), sudo=True)

    def hgclone(self, path, sshcmd="", ignore_key=False, **kw):
        cmd = "hg clone"
        if ignore_key:
            if not sshcmd:
                sshcmd = "ssh -oStrictHostKeyChecking=no"
            else:
                raise ValueError("ignore_key can't be used with sshcmd")
        if path.startswith("ssh") and sshcmd:
            cmd += " --ssh '%s'" % sshcmd
        cmd = "%s %s" % (cmd, path)
        self.run(cmd, **kw)

    def deploy_minimal(self):
        # Use OS package manager to install tools.
        self.update_package_metadata()
        self.install(self.minpkgs)
        # Install miniconda.
        mcurl = self.MINICONDA_URL
        mcfn = mcurl.split("/")[-1]
        run = functools.partial(self.run, wd="/tmp")
        run("rm -f %s" % mcfn)
        run("wget %s" % mcurl)
        run("bash %s -p $HOME/opt/miniconda -b" % mcfn)
        # Update conda packages.
        run = functools.partial(
            self.run, env="PATH=%s:$PATH" % self.MINICONDA_PATH)
        run("conda update --all --yes")
        # Install basic development tools with conda.
        run("conda install jinja2 binstar conda-build grin --yes")
        # Install standard dependencies with conda.
        run("conda install setuptools mercurial "
            "scons cython numpy netcdf4 vtk nose sphinx paramiko boto "
            "--yes")
        # Install customized dependencies with conda.
        run("conda install gmsh graphviz scotch --yes "
            "-c https://conda.binstar.org/yungyuc/channel/solvcon")

    def obtain_solvcon(self):
        # Clone the remote repository.
        self.run("mkdir -p $HOME/sc")
        self.hgclone("http://bitbucket.org/solvcon/solvcon", wd="$HOME/sc")

    def set_config_files(self):
        # Write conda channel settings.
        condarc = ("channels: [ "
                   "\"https://conda.binstar.org/yungyuc/channel/solvcon\", "
                   "defaults ]")
        self.run("echo '%s' > $HOME/.condarc" % condarc)
        # Back up bashrc.
        self.run("cp $HOME/.bashrc /tmp")
        # Write conda path to bashrc.
        self.run("echo 'if ! echo $PATH | egrep -q \"(^|:)%s($|:)\" ; "
                 "then export PATH=%s:$PATH ; fi' > $HOME/.bashrc" %
                 (self.MINICONDA_PATH, self.MINICONDA_PATH))
        # Write SOLVCON settings to bashrc.
        self.run("echo 'export SCSRC=%s' >> $HOME/.bashrc" %
                 self.SOLVCON_PATH)
        self.run("echo 'export PYTHONPATH=$SCSRC' >> $HOME/.bashrc")
        self.run("echo 'if ! echo $PATH | egrep -q \"(^|:)%s($|:)\" ; "
                 "then export PATH=%s:$PATH ; fi' >> $HOME/.bashrc" %
                 ("$SCSRC", "$SCSRC"))
        # Copy back the backed up bashrc.
        self.run("cat /tmp/.bashrc >> $HOME/.bashrc; rm -f /tmp/.bashrc")

    def build_solvcon(self):
        self.run("scons scmods", wd="$SCSRC")
        self.run("nosetests", wd="$SCSRC")


class AwsOperatorRegistry(dict):
    @classmethod
    def populate(cls):
        regy = cls()
        regy["RHEL64"] = AwsOperator(
            region="us-west-2", ami="ami-77d7a747",
            osfamily="RHEL", username="ec2-user")
        regy["trusty64"] = AwsOperator(
            region="us-west-2", ami="ami-d34032e3",
            osfamily="Ubuntu", username="ubuntu")
        regy[""] = regy["trusty64"]
        return regy

aoregy = AwsOperatorRegistry.populate()
