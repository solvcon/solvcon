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
Basic support for cluster batch systems.

* batregy: The registry containing all usable batch system abstraction.
* Batch: The fundamental class for batch systems.
* Localhost: A dummy batch systems for domain decomposition working on
    localhost.
* Torque: The Torque batch system.
"""

from .gendata import TypeNameRegistry

class Node(object):
    def __init__(self, name, ncore=1, serial=None, attrs=None):
        self.name = name
        self.ncore = ncore
        self.serial = serial
        self.pserial = serial
        self.attrs = attrs if attrs != None else list()

    def __str__(self):
        attrstr = ','.join(self.attrs)
        return '{%s(%d)%s}' % (self.name, self.ncore,
            ': '+attrstr if attrstr else '')

    @property
    def address(self):
        from socket import gethostbyname
        return gethostbyname(self.name)

batregy = TypeNameRegistry() # registry singleton.
class BatchMeta(type):
    def __new__(cls, name, bases, namespace):
        newcls = super(BatchMeta, cls).__new__(cls, name, bases, namespace)
        # register.
        batregy.register(newcls)
        return newcls

class Batch(object):
    """
    Batch system submitter.

    @cvar _subcmd_: the command name for the batch submission.
    @ctype _subcmd_: str
    @cvar BASH_HOME_SOURCE: the files to be sourced in user's home.
    @ctype BASH_HOME_SOURCE: list
    @ivar case: The Case corresponding object.
    @itype case: solvcon.case.core.Case
    @ivar arnname: The name of the arrangement to be run.
    @itype arnname: str
    @ivar jobname: The name to send to batch system.
    @itype jobname: str
    @ivar jobdir: The absolute path for job.
    @itype jobdir: str
    @ivar output: Batch output type.
    @itype output: str
    @ivar shell: Shell to be used on the cluster.
    @itype shell: str
    @ivar use_mpi: Indicate to use MPI as transport layer.
    @itype use_mpi: bool
    @ivar resource: Specified resources.
    @itype resource: dict
    """

    __metaclass__ = BatchMeta

    _subcmd_ = 'qsub'

    DEFAULT_OUTPUT = 'oe'
    DEFAULT_SHELL = '/bin/sh'
    BASH_HOME_SOURCE = []
    
    def __init__(self, case, **kw):
        """
        @keyword rootdir: Root directory for the project/code.
        @type rootdir: str
        @keyword arnname: The arrangement to be run.
        @type arnname: str
        """
        import os
        self.case = case
        self.arnname = kw.pop('arnname', None)
        self.jobname = kw.pop('jobname', self.arnname)
        if case.io.basedir == None:
            if self.jobname != None:
                self.jobdir = os.path.abspath(os.path.join(
                    self.case.io.rootdir, self.jobname))
            else:
                self.jobdir = None
        else:
            self.jobdir = case.io.basedir
        self.output = kw.pop('output', self.DEFAULT_OUTPUT)
        self.shell = kw.pop('shell', self.DEFAULT_SHELL)
        self.use_mpi = kw.pop('use_mpi', False)
        self.resource = case.execution.resources.copy()
        self.resource.update(kw)
        super(Batch, self).__init__()

    @property
    def str_header(self):
        return '#!/bin/sh'

    @property
    def str_resource(self):
        raise NotImplementedError

    @property
    def str_jobname(self):
        raise NotImplementedError

    @property
    def str_output(self):
        raise NotImplementedError

    @property
    def str_shell(self):
        raise NotImplementedError

    @property
    def str_path(self):
        import os
        ret = list()
        ret.append('echo "Customized paths for job:"')
        # source bash rc files per user.
        home = os.environ['HOME']
        for fn in self.BASH_HOME_SOURCE:
            if os.path.exists(os.path.join(home, fn)):
                ret.append('. $HOME/%s'%fn)
        ret.append('export PYTHONPATH=%s:$PYTHONPATH' % self.case.io.rootdir)
        return '\n'.join(ret)

    @property
    def str_prerun(self):
        msgs = list()
        envar = self.case.solver.envar
        if envar != None:
            for key in envar:
                msgs.append('export %s=%s' % (key, envar[key]))
        msgs.append('echo "Run @`date`:"')
        return '\n'.join(msgs)

    @property
    def str_postrun(self):
        return 'echo "Finish @`date`."'

    def build_scg_command(self):
        import os
        from .conf import env
        scgargs = ' '.join(['run', self.arnname])
        if env.command != None:
            ops, args = env.command.opargs
            scgops = list()
            if ops.npart != None:
                scgops.append('\\\n')
                scgops.append('--npart=%d' % ops.npart)
                scgops.append('--batch=%s' % ops.batch)
            if ops.envar:
                scgops.append('\\\n')
                envar = env.command.envar
                scgops.append('--envar %s' % ':'.join([
                    '%s=%s' % (key, envar[key]) for key in envar
                ]))
            if ops.compress_nodelist:
                scgops.append('--compress-nodelist')
            if ops.use_profiler:
                scgops.append('\\\n')
                scgops.append('--use-profiler')
                scgops.extend([
                    '--profiler-sort=%s' % ops.profiler_sort,
                    '--profiler-dat=%s' % ops.profiler_dat,
                    '--profiler-log=%s' % ops.profiler_log,
                ])
            if ops.solver_output:
                scgops.append('--solver-output')
            if ops.basedir:
                scgops.append('\\\n')
                scgops.append('--basedir=%s' % os.path.abspath(ops.basedir))
            scgops = ' '.join(scgops)
        else:
            scgops = ''
        scgops = '--runlevel %%d %s' % scgops
        return ' '.join([env.get_entry_point(), scgargs, scgops])

    def build_mpi_runner(self):
        return ''

    @property
    def str_run(self):
        cmds = ['time']
        if self.use_mpi:
            mpi_runner = self.build_mpi_runner()
            if not mpi_runner:
                raise RuntimeError(
                    '%s gave null mpi_runner' % str(self.__class__))
            cmds.append(mpi_runner)
        cmds.append(self.build_scg_command())
        return '\n'.join([
            'cd %s' % self.jobdir,
            ' '.join(cmds).strip()
        ])

    def __str__(self):
        """
        Collect all string properties.
        """
        return '\n'.join([
            self.str_header,
            self.str_resource,
            self.str_jobname,
            self.str_output,
            self.str_shell,
            self.str_path,
            self.str_prerun,
            self.str_run,
            self.str_postrun,
        ])

    def tofile(self, basename=None):
        """
        Write self into the file for the submitting script.
        """
        import os
        from glob import glob
        info = self.case.info
        basename = self.jobname+'.pbs' if basename == None else basename
        if os.path.exists(self.jobdir):
            info('Job directory was there: %s\n' % self.jobdir)
            if self.case.io.empty_jobdir:
                info('Delete all file in job directory.\n')
                for fn in glob(os.path.join(self.jobdir, '*')):
                    os.unlink(fn)
        else:
            os.makedirs(self.jobdir)
        fn = os.path.abspath(os.path.join(self.jobdir, basename))
        fnlist = list()
        for ilevel in range(3):
            fnlist.append(fn+str(ilevel))
            f = open(fnlist[-1], 'w')
            f.write(str(self) % ilevel)
            f.close()
        return fnlist

    def __call__(self, runlevel=0, basename=None, postpone=False):
        """
        Make submitting script and invoke the batch system.
        """
        import os
        from subprocess import call
        info = self.case.info
        fnlist = self.tofile(basename=basename)
        os.chdir(self.jobdir)
        if postpone:
            return
        else:
            info('submit runlevel %d\n' % runlevel)
            return call('%s %s'%(self._subcmd_, fnlist[runlevel]), shell=True)

    def nodelist(self):
        raise NotImplementedError

    def create_worker(self, *args, **kw):
        """
        True implementations are in create_worker_*() and have identical spec.
        """
        raise NotImplementedError

    def create_worker_ssh(self, node, authkey,
            envar=None, paths=None, profiler_data=None):
        """
        Use secure shell to create worker object.

        @param node: node information.
        @type node: Node
        @param authkey: the authkey for the worker.
        @type authkey: str
        @keyword envar: additional environment variables to remote.
        @type envar: dict
        @keyword paths: path for remote execution.
        @type paths: dict
        @keyword profiler_data: profiler setting for remote worker.
        @type profiler_data: tuple
        @return: the port that the remote worker listen on.
        @rtype: int
        """
        import os
        from subprocess import PIPE
        from .connection import Client  # XXX: no need.
        from .rpc import SecureShell
        remote = SecureShell(node.address, paths=paths)
        # determine remotely available port.
        val = int(remote([
            'import sys',
            'from solvcon.connection import pick_unused_port',
            'sys.stdout.write(str(pick_unused_port()))',
        ], stdout=PIPE))
        try:
            port = int(val)
        except ValueError:
            raise IOError, 'remote port detection fails'
        # create remote worker objects and return.
        pdata = str(profiler_data).replace("'", '"')
        remote([
            'import os',
            'os.chdir("%s")' % os.path.abspath(os.getcwd()),
            'from solvcon.rpc import Worker',
            'wkr = Worker(None, profiler_data=%s)' % pdata,
            'wkr.run(("%s", %d), "%s")' % (node.address, port, authkey),
        ], envar=envar)
        return port

    def create_worker_mpi(self, node, authkey,
            envar=None, paths=None, profiler_data=None):
        """
        Use MPI to link remote worker object.

        @param node: node information.
        @type node: Node
        @param authkey: the authkey for the worker.
        @type authkey: str
        @keyword envar: additional environment variables to remote.
        @type envar: dict
        @keyword paths: path for remote execution.
        @type paths: dict
        @keyword profiler_data: profiler setting for remote worker.
        @type profiler_data: tuple
        @return: the port that the remote worker listen on.
        @rtype: int
        """
        from .conf import env
        return env.mpi.recv(node.serial, 1)

class Localhost(Batch):
    """
    Dummy batch abstraction for localhost.
    """
    def nodelist(self):
        return [Node('127.0.0.1', ncore=1, serial=i)
            for i in range(self.case.execution.npart)]
    def create_worker(self, *args, **kw):
        return self.create_worker_ssh(*args, **kw)

class Torque(Batch):
    """
    Torque/OpenPBS.
    """

    def __init__(self, case, **kw):
        super(Torque, self).__init__(case, **kw)
        self._nodelist = None

    @property
    def str_resource(self):
        res = self.resource.copy()
        if self.case.execution.npart != None:
            res['nodes'] = self.case.execution.npart
            if self.use_mpi:
                res['nodes'] += 1
        # build resource tokens.
        tokens = list()
        for key in sorted(res.keys()):
            val = res[key]
            if val == None:
                token = key
            else:
                token = '%s=%s' % (key, val)
            if token:
                tokens.append(token)
        # nodes and ppn must be together.
        idx1 = 0
        while idx1 < len(tokens):
            if 'nodes' in tokens[idx1]:
                break
            idx1 += 1
        idx2 = 0
        while idx2 < len(tokens):
            if 'ppn' in tokens[idx2]:
                break
            idx2 += 1
        if idx1 != len(tokens) and idx2 != len(tokens) and idx1 != idx2:
            tokens[idx1] = ':'.join([tokens[idx1], tokens[idx2]])
            del tokens[idx2]
        # return resource string.
        if tokens:
            return '#PBS -l %s' % ','.join(tokens)
        else:
            return ''

    @property
    def str_jobname(self):
        return '#PBS -N %s' % self.jobname

    @property
    def str_output(self):
        return '#PBS -j %s' % self.output

    @property
    def str_shell(self):
        return '#PBS -S %s' % self.shell

    def nodelist(self):
        import os
        from .conf import env
        if not self._nodelist:
            # read node file.
            f = open(os.environ['PBS_NODEFILE'])
            entries = [item.strip() for item in f.readlines()]
            f.close()
            nodelist = [Node(entries[it], ncore=1, serial=it) for it in
                range(len(entries))]
            # compress nodelist.
            if env.command != None:
                ops, args = env.command.opargs
                if ops.compress_nodelist:
                    cnodelist = [nodelist[0]]
                    for nodeitem in nodelist[1:]:
                        cnodeitem = cnodelist[-1]
                        if nodeitem.address == cnodeitem.address:
                            cnodeitem.ncore += 1
                        else:
                            nodeitem.serial = len(cnodelist)
                            cnodelist.append(nodeitem)
                    nodelist = cnodelist
            # exclude head when using MPI.
            if env.mpi:
                nodelist = nodelist[1:]
            # cut nodelist.
            self._nodelist = nodelist[:self.case.execution.npart]
        return self._nodelist

    def create_worker_torque(self, node, authkey,
            envar=None, paths=None, profiler_data=None):
        """
        Use Torque TM API to create worker object.

        @param node: node information.
        @type node: Node
        @param authkey: the authkey for the worker.
        @type authkey: str
        @keyword envar: additional environment variables to remote.
        @type envar: dict
        @keyword paths: path for remote execution.
        @type paths: dict
        @keyword profiler_data: profiler setting for remote worker.
        @type profiler_data: tuple
        @return: the port that the remote worker listen on.
        @rtype: int
        """
        import sys, os
        from threading import Thread
        from Queue import Queue
        from .batch_torque import TaskManager
        from .connection import pick_unused_port
        # setup listener for remote port.
        myhost = self.nodelist()[0].address
        myport = pick_unused_port()
        portq = Queue()
        def get_port():
            from .connection import Listener
            lsnr = Listener((myhost, myport), authkey=authkey)
            conn = lsnr.accept()
            portq.put(conn.recv())
            conn.close()
        thd = Thread(target=get_port)
        thd.start()
        # start remote worker.
        tm = TaskManager(paths=paths)
        tm.spawn(sys.executable, '-c',
            "from solvcon.batch_torque import run_worker; "
            "run_worker(%s, %s, %s, %s, %s)" % (
                "('%s', %d)" % (myhost, myport),
                "'%s'" % os.getcwd(),
                str(profiler_data),
                "'%s'" % node.address,
                "'%s'" % authkey,
            ), where=node.pserial, envar=envar)
        # stop listening thread.
        thd.join()
        return portq.get()

    def create_worker(self, *args, **kw):
        from .conf import env
        from .batch_torque import TaskManager
        if env.mpi:
            return self.create_worker_mpi(*args, **kw)
        elif TaskManager._clib_torque:
            return self.create_worker_torque(*args, **kw)
        else:
            return self.create_worker_ssh(*args, **kw)

class Generic(Batch):
    """
    A generic batch system, should be compatible with all batch systems.
    Requires environmental variable 'SOLVCON_NODELIST' to load the correct
    nodes to be used for the parallel run.
    """

    def __init__(self, case, **kw):
        super(Generic, self).__init__(case, **kw)
        self._nodelist = None

    def nodelist(self):
        import os
        from .conf import env
        if not self._nodelist:
            # read node file.
            solvcon_nodelist= str(  os.environ['SOLVCON_NODELIST']  )
            f = open( solvcon_nodelist )
            entries = [item.strip() for item in f.readlines()]
            f.close()
            nodelist = [Node(entries[it], ncore=1, serial=it) for it in
                range(len(entries))]
            # compress nodelist.
            if env.command != None:
                ops, args = env.command.opargs
                if ops.compress_nodelist:
                    cnodelist = [nodelist[0]]
                    for nodeitem in nodelist[1:]:
                        cnodeitem = cnodelist[-1]
                        if nodeitem.address == cnodeitem.address:
                            cnodeitem.ncore += 1
                        else:
                            nodeitem.serial = len(cnodelist)
                            cnodelist.append(nodeitem)
                    nodelist = cnodelist
            # exclude head when using MPI.
            if env.mpi:
                nodelist = nodelist[1:]
            # cut nodelist.
            self._nodelist = nodelist[:self.case.execution.npart]
        return self._nodelist

    def create_worker(self, *args, **kw):
        from .conf import env
        if env.mpi:
            return self.create_worker_mpi(*args, **kw)
        else:
            return self.create_worker_ssh(*args, **kw)
