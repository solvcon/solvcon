# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Remote procedure call and inter-process communication.
"""

DEFAULT_AUTHKEY = 'solvcon.rpc'
DEFAULT_SLEEP = 0.1

def pick_unused_port():
    """
    Use socket to find out a unused (inet) port.
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    addr, port = s.getsockname()
    s.close()
    return port

def guess_address(family, localhost=True):
    """
    Guess a unused address according to given family.

    @param family: AF_INET, AF_UNIX, AF_PIPE.
    @type family: str
    @keyword localhost: use 'localhost' as hostname or not.
    @type localhost: bool
    """
    from socket import gethostname
    from random import sample
    string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    strlen = 8
    port = pick_unused_port()
    if family == 'AF_INET':
        if localhost:
            hostname = 'localhost'
        else:
            hostname = gethostname()
        address = (hostname, port)
    elif family == 'AF_UNIX':
        strpart = ''.join(sample(string, strlen))
        address = '/tmp/srpc%s%d' % (strpart, port)
    elif family == 'AF_PIPE':
        strpart = ''.join(sample(string, strlen))
        address = r'\\.\pipe\srpc' + "%s%d"%(strpart, port)
    else:
        raise ValueError, "family can't be %s" % family
    return address

class Terminate(Exception):
    """
    Signaling termination of Worker event loop.
    """
    pass

class Notice(object):
    """
    Base class for notification for ipc.
    """
    pass
class Barrier(Notice):
    pass
class Command(Notice):
    """
    The command to remote process.  The base class is for all kinds of commands
    designated to Muscle object.
    """
    def __init__(self, methodname, *args, **kw):
        self.with_worker = kw.pop('with_worker', False)
        self.methodname = methodname
        self.args = args
        self.kw = kw
class Control(Command):
    """
    Special commands to Worker object.
    """
    pass

class Worker(object):
    """
    The whole worker object will run remotely (means in separated process).

    @ivar muscle: the muscle object.
    @itype muscle: Muscle
    @ivar serial: serial number of the worker process.
    @itype serial: int
    @ivar lsnr: the listener object for master.
    @itype lsnr: solvcon.connection.Listener
    @ivar conn: the connection object to master.
    @itype conn: solvcon.connection.Client
    @ivar plsnrs: dictionary of listener objects for peers.
    @itype plsnrs: dict
    @ivar pconns: dictionary of connection objects to peers.
    @itype pconns: dict
    """
    def __init__(self, muscle, profiler_data=None):
        self.muscle = muscle
        self.serial = None
        self.lsnr = None
        self.conn = None
        self.plsnrs = dict()
        self.pconns = dict()
        self.do_profile = True if profiler_data else False
        self.profiler_dat = profiler_data[0] if profiler_data else None
        self.profiler_log = profiler_data[1] if profiler_data else None
        self.profiler_sort = profiler_data[2] if profiler_data else None

    def _eventloop(self):
        """
        Event loop.
        """
        while True:
            ntc = self.conn.recv()
            try:
                if isinstance(ntc, Command):
                    obj = self.muscle
                    if isinstance(ntc, Control):
                        obj = self
                    method = getattr(obj, ntc.methodname)
                    if ntc.with_worker:
                        ntc.kw.update(worker=self)
                    ret = method(*ntc.args, **ntc.kw)
            except Terminate:
                break

    def eventloop(self):
        import cProfile
        import pstats
        if self.do_profile:
            cProfile.runctx('self._eventloop()', globals(), locals(),
                self.profiler_dat)
            plog = open(self.profiler_log, 'w')
            p = pstats.Stats(self.profiler_dat, stream=plog)
            p.sort_stats(*self.profiler_sort.split(','))
            p.dump_stats(self.profiler_dat)
            p.print_stats()
            plog.close()
        else:
            self._eventloop()

    def register(self, address, authkey, *args, **kw):
        """
        Connect to remote listener and run event lop.  In this case, the worker
        don't have a valid listener since it acts as a client to the dealer.

        @param address: address to connect.
        @type address: str or tuple
        @param authkey: authentication key for connection.
        @type authkey: str
        """
        from time import sleep
        from .connection import Client
        # connect to the public address to the dealer.
        conn = Client(address=address, authkey=authkey)
        # get the actual/random/private address from dealer.
        address, authkey, wait_for_connect = conn.recv()
        # close the original public connection and wait for a period of time.
        conn.close()
        sleep(wait_for_connect)
        # make the private connection and save for self.
        self.conn = Client(address=address, authkey=authkey)
        # start eventloop.
        self.eventloop()

    def run(self, address, authkey, *args, **kw):
        """
        Listen to given address and run event loop.  In this case, the worker
        has the listener and acts as a server.

        @param address: address to listen.
        @type address: str or tuple
        @param authkey: authentication key for connection.
        @type authkey: str
        """
        from .connection import Listener
        # listen on the given address and accept connection.
        self.lsnr = Listener(address=address, authkey=authkey)
        self.conn = self.lsnr.accept()
        # start eventloop.
        self.eventloop()

    def chdir(self, dirname):
        import os
        os.chdir(dirname)

    def remote_setattr(self, name, var):
        """
        Remotely set attribute of worker.
        """
        return setattr(self, name, var)

    def remote_loadobj(self, name, objfn):
        """
        Remotely unpickle a file and set it to self with the specified name.
        """
        import cPickle as pickle
        setattr(self, name, pickle.load(open(objfn)))

    def barrier(self):
        """
        Send barrier signal for synchronization.
        """
        self.conn.send(Barrier)

    def accept_peer(self, peern, family, authkey):
        """
        Accept connection from specified peer.

        @param peern: index of the peer who wants to connect.
        @type peern: int
        @param family: the family of address needed to be guessed.
        @type family: str
        @param authkey: authentication key for connection.
        @type authkey: str
        """
        lsnr = self.lsnr
        self.conn.send(lsnr.address)
        # bind the address to set up a connection.
        conn = lsnr.accept()
        # after get connected, save the listener and connection.
        self.plsnrs[peern] = lsnr
        self.pconns[peern] = conn

    def connect_peer(self, peern, address, authkey):
        """
        Make a connection to specified peer (it has to be accepting
        connection).

        @param peern: index of the peer who wants to connect.
        @type peern: int
        @param address: the address to connect to.
        @type address: str or tuple
        @param authkey: authentication key for connection.
        @type authkey: str
        """
        from .connection import Client
        conn = Client(address=address, authkey=authkey)
        self.pconns[peern] = conn

    def terminate(self):
        raise Terminate

class Agent(object):
    """
    Remote agent to worker.

    @ivar connection: connection to the worker.
    @itype connection: solvcon.connection.Client
    @ivar noticetype: type of notice object to send.
    @itype noticetype: Notice
    """
    def __init__(self, connection=None, noticetype=Command):
        self.connection = connection
        self.noticetype = noticetype

    def __getattr__(self, name):
        conn = self.connection
        ntype = self.noticetype
        def func(*arg, **kw):
            conn.send(ntype(name, *arg, **kw))
        return func

class Shadow(object):
    """
    Convenient wrapper for two agents that send commands to remote worker and
    muscle.  The default agent is to the worker (ctl).

    @ivar listener: listener to worker.
    @itype listener: solvcon.connection.Listener
    @ivar connection: connection to the worker.
    @itype connection: solvcon.connection.Client
    @ivar address: remote address.
    @itype address: tuple or str
    @ivar cmd: agent to muscle.
    @itype cmd: Agent
    @ivar ctl: agent to worker.
    @itype ctl: Agent
    """
    def __init__(self, listener=None, connection=None, address=None):
        self.listener = listener
        self.connection = connection
        self.address = address
        self.cmd = Agent(connection=connection, noticetype=Command)
        self.ctl = Agent(connection=connection, noticetype=Control)

    def __getattr__(self, name):
        """
        Default to worker.
        """
        return getattr(self.ctl, name)

    def recv(self, *args, **kw):
        """
        Receive data from worker/muscle.
        """
        return self.connection.recv(*args, **kw)

class Dealer(list):
    """
    Contains shadows to workers.  Workers can be hired or recruited.  A hired
    worker is local to the dealer so that the dealer can directly start it.  A
    recruited worker is remote, and the a dealer can only wait for it to
    register.  A recruited worker is instantiated by itself, in a standalone
    process, and usually remotely.

    @ivar publicaddress: the public address for worker to gain connection
        information.  It is used for recruitment.
    @itype publicaddress: tuple or str
    @ivar authkey: authentication key for worker connection.
    @itype authkey: str
    @ivar family: connection family for automatically connection generation.
        Can be 'AF_PIPE', 'AF_UNIX', or 'AF_INET'.
    @itype family: str
    """
    WAIT_FOR_ACCEPT = 0.1

    def __init__(self, *args, **kw):
        import sys
        self.publicaddress = kw.pop('publicaddress', None)
        self.authkey = kw.pop('authkey', DEFAULT_AUTHKEY)
        self.family = kw.pop('family', None)
        if self.family == None:
            if sys.platform.startswith('win'):
                self.family = 'AF_PIPE'
            elif sys.platform.startswith('linux'):
                self.family = 'AF_UNIX'
            else:
                self.family = 'AF_INET'
        super(Dealer, self).__init__(*args, **kw)

    def hire(self, worker, inetaddr=None, wait_for_accept=None):
        """
        Create a process for a worker object.  The worker will be sent to
        the process.

        @param worker: worker object.
        @type worker: Worker
        @keyword wait_for_accept: seconds to wait after accepting.  If None use
            DEFAULT.
        @type wait_for_accept: float
        """
        from time import sleep
        from .connection import Process, Client
        # create and start the process.
        address = guess_address(self.family)
        proc = Process(
            target=worker.run,
            args=(address, self.authkey),
        )
        proc.start()
        sleep(wait_for_accept if wait_for_accept!=None else self.WAIT_FOR_ACCEPT)
        # connect to the created process and make its shadow.
        conn = Client(address=address, authkey=self.authkey)
        shadow = Shadow(connection=conn, address=address)
        shadow.remote_setattr('serial', len(self))
        self.append(shadow)

    def appoint(self, inetaddr, port, authkey):
        """
        Connect to the remotely-created worker will be sent to the process.

        @param inetaddr: TCP/IP address (or domain/host name).
        @type inetaddr: str
        @param port: port to connect to.
        @type port: int
        @param authkey: remote authkey.
        @type authkey: str
        """
        from .connection import Client
        # connect to the remotely created process and make its shadow.
        conn = Client(address=(inetaddr, port), authkey=authkey)
        shadow = Shadow(connection=conn, address=(inetaddr, port))
        shadow.remote_setattr('serial', len(self))
        self.append(shadow)

    def recruit(self, family=None, wait_for_accept=None):
        """
        Wait for a remote worker to register in.  The sequence of actions is:
          1. The dealer will create a listener on a given/known/public address
             for the worker to connect to.
          2. Prepare for the information (address, etc.) about another listener 
             to be created.
          3. Transfer the information about the new listener to the
             worker at the client side.
          4. Close the public connection and create the new listener.
        The rest is book-keeping the information about connection and make the
        shadow out of it.

        @param family: family of the connection to establish.
        @type family: str
        @keyword wait_for_accept: seconds to wait after accepting.  If None use
            DEFAULT.
        @type wait_for_accept: float
        """
        from .connection import Listener
        # start a listener at here, the dealer's side.
        publiclsnr = Listener(address=self.publicaddress, authkey=self.authkey)
        publicconn = publiclsnr.accept()
        # prepare the data to be sent to the worker for the persistent
        # connection.
        address = guess_address(family if family!=None else 'AF_INET')
        wait_for_accept = wait_for_accept if wait_for_accept!=None \
                                          else self.WAIT_FOR_ACCEPT
        # create the new listener and send the information.
        lsnr = Listener(address=address, authkey=self.authkey)
        publicconn.send((address, self.authkey, wait_for_accept))
        # close the old/public connection and accept for the new/persistent one.
        publicconn.close()
        conn = lsnr.accept()
        # book-keeping the shadows.
        shadow = Shadow(listener=lsnr, connection=conn)
        shadow.remote_setattr('serial', len(self))
        self.append(shadow)

    def bridge(self, peers, wait_for_accept=None):
        """
        Tell two peering worker to establish a connection.
        """
        from time import sleep
        plow, phigh = peers
        assert plow != phigh    # makes no sense.
        if plow > phigh:
            tmp = plow
            plow = phigh
            phigh = tmp
        # ask higher to accept connection.
        self[phigh].accept_peer(plow, self.family, self.authkey)
        address = self[phigh].recv()
        assert address == self[phigh].address
        # ask lower to make connection.
        sleep(wait_for_accept if wait_for_accept!=None else self.WAIT_FOR_ACCEPT)
        self[plow].connect_peer(phigh, address, self.authkey)

    def terminate(self, idx=slice(None,None,None), msg=None):
        """
        Termiinate workers.
        
        @param idx: what to terminate
        @type idx: slice or list
        @param msg: message to output after temination.
        @type msg: str
        """
        import sys
        for sdw in self[idx]:
            sdw.terminate()
        if msg:
            sys.stdout.write(msg)

    def barrier(self, idx=slice(None,None,None), msg=None):
        """
        Check for barrier signals sent from workers.  Used for synchronization.
        
        @param idx: what to synchronize.
        @type idx: slice or list
        @param msg: message to output after synchronization.
        @type msg: str
        """
        import sys
        for sdw in self[idx]:
            sdw.barrier()
        for sdw in self[idx]:
            assert issubclass(sdw.recv(), Barrier)
        if msg:
            sys.stdout.write(msg)

class Outpost(object):
    """
    A server object to listen to a certain port and create a worker when
    requested.

    @ivar publicaddress: the public address for worker to gain connection
        information.  It is used for recruitment.
    @itype publicaddress: tuple or str
    @ivar authkey: authentication key for worker connection.
    @itype authkey: str
    @ivar _procs: list of processes.
    @itype _procs: list
    """
    def __init__(self, *args, **kw):
        import sys
        self.publicaddress = kw.pop('publicaddress')
        self.authkey = kw.pop('authkey', DEFAULT_AUTHKEY)
        super(Outpost, self).__init__(*args, **kw)
        self.conn = None
        self._procs = list()

    def run(self):
        """
        Run the outpost by implementing an event loop.

        @return: nothing.
        """
        from .connection import Listener
        lsnr = Listener(address=self.publicaddress, authkey=self.authkey)
        while True:
            # accept the connection, get the control notice, and close
            # the connection immediately.
            self.conn = lsnr.accept()
            ntc = self.conn.recv()
            try:
                if isinstance(ntc, Control):
                    method = getattr(self, ntc.methodname)
                    ret = method(*ntc.args, **ntc.kw)
            except Terminate:
                for proc in self._procs:
                    proc.terminate()
                self.conn.close()
                break
            self.conn.close()

    def chdir(self, dirname):
        import os
        os.chdir(dirname)
        self.conn.send(None)
    def getpid(self):
        import os
        self.conn.send(os.getpid())

    def create(self, **kw):
        """
        Create another process for an empty worker, and register the worker to
        given address and authentication key.
        """
        from .connection import Process
        port = pick_unused_port()
        address = (self.publicaddress[0], port)
        proc = Process(
            target=Worker(None, **kw).run,
            args=(address, self.authkey),
        )
        proc.start()
        self._procs.append(proc)
        self.conn.send(port)

    def get_publicaddress(self):
        self.conn.send(self.publicaddress)

    def terminate(self):
        self.conn.send(None)
        raise Terminate

class Remote(object):
    """
    Remote execution through ssh.

    @cvar DEFAULT_SSH_CONFIG: default ssh configuration options.
    @ctype DEFAULT_SSH_CONFIG: dict
    @ivar address: inet address.
    @itype address: str
    @ivar username: username for the connecting machine.
    @itype username: str
    @ivar prescript: list of the Python statements to prepend before the main 
        body of execution.
    @itype prescript: list
    @ivar paths: dict of lists for various environmental variables for paths.
    @itype paths: dict
    @ivar ssh_config: ssh configuration options.
    @itype ssh_config: dict
    """

    DEFAULT_SSH_CONFIG = {
        'UserKnownHostsFile': '/dev/null',
        'StrictHostKeyChecking': 'no',
        'LogLevel': 'ERROR',
    }

    def __init__(self, address,
            username=None, prescript=None, paths=None, ssh_config=None):
        from .helper import get_username
        from .conf import env
        self.address = address
        self.username = username if username else get_username()
        self.prescript = prescript if prescript != None else list()
        # customiza paths.
        paths = paths if paths != None else dict()
        paths.setdefault('PYTHONPATH', list())
        if env.pkgdir not in paths['PYTHONPATH']:
            paths['PYTHONPATH'].append(env.pkgdir)
        self.paths = paths
        # ssh options.
        self.ssh_config = self.DEFAULT_SSH_CONFIG.copy()
        if ssh_config != None:
            self.ssh_config.update(ssh_config)

    @staticmethod
    def _pathmunge(key, pathlist):
        from .helper import iswin
        pathlist = pathlist[:]
        # scan for duplication.
        ip = 0
        while ip < len(pathlist):
            jp = ip + 1
            while jp < len(pathlist):
                if pathlist[ip] == pathlist[jp]:
                    del pathlist[jp]
                else:
                    jp += 1
            ip += 1
        # join the path.
        sep = ';' if iswin() else ':'
        pathstr = sep.join([path for path in pathlist if path])
        return 'export %s=%s:$%s' % (key, pathstr, key)

    @property
    def ssh_cmds(self):
        ssh_cmds = ['ssh', '-n']
        for key in self.ssh_config:
            ssh_cmds.append('-o')
            ssh_cmds.append('%s=%s' % (key, str(self.ssh_config[key])))
        ssh_cmds.append('%s@%s' % (self.username, self.address))
        return ssh_cmds

    def shell(self, script):
        from subprocess import Popen, PIPE, STDOUT
        cmds = self.ssh_cmds
        cmds.append('; '.join(script))
        subp = Popen(cmds, stdout=PIPE, stderr=STDOUT)
        return subp.stdout.read()

    def __call__(self, script, envar=None, stdout=None):
        """
        @param script: the script to be send to remote machine to execute.
        @type script: list
        @keywork envar: additional environment variables to remote.
        @type envar: dict
        """
        import sys
        from subprocess import Popen
        script = self.prescript + script
        # build the commands to be run remotely.
        remote_cmds = [self._pathmunge(k, self.paths[k]) for k in self.paths]
        if envar:
            remote_cmds.extend([
                'export %s=%s' % (key, envar[key]) for key in envar
            ])
        remote_cmds.append('%s -c \'%s\''%(sys.executable, '; '.join(script)))
        # build the commands for ssh.
        ssh_cmds = self.ssh_cmds
        # join ssh commands and remote commands and fire.
        subp = Popen(ssh_cmds + ['; '.join(remote_cmds)], stdout=stdout)
        # get the return from ssh.
        if subp.stdout != None:
            return subp.stdout.read()
        else:
            return None

class Footway(object):
    """
    Responsible for making connection to Outpost and submit a command to it.

    @ivar address: the AF_INET address to the outpost.
    @itype address: tuple
    @ivar authkey: the authkey for the outpost.
    @itype authkey: str
    @ivar pid: the pid on remote.
    @itype pid: int
    """

    WAIT_FOR_REACT = 0.1

    def __init__(self, address, authkey, wait_for_react=None):
        from time import sleep
        self.address = address
        self.authkey = authkey
        self.wait_for_react = wait_for_react if wait_for_react \
                                             else self.WAIT_FOR_REACT
        while not self.ready():
            sleep(self.wait_for_react)
        self.pid = self.getpid()

    def kill_remote(self):
        return Remote(self.address[0]).shell([
            'kill -9 %d' % self.pid,
        ])

    def __del__(self):
        self.kill_remote()

    def __getattr__(self, key):
        from time import sleep
        from .connection import Client
        from solvcon.rpc import Control
        def func(*arg, **kw):
            conn = Client(address=self.address, authkey=self.authkey)
            conn.send(Control(key, *arg, **kw))
            ret = conn.recv()
            sleep(self.wait_for_react)
            return ret
        return func

    def ready(self):
        from .connection import Client
        try:
            conn = Client(address=self.address, authkey=self.authkey)
        except:
            return False
        conn.send(Control('get_publicaddress'))
        assert conn.recv() == self.address
        return True

    @staticmethod
    def build_outpost(address, authkey=DEFAULT_AUTHKEY, envar=None, paths=None):
        """
        @param address: the IP/DN of the machine to build an outpost.
        @type address: str
        @keyword authkey: the authkey for the outpost.
        @type authkey: str
        @keywork envar: additional environment variables to remote.
        @type envar: dict
        @return: the port that the remote outpost listen on.
        @rtype: int
        """
        from subprocess import PIPE
        remote = Remote(address, paths=paths)
        val = int(remote([
            'import sys',
            'from solvcon.rpc import pick_unused_port',
            'sys.stdout.write(str(pick_unused_port()))',
        ], stdout=PIPE))
        try:
            port = int(val)
        except ValueError:
            raise IOError, 'remote port detection fails'
        remote([
            'from solvcon.rpc import Outpost',
            'outpost = Outpost(publicaddress=("%s", %d), authkey="%s")' % (
                address, port, authkey),
            'outpost.run()',
        ], envar=envar)
        return port

def run_server(dealer, nworker):
    """
    Take a dealer and recruit a number of workers for it.

    @param dealer: the dealer that can recruit.  In order to recruit, the 
        dealer must have publicaddress.
    @type dealer: Dealer
    @param nworker: number of workers the dealer needs.
    @type nworker: int
    """
    for iworker in range(nworker):
        dealer.recruit()
    return dealer

def run_client(address, authkey):
    """
    Summon a worker without any muscle and register it to the dealer specified
    on the address with the authentication key.

    @param address: the address to connect to.
    @type address: str or tuple
    @param authkey: authentication key for connection.
    @type authkey: str
    @return: nothing.
    """
    Worker(None).register(address, authkey)

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        # tell me a unused port on this machine.
        sys.stdout.write("%d\n" % pick_unused_port())
    elif len(sys.argv) == 5:
        # create a dealer according to the given information and wait for
        # workers to register.  this is a demonstration.
        publicaddress = (sys.argv[1], int(sys.argv[2]))
        authkey = sys.argv[3]
        dealer = Dealer(publicaddress=publicaddress, authkey=authkey)
        run_server(dealer, int(sys.argv[4]))
        for sdw in dealer:
            sys.stdout.write("%s at %s\n" % (
              sdw, sdw.listener.address))
        raw_input() # pause.
    elif len(sys.argv) == 4:
        # create a worker without muscle and register it to the remote dealer.
        run_client((sys.argv[1], int(sys.argv[2])), sys.argv[3])
