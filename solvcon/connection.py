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
Remote connection and communication.
"""

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

def guess_family(address):
    """
    Determine the family of address.
    """
    if type(address) == tuple:
        return 'AF_INET'
    elif type(address) is str and address.startswith('\\\\'):
        raise ValueError('Windows pipe is not supported')
        #return 'AF_PIPE'
    elif type(address) is str:
        return 'AF_UNIX'
    else:
        raise ValueError('address type of %r unrecognized' % address)

class Credential(object):
    """
    Authenticating information to be exchanged between two ends of a
    connection.
    """
    MSGCAP = 256
    MSGLEN = 20
    CHALLENGE = b'#CHALLENGE#'
    SUCCESS = b'#SUCCESS#'
    FAILURE = b'#FAILURE#'
    def __init__(self, conn, authkey):
        """
        @param conn: the communicating connection.
        @type conn: solvcon.connection.Connection
        @param authkey: authenticating key.
        @type authkey: str
        """
        assert isinstance(authkey, bytes)
        self.conn = conn
        self.authkey = authkey
    def question(self):
        import os, hmac
        conn = self.conn
        msg = os.urandom(self.MSGLEN)
        conn.send_bytes(self.CHALLENGE+msg)
        digest = hmac.new(self.authkey, msg).digest()
        res = conn.recv_bytes(self.MSGCAP)
        if res == digest:
            conn.send_bytes(self.SUCCESS)
        else:
            conn.send_bytes(self.FAILURE)
            raise IOError('digest received was wrong')
    def answer(self):
        import hmac
        conn = self.conn
        msg = conn.recv_bytes(self.MSGCAP)
        assert msg[:len(self.CHALLENGE)] == self.CHALLENGE, 'msg = %r'%msg
        msg = msg[len(self.CHALLENGE):]
        digest = hmac.new(self.authkey, msg).digest()
        conn.send_bytes(digest)
        res = conn.recv_bytes(self.MSGCAP)
        if res != self.SUCCESS:
            raise IOError('digest sent was rejected')

class SocketConnection(object):
    def __init__(self, *args, **kw):
        from _multiprocessing import Connection
        self.conn = Connection(*args, **kw)
    def send_bytes(self, *args, **kw):
        return self.conn.send_bytes(*args, **kw)
    def recv_bytes(self, *args, **kw):
        return self.conn.recv_bytes(*args, **kw)
    def send(self, *args, **kw):
        return self.conn.send(*args, **kw)
    def recv(self, *args, **kw):
        return self.conn.recv(*args, **kw)
    def close(self, *args, **kw):
        return self.conn.close(*args, **kw)
    def sendarr(self, arr):
        self.send(arr)
    def recvarr(self, arr):
        arr[:] = self.recv()[:]

class MPIConnection(object):
    TAG = 1
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
    def send(self, dat):
        from .conf import env
        env.mpi.send(dat, self.dst, self.TAG)
    def recv(self):
        from .conf import env
        return env.mpi.recv(self.dst, self.TAG)
    def sendarr(self, arr):
        from .conf import env
        env.mpi.sendarr(arr, self.dst, self.TAG)
    def recvarr(self, arr):
        from .conf import env
        env.mpi.recvarr(arr, self.dst, self.TAG)

CLIENT_TIMEOUT = 20.
def Client(address, family=None, authkey=None):
    """
    Establish a connection to a Listener.

    @param address: The address of a Unix or TCP/IP socket.
    @type address: str or tuple
    @keyword family: The family of address.
    @type family: str
    @keyword authkey: Authenticating key.
    @type authkey: str
    """
    import os, time, errno, socket
    timeout = time.time() + CLIENT_TIMEOUT
    family = family or guess_family(address)
    # create socket.
    skt = socket.socket(getattr(socket, family))
    while True:
        try:
            skt.connect(address)
        except socket.error, e:
            if e.args[0] != errno.ECONNREFUSED or time.time() > timeout:
                raise
            time.sleep(0.01)
        else:
            break
    # create connection.
    conn = SocketConnection(os.dup(skt.fileno()))
    skt.close()
    # authenticate.
    if authkey is not None:
        if not isinstance(authkey, bytes):
            raise TypeError('authkey must be a byte string')
        credit = Credential(conn, authkey)
        credit.answer()
        credit.question()
    return conn

class Listener(object):
    """
    Socket listener for connection.
    """
    def __init__(self, address, family=None, authkey=None):
        """
        @param address: The address of a Unix or TCP/IP socket.
        @type address: str or tuple
        @keyword family: The family of address.
        @type family: str
        @keyword authkey: Authenticating key.
        @type authkey: str
        """
        import socket
        family = family or (address and guess_family(address))
        # create socket.
        self._socket = socket.socket(getattr(socket, family))
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(address)
        self._socket.listen(1)
        # store extra information.
        self.address = self._socket.getsockname()
        self._last_accepted = None
        self._authkey = authkey
    def accept(self):
        import os
        # accepting connection.
        skt, self._last_accepted = self._socket.accept()
        # establish connection.
        conn = SocketConnection(os.dup(skt.fileno()))
        skt.close()
        # authenticate.
        if self._authkey:
            credit = Credential(conn, self._authkey)
            credit.question()
            credit.answer()
        return conn
    def close(self):
        self._socket.close()

class SpanningTreeNode(dict):
    def __init__(self, *args, **kw):
        self.val = kw.pop('val')
        self.level = kw.pop('level')
        super(SpanningTreeNode, self).__init__(*args, **kw)
    def traverse(self, graph, visited):
        if self.val not in visited:
            visited[self.val] = True
            for it in graph[self.val]:
                nd = SpanningTreeNode(val=it, level=self.level+1)
                if nd.traverse(graph, visited) == True:
                    self[it] = nd
            return True
        else:
            return False
