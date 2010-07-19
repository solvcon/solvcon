# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Remote connection and communication.
"""

from multiprocessing import Process

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
    from _multiprocessing import Connection
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
    conn = Connection(os.dup(skt.fileno()))
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
        self.address = address
        self._last_accepted = None
        self._authkey = authkey
    def accept(self):
        import os
        from _multiprocessing import Connection
        # accepting connection.
        skt, self._last_accepted = self._socket.accept()
        # establish connection.
        conn = Connection(os.dup(skt.fileno()))
        skt.close()
        # authenticate.
        if self._authkey:
            credit = Credential(conn, self._authkey)
            credit.question()
            credit.answer()
        return conn
    def close(self):
        self._socket.close()
