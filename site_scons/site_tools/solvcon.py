"""
SOLVCON Tool for SCons
"""

import sys
import os

def has_sse4(env):
    if not sys.platform.startswith('linux'):
        return False
    entries = [line.split(':') for line in
        open('/proc/cpuinfo').read().strip().split('\n') if len(line) > 0]
    cpuinfo = dict([(entry[0].strip(), entry[1].strip()) for entry in entries])
    if 'sse4' in cpuinfo['flags']:
        return True
    return False

def get_scdata(env, url, datapath):
    if os.path.exists(datapath):
        orig = os.getcwd()
        os.chdir(datapath)
        os.system('hg pull -u')
        os.chdir(orig)
    else:
        os.system('hg clone %s %s' % (url, datapath))

def generate(env):
    env.AddMethod(has_sse4, 'HasSse4')
    env.AddMethod(get_scdata, 'GetScdata')

def exists(env):
    return env.Detect('solvcon')
