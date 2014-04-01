# -*- coding: UTF-8 -*-
# Copyright (C) 2010-2011 by Yung-Yu Chen.  All rights reserved.

"""
Batch systems for supercomputers.
"""

from solvcon.batch import Torque

class OscGlenn(Torque):
    BASH_HOME_SOURCE = ['.bashrc_path', '.bash_acct']
    @property
    def _hostfile(self):
        import os
        return os.path.join(self.jobdir, self.jobname+'.hostfile')
    @property
    def str_prerun(self):
        import os
        from solvcon.conf import env
        ops, args = env.command.opargs
        msgs = [super(OscGlenn, self).str_prerun]
        msgs.append('export')   # DEBUG
        if self.use_mpi:
            msgs.extend([
                #'export I_MPI_DEBUG=2',
                #'export I_MPI_DEVICE=rdma:OpenIB-cma',
                'module unload mpi',
                'module unload mpi2',
                'module load mvapich2-1.5-gnu',
                'module load pvfs2',
                #'module load intel-compilers-11.1',
                #'module load mvapich2-1.5-intel',
                #'module load mvapich2-1.4.1-gnu',
                #'module load mvapich2-1.4-gnu',
                #'module load intel-mpi-4.0.0.028',
            ])
            msgs.append('%s %s \\\n %s' % (
                env.get_entry_point(),
                'mpi --compress-nodelist' if ops.compress_nodelist else 'mpi',
                self._hostfile,
            ))
        return '\n'.join(msgs)
    def build_mpi_runner(self):
        from solvcon.conf import env
        ops, args = env.command.opargs
        cmds = ['LD_PRELOAD=libmpich.so', 'SOLVCON_MPI=1',
            'mpiexec', '-n %d'%(ops.npart+1), '\\\n']
        if ops.compress_nodelist:
            cmds.append('-pernode')
        #cmds = ['mpirun_rsh', '-rsh',
        #    '-np %d'%(ops.npart+1), '\\\n',
        #    '-hostfile %s' % self._hostfile, '\\\n',
        #    'LD_PRELOAD=libmpich.so',
        #    'PBS_NODEFILE=$PBS_NODEFILE', 'SOLVCON_MPI=1', '\\\n',
        #]
        return ' '.join(cmds)
    @property
    def str_postrun(self):
        import sys, os
        newstr = [super(OscGlenn, self).str_postrun]
        newstr.extend([
            'mpiexec -comm none -pernode killall %s' % \
            os.path.basename(sys.executable),
        ])
        return '\n'.join(newstr)

class OscGlennGbE(OscGlenn):
    pass
class OscGlennIB(OscGlenn):
    def nodelist(self):
        ndlst = super(OscGlennIB, self).nodelist()
        for node in ndlst:
            if '-ib-' not in node.name:
                node.name = node.name[:3] + '-ib-' + node.name[3:]
        return ndlst
