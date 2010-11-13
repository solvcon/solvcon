# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2010 Yung-Yu Chen.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
Commands for users.
"""

from .cmdutil import Command

class mpi(Command):
    """
    Utilities for MPI.
    """

    min_args = 0

    def __init__(self, env):
        from optparse import OptionGroup
        super(mpi, self).__init__(env)
        op = self.op
        opg = OptionGroup(op, 'MPI')
        opg.add_option('--node-env', action='store',
            dest='node_env', default='PBS_NODEFILE',
            help='Environment variable of the host file.',
        )
        opg.add_option('--compress-nodelist', action='store_true',
            dest='compress_nodelist', default=False,
            help='To compress nodelist on the head node.',
        )
        opg.add_option('--head-last', action='store_true',
            dest='head_last', default=False,
            help='Make the head node to be last in the host file.',
        )
        op.add_option_group(opg)
        self.opg_arrangement = opg

    def __call__(self):
        import os, sys
        ops, args = self.opargs
        f = open(os.environ[ops.node_env])
        nodes = [node.strip() for node in f.readlines()]
        f.close()
        if ops.compress_nodelist:
            newnodes = []
            for node in nodes:
                if not newnodes or newnodes[-1] != node:
                    newnodes.append(node)
            nodes = newnodes
        if ops.head_last:
            first = nodes[0]
            ind = 1
            while ind < len(nodes):
                if nodes[ind] != first:
                    break
                ind += 1
            nodes = nodes[ind:] + nodes[:ind]
        if len(args) > 0:
            f = open(args[0], 'w')
            f.write('\n'.join([node.strip() for node in nodes]))
            f.close()
        else:
            sys.stdout.write('\n'.join([node.strip() for node in nodes]))

class mesh(Command):
    """
    Mesh manipulation.
    """

    min_args = 1

    def __init__(self, env):
        from optparse import OptionGroup
        super(mesh, self).__init__(env)
        op = self.op

        opg = OptionGroup(op, 'Mesh')
        opg.add_option('--ascii', action='store_false',
            dest='binary', default=True,
            help='Use ASCII in VTK (default is binary).',
        )
        opg.add_option('--encoding', action='store', type='string',
            dest='encoding', default='raw',
            help='The encoding used in VTK XML binary data (raw/base64). '
                 'Must be base64 for inline binary data.',
        )
        opg.add_option('--inline', action='store_false',
            dest='appended', default=True,
            help='Inline binary data in VTK XML file.',
        )
        opg.add_option('--fpdtype', action='store', type='string',
            dest='fpdtype', default='float64',
            help='dtype for floating-point (default is float64).',
        )
        opg.add_option('--formats', action='store', type='string',
            dest='formats', default='',
            help='Assign the I/O formats as InputIO,OutputIO.',
        )
        opg.add_option('--compressor', action='store', type='string',
            dest='compressor', default='',
            help='Empty string (no compression), gz or bz2.',
        )
        opg.add_option('--split', action='store', type='int',
            dest='split', default=None,
            help='Split the loaded block into given number of parts.',
        )
        opg.add_option('--bc-reject', action='store', type='string',
            dest='bc_reject', default='',
            help='The BC (name) to be rejected in conversion.',
        )
        opg.add_option('--no-print-block', action='store_false',
            dest='print_block', default=True,
            help='Prevent printing block information.',
        )
        opg.add_option('--no-print-bcs', action='store_false',
            dest='print_bc', default=True,
            help='Prevent printing BC objects information.',
        )
        op.add_option_group(opg)
        self.opg_arrangement = opg

    @staticmethod
    def _save_block(ops, blk, blkfn):
        from time import time
        from .io.block import BlockIO
        from .helper import info
        bio = BlockIO(blk=blk, compressor=ops.compressor)
        info('Save to %s of blk format ... ' % blkfn)
        timer = time()
        bio.save(stream=blkfn)
        info('done. (%gs)\n' % (time()-timer))
    @staticmethod
    def _save_domain(ops, blk, dirname):
        import os
        from time import time
        from .domain import Collective
        from .io.domain import DomainIO
        from .helper import info
        info('Create domain ... ')
        timer = time()
        dom = Collective(blk)
        info('done. (%gs)\n' % (time()-timer))
        info('Split domain into %d parts ... ' % ops.split)
        timer = time()
        dom.split(ops.split)
        info('done. (%gs)\n' % (time()-timer))
        dio = DomainIO(dom=dom, compressor=ops.compressor)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        info('Save to directory %s/ ... ' % dirname)
        timer = time()
        dio.save(dirname=dirname)
        info('done. (%gs)\n' % (time()-timer))
    @staticmethod
    def _save_vtklegacy(ops, blk, vtkfn, binary, fpdtype):
        from time import time
        from .io.vtk import VtkLegacyUstGridWriter
        from .helper import info
        info('Save to file %s (%s/%s)... ' % (
            vtkfn, 'binary' if binary else 'ascii', fpdtype))
        timer = time()
        VtkLegacyUstGridWriter(blk,
            binary=binary, fpdtype=fpdtype).write(vtkfn)
        info('done. (%gs)\n' % (time()-timer))
    @staticmethod
    def _save_vtkxml(ops, blk, vtkfn, appended, binary, encoding, compressor,
        fpdtype):
        from time import time
        from .io.vtkxml import VtkXmlUstGridWriter
        from .helper import info
        if appended:
            fmt = 'appended'
        else:
            fmt = 'binary' if binary else 'ascii'
        timer = time()
        wtr = VtkXmlUstGridWriter(blk, appended=appended, binary=binary,
            encoding=encoding, compressor=compressor, fpdtype=fpdtype)
        info('Save to file %s (%s/%s/%s/%s)... ' % (vtkfn, fmt, wtr.encoding,
            compressor, fpdtype))
        wtr.write(vtkfn)
        info('done. (%gs)\n' % (time()-timer))
    @staticmethod
    def _determine_formats(ops, args):
        """
        Determine I/O formats based on arguments.

        @return: I/O formats.
        @rtype: FormatIO, str
        """
        from .helper import info
        from .io import gambit, block, domain   # refresh formatio registry.
        from .io.core import fioregy
        iio = oio = None
        if ops.formats:
            tokens = ops.formats.split(',')
            if len(tokens) == 1:
                iio = tokens[0]
            elif len(tokens) == 2:
                iio, oio = tokens
        if iio is None:
            fn = args[0]
            if fn.endswith('.blk'):
                iio = 'BlockIO'
            elif '.neu' in fn:
                iio = 'NeutralIO'
            else:
                iio = 'NeutralIO'
            if len(args) == 2:
                fn = args[1]
                if fn.endswith('.blk'):
                    oio = 'BlockIO'
                elif fn.endswith('.dom'):
                    oio = 'DomainIO'
                elif fn.endswith('.vtk'):
                    oio = 'VtkLegacy'
                elif fn.endswith('.vtu'):
                    oio = 'VtkXml'
            info('I/O formats are determined as: %s, %s.\n' % (iio, oio))
        iio = fioregy[iio]()
        return iio, oio
    def __call__(self):
        import os
        from time import time
        from .helper import info
        ops, args = self.opargs
        # determine file formats.
        iio, oio = self._determine_formats(ops, args)
        # load.
        info('Load %s (%s) ... ' % (args[0], type(iio).__name__))
        timer = time()
        blk = iio.load(stream=args[0])
        info('done. (%gs)\n' % (time()-timer))
        # print block information.
        if ops.print_block:
            info('Block information:')
            info('\n  %s\n' % str(blk))
            if ops.print_bc:
                for bc in blk.bclist:
                    info('    %s\n' % str(bc))
            info('  Cell groups:\n')
            for igrp in range(len(blk.grpnames)):
                grpname = blk.grpnames[igrp]
                info('    grp#%d: %s\n' % (igrp, grpname))
            info('  Cell volume (min, max, all): %g, %g, %g.\n' % (
                blk.clvol.min(), blk.clvol.max(), blk.clvol.sum()))
        # save.
        if oio is not None:
            path = args[1]
            if oio == 'BlockIO':
                self._save_block(ops, blk, path)
            elif oio == 'DomainIO':
                if ops.split is None:
                    info('No saving: split must be specified.\n')
                    return
                self._save_domain(ops, blk, path)
            elif oio == 'VtkLegacy':
                self._save_vtklegacy(ops, blk, path, ops.binary, ops.fpdtype)
            elif oio == 'VtkXml':
                self._save_vtkxml(ops, blk, path, ops.appended, ops.binary,
                    ops.encoding, ops.compressor, ops.fpdtype)

class SolverLog(Command):
    """
    Actions related Solver log.
    """

    def __init__(self, env):
        from optparse import OptionGroup
        super(SolverLog, self).__init__(env)
        op = self.op

        opg = OptionGroup(op, 'Solver Log')
        opg.add_option('-f', action='store',
            dest='filename', default=None,
            help='Save plot to a file with specified name.',
        )
        opg.add_option('--backend', action='store',
            dest='backend', default='TkAgg',
            help='The backend for matplotlib.  Default is TkAgg.',
        )
        op.add_option_group(opg)
        self.opg_arrangement = opg

    def _get_datas(self):
        import os, glob
        ops, args = self.opargs
        fn = args[0]
        fns = list()
        if os.path.isdir(fn):
            if ops.filename != None:
                main, ext = os.path.splitext(ops.filename)
                dsttmpl = main+'%d'+ext
            else:
                dsttmpl = None
            nfn = len(glob.glob(os.path.join(fn, 'solvcon.solver*.log')))
            for idx in range(nfn):
                src = os.path.join(fn, 'solvcon.solver%d.log'%idx)
                if dsttmpl != None:
                    dst = dsttmpl%idx
                else:
                    dst = None
                lines = open(src).readlines()
                fns.append((lines, src, dst))
        else:
            lines = open(fn).readlines()
            fns.append((lines, fn, ops.filename))
        return fns

class log_runtime(SolverLog):
    """
    Show output from RuntimeStatAnchor.
    """

    min_args = 1
    PLOTS = ['cpu', 'loadavg', 'mem']

    def __init__(self, env):
        from optparse import OptionGroup
        super(log_runtime, self).__init__(env)
        op = self.op

        opg = OptionGroup(op, 'Show Runtime')
        opg.add_option('-c', action='store_true',
            dest='cpu', default=False,
            help='Plot CPU usage.',
        )
        opg.add_option('-l', action='store_true',
            dest='loadavg', default=False,
            help='Plot load average.',
        )
        opg.add_option('-m', action='store_true',
            dest='mem', default=False,
            help='Plot memory usage.',
        )
        opg.add_option('-t', action='store_true',
            dest='xtime', default=False,
            help='Use time as x-axis.',
        )
        opg.add_option('--lloc', action='store',
            dest='lloc', default=None,
            help='Legend location.  Default is None (by plot).',
        )
        opg.add_option('--scale', action='store', type=int,
            dest='scale', default=0.6,
            help='The scale when having more than one subplot.'
                 ' Default is 0.6.',
        )
        op.add_option_group(opg)
        self.opg_arrangement = opg

    def _init_mpl(self, nplot):
        import matplotlib
        ops, args = self.opargs
        figsize = matplotlib.rcParams['figure.figsize']
        top = matplotlib.rcParams['figure.subplot.top']
        bottom = matplotlib.rcParams['figure.subplot.bottom']
        if nplot > 1:
            upscale = nplot*ops.scale
            top = 1.0 - (1.0-top)*(1.0-top)/((1.0-top)*upscale)
            bottom = bottom*bottom/(bottom*upscale)
            figsize = figsize[0], figsize[1]*upscale
        matplotlib.rcParams.update({
            'backend': ops.backend,
            'figure.figsize': figsize,
            'figure.subplot.top': top,
            'figure.subplot.bottom': bottom,
        })
        matplotlib.use(ops.backend)

    def __call__(self):
        import os, sys
        from .anchor import RuntimeStatAnchor
        ops, args = self.opargs
        # count plots.
        nplot = 0
        for key in self.PLOTS:
            if getattr(ops, key):
                nplot += 1
        self._init_mpl(nplot)
        from matplotlib import pyplot as plt
        # get source and destination.
        datas = self._get_datas()
        # plot.
        for lines, src, dst in datas:
            if nplot:
                fig = plt.figure()
            iplot = 1
            for key in self.PLOTS:
                if getattr(ops, key):
                    ax = fig.add_subplot(nplot, 1, iplot)
                    kws = {
                        'xtime': ops.xtime,
                        'showx': iplot==nplot,
                    }
                    if ops.lloc != None:
                        kws['lloc'] = ops.lloc
                    getattr(RuntimeStatAnchor, 'plot_'+key)(lines, ax, **kws)
                    iplot += 1
            if nplot:
                sys.stdout.write('%s processed' % src)
                if dst != None:
                    plt.savefig(dst)
                    sys.stdout.write(' and written to %s.' % dst)
                sys.stdout.write('\n')
        # show.
        if nplot and ops.filename == None:
            plt.show()

class log_march(SolverLog):
    """
    Show output from MarchStatAnchor.
    """

    min_args = 1

    def __init__(self, env):
        from optparse import OptionGroup
        super(log_march, self).__init__(env)
        op = self.op

        opg = OptionGroup(op, 'Show March')
        opg.add_option('-k', action='store',
            dest='plotkeys', default='',
            help='Keys to plot.',
        )
        opg.add_option('--lloc', action='store',
            dest='lloc', default=None,
            help='Legend location.  Default is None (by plot).',
        )
        opg.add_option('--scale', action='store', type=int,
            dest='scale', default=0.6,
            help='The scale when having more than one subplot.'
                 ' Default is 0.6.',
        )
        op.add_option_group(opg)
        self.opg_arrangement = opg

    def _init_mpl(self, nplot):
        import matplotlib
        ops, args = self.opargs
        matplotlib.rcParams.update({
            'backend': ops.backend,
        })
        matplotlib.use(ops.backend)

    def __call__(self):
        import os, sys
        from .anchor import MarchStatAnchor
        ops, args = self.opargs
        # count plots.
        plotkeys = ops.plotkeys.split(',')
        nplot = len(plotkeys)
        self._init_mpl(nplot)
        from matplotlib import pyplot as plt
        # get source and destination.
        datas = self._get_datas()
        # plot.
        if nplot == 0:
            return
        for lines, src, dst in datas:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            kws = dict()
            if ops.lloc != None:
                kws['lloc'] = ops.lloc
            MarchStatAnchor.plot(plotkeys, lines, ax, **kws)
            sys.stdout.write('%s processed' % src)
            if dst != None:
                plt.savefig(dst)
                sys.stdout.write(' and written to %s.' % dst)
            sys.stdout.write('\n')
        # show.
        if ops.filename == None:
            plt.show()

class log_tpool(SolverLog):
    """
    Show output from TpoolStatAnchor.
    """

    min_args = 1

    def __init__(self, env):
        from optparse import OptionGroup
        super(log_tpool, self).__init__(env)
        op = self.op

        opg = OptionGroup(op, 'Show Tpool')
        opg.add_option('-k', action='store',
            dest='plotkeys', default='',
            help='Keys to plot.',
        )
        opg.add_option('--lloc', action='store',
            dest='lloc', default=None,
            help='Legend location.  Default is None (by plot).',
        )
        opg.add_option('--scale', action='store', type=int,
            dest='scale', default=0.6,
            help='The scale when having more than one subplot.'
                 ' Default is 0.6.',
        )
        op.add_option_group(opg)
        self.opg_arrangement = opg

    def _init_mpl(self, nplot):
        import matplotlib
        ops, args = self.opargs
        figsize = matplotlib.rcParams['figure.figsize']
        top = matplotlib.rcParams['figure.subplot.top']
        bottom = matplotlib.rcParams['figure.subplot.bottom']
        if nplot > 1:
            upscale = nplot*ops.scale
            top = 1.0 - (1.0-top)*(1.0-top)/((1.0-top)*upscale)
            bottom = bottom*bottom/(bottom*upscale)
            figsize = figsize[0], figsize[1]*upscale
        matplotlib.rcParams.update({
            'backend': ops.backend,
            'figure.figsize': figsize,
            'figure.subplot.top': top,
            'figure.subplot.bottom': bottom,
        })
        matplotlib.use(ops.backend)

    def __call__(self):
        import os, sys
        from .anchor import TpoolStatAnchor
        ops, args = self.opargs
        # count plots.
        plotkeys = ops.plotkeys.split(',')
        nplot = len(plotkeys)
        self._init_mpl(nplot)
        from matplotlib import pyplot as plt
        # get source and destination.
        datas = self._get_datas()
        # plot.
        for lines, src, dst in datas:
            if nplot:
                fig = plt.figure()
            iplot = 1
            for key in plotkeys:
                ax = fig.add_subplot(nplot, 1, iplot)
                kws = {
                    'showx': iplot==nplot,
                }
                if ops.lloc != None:
                    kws['lloc'] = ops.lloc
                TpoolStatAnchor.plot(key, lines, ax, **kws)
                iplot += 1
            if nplot:
                sys.stdout.write('%s processed' % src)
                if dst != None:
                    plt.savefig(dst)
                    sys.stdout.write(' and written to %s.' % dst)
                sys.stdout.write('\n')
        # show.
        if nplot and ops.filename == None:
            plt.show()

class ArrangementCommand(Command):
    """
    @ivar opg_arrangement: group for options for arrangement.
    @itype opg_arrangement: optparse.OptionGroup
    """

    def __init__(self, env):
        from optparse import OptionGroup
        super(ArrangementCommand, self).__init__(env)
        op = self.op

        opg = OptionGroup(op, 'Arrangement')
        opg.add_option('--runlevel', action='store', type=int,
            dest='runlevel', default=0,
            help='0: fresh run, 1: restart, 2: init only.',
        )
        opg.add_option('--solver-output', action='store_true',
            dest='solver_output', default=False,
            help='Turn on the output device in the solver object(s).',
        )
        opg.add_option('--npart', action='store', type=int,
            dest='npart', default=None,
            help='The number of partitions.',
        )
        opg.add_option('--compress-nodelist', action='store_true',
            dest='compress_nodelist', default=False,
            help='To compress nodelist on the head node.',
        )
        opg.add_option('-e', '--envar', action='store',
            dest='envar', default=None,
            help='Additional environmental variable to remote solvers.',
        )
        opg.add_option('-b', '--batch', action='store',
            dest='batch', default='Batch',
            help='The name of batch system.',
        )
        opg.add_option('--use-profiler', action='store_true',
            dest='use_profiler', default=False,
            help='Flag to use profiler in running or not.',
        )
        opg.add_option('--profiler-sort', action='store', type='string',
            dest='profiler_sort', default='cum,time',
            help='Fields for sorting stats in profiler; comma separated.',
        )
        opg.add_option('--profiler-dat', action='store', type='string',
            dest='profiler_dat', default='profiler.dat',
            help='File name for raw profiler output.',
        )
        opg.add_option('--profiler-log', action='store', type='string',
            dest='profiler_log', default='profiler.log',
            help='File name for human-readable profiler output.',
        )
        opg.add_option('--basedir', action='store',
            dest='basedir', default='',
            help='Suggested basedir (may or may not used by arrangement).',
        )
        opg.add_option('--test', action='store_true',
            dest='test', default=False,
            help='General flags for test run.',
        )
        op.add_option_group(opg)
        self.opg_arrangement = opg

    @property
    def envar(self):
        ops, args = self.opargs
        dct = dict()
        if ops.envar:
            for ent in ops.envar.split(','):
                key, val = [it.strip() for it in ent.split('=')]
                dct[key] = val
        return dct

class run(ArrangementCommand):
    """
    Run arrangement.
    """

    min_args = 0

    def __call__(self):
        import os
        import cProfile
        import pstats
        from socket import gethostname
        from .helper import info
        from .conf import use_application, env
        from . import domain
        from .batch import batregy
        from .case import arrangements
        from .rpc import Worker, DEFAULT_AUTHKEY
        ops, args = self.opargs
        if len(args) > 0:
            name = args[0]
        else:
            name = os.path.basename(os.getcwd())
        # get batch.
        batch = batregy[ops.batch]
        # get partition number and domain type.
        npart = ops.npart
        if npart != None:
            if batch == batregy.Batch:
                domaintype = domain.Collective
            else:
                domaintype = domain.Distributed
        else:
            domaintype = domain.Domain
        # run.
        funckw = {
            'envar': self.envar,
            'runlevel': ops.runlevel,
            'solver_output': ops.solver_output,
            'batch': batch,
            'npart': npart, 'domaintype': domaintype,
        }
        func = arrangements[name]
        if env.mpi and env.mpi.rank != 0:
            pdata = (
                ops.profiler_dat,
                ops.profiler_log,
                ops.profiler_sort,
            ) if ops.use_profiler else None
            wkr = Worker(None, profiler_data=pdata)
            wkr.run(('0.0.0.0', 0), DEFAULT_AUTHKEY)    # FIXME
        else:
            if ops.use_profiler:
                cProfile.runctx('func(submit=False, **funckw)',
                    globals(), locals(), ops.profiler_dat)
                plog = open(ops.profiler_log, 'w')
                p = pstats.Stats(ops.profiler_dat, stream=plog)
                p.sort_stats(*ops.profiler_sort.split(','))
                p.dump_stats(ops.profiler_dat)
                p.print_stats()
                plog.close()
                info('*** Profiled information saved in '
                    '%s (raw) and %s (text).\n' % (
                    ops.profiler_dat, ops.profiler_log))
            else:
                func(submit=False, **funckw)

class submit(ArrangementCommand):
    """
    Submit arrangement to batch system.
    """

    min_args = 1

    def __init__(self, env):
        from optparse import OptionGroup
        super(submit, self).__init__(env)
        op = self.op

        opg = OptionGroup(op, 'Batching')
        opg.add_option('-l', '--resources', action='store',
            dest='resources', default='',
            help='Resource list with "," as delimiter.',
        )
        opg.add_option('--use-mpi', action='store_true',
            dest='use_mpi', default=False,
            help='Indicate to use MPI as transport layer.',
        )
        opg.add_option('--postpone', action='store_true',
            dest='postpone', default=False,
            help='Postpone feeding into batch system.',
        )
        op.add_option_group(opg)
        self.opg_batch = opg

    def __call__(self):
        import os
        from .conf import use_application
        from .batch import batregy
        from .case import arrangements
        ops, args = self.opargs
        if len(args) > 0:
            name = args[0]
        else:
            name = os.path.basename(os.getcwd())
        # import application packages.
        for modname in self.env.modnames:
            use_application(modname)
        # build resource list.
        resources = dict([(line, None) for line in ops.resources.split(',')])
        # get batch class.
        batch = batregy[ops.batch]
        # submit to arrangement.
        arrangements[name](submit=True,
            use_mpi=ops.use_mpi, postpone=ops.postpone,
            envar=self.envar,
            runlevel=ops.runlevel,
            resources=resources, batch=batch, npart=ops.npart,
        )

class help(Command):
    """
    Print general help.
    """

    def __call__(self):
        ops, args = self.opargs
        self.op.print_help()

    @property
    def usage(self):
        return '\n'.join([self._usage+'\n', self.command_description])
