# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Commands for users.
"""

from .cmdutil import Command

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
        opg.add_option('--from-neu-file', action='store', type='string',
            dest='neu_file', default='',
            help='The file name for source Gambit Neutral file.',
        )
        opg.add_option('--compressor', action='store', type='string',
            dest='compressor', default='',
            help='Empty string (no compression), gz or bz2.',
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

    def _load_mesh(self, ops, args):
        from time import time
        import gzip
        from .io.gambit import GambitNeutral
        from .io.block import BlockIO
        from .helper import info
        if ops.neu_file != '':
            neufn = ops.neu_file
            # load block from other format.
            if neufn.endswith('.gz'):
                opener = gzip.open
            else:
                opener = open
            info('Loading %s ... ' % neufn)
            timer = time()
            data = opener(neufn).read()
            info('%d byte. (%gs)\n' % (len(data), time()-timer))
            # create neu object.
            info('Parse Gambit Neutral ... ')
            timer = time()
            neu = GambitNeutral(data)
            if ops.print_block:
                info('\n  %s\n' % str(neu))
                if ops.print_bc:
                    for bc in neu.bcs:
                        info('    %s\n' % str(bc))
            info('  (%gs)\n' % (time()-timer))
            # creat block object.
            info('Convert to block object ...')
            timer = time()
            if ops.bc_reject:
                rejects = ops.bc_reject.split(',')
                onlybcnames = list()
                for bc in neu.bcs:
                    if bc.name not in rejects:
                        onlybcnames.append(bc.name)
                    else:
                        info('\n  CAUTION: "%s" BC is rejected.' % bc.name)
            else:
                onlybcnames = None
            blk = neu.toblock(onlybcnames=onlybcnames)
            info('  (%gs)\n' % (time()-timer))
        else:
            bio = BlockIO()
            blk = bio.load(stream=args[0])
        return blk

    def _save_mesh(self, ops, args, blk):
        from time import time
        from .io.block import BlockIO
        from .helper import info
        blkfn = args[0]
        # save block.
        bio = BlockIO(blk=blk, flag_compress=ops.compressor)
        info('Save to blk format ... ')
        timer = time()
        bio.save(stream=blkfn)
        info('done. (%gs)\n' % (time()-timer))

    def __call__(self):
        from time import time
        from .helper import info
        ops, args = self.opargs
        # load.
        blk = self._load_mesh(ops, args)
        # print block information.
        info('Block information:')
        if ops.print_block:
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
            info('  Face area (min, max, all): %g, %g, %g.\n' % (
                blk.fcarea.min(), blk.fcarea.max(), blk.fcarea.sum()))
        # save.
        if ops.neu_file:
            self._save_mesh(ops, args, blk)

class log(Command):
    """
    Log output.
    """

    min_args = 1
    PLOTS = ['cpu', 'perf', 'march', 'marchsol', 'marchdsol', 'loadavg', 'mem']

    def __init__(self, env):
        from optparse import OptionGroup
        super(log, self).__init__(env)
        op = self.op

        opg = OptionGroup(op, 'Show Log')
        opg.add_option('-c', action='store_true',
            dest='cpu', default=False,
            help='Plot CPU usage.',
        )
        opg.add_option('-p', action='store_true',
            dest='perf', default=False,
            help='Plot performance (inverse of marching time).',
        )
        opg.add_option('-a', action='store_true',
            dest='march', default=False,
            help='Plot time spent in march for update.',
        )
        opg.add_option('-s', action='store_true',
            dest='marchsol', default=False,
            help='Plot time spent in march for sol.',
        )
        opg.add_option('-d', action='store_true',
            dest='marchdsol', default=False,
            help='Plot time spent in march for dsol.',
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
        opg.add_option('--backend', action='store',
            dest='backend', default='Agg',
            help='The backend for matplotlib.',
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

    def __call__(self):
        from matplotlib import pyplot as plt
        from .anchor import RuntimeStatAnchor
        ops, args = self.opargs
        # count plots.
        nplot = 0
        for key in self.PLOTS:
            if getattr(ops, key):
                nplot += 1
        self._init_mpl(nplot)
        # load log data.
        lines = open(args[0]).readlines()
        # plot.
        iplot = 1
        if nplot:
            fig = plt.figure()
        for key in self.PLOTS:
            if getattr(ops, key):
                ax = fig.add_subplot(nplot, 1, iplot)
                showx = iplot==nplot
                getattr(RuntimeStatAnchor, 'plot_'+key)(
                    lines, ax, xtime=ops.xtime, showx=showx,
                )
                iplot += 1
        # show.
        if nplot:
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
        opg.add_option('-s', '--scheduler', action='store',
            dest='scheduler', default='Scheduler',
            help='The name of scheduler.',
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
            for ent in ops.envar.split(':'):
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
        from .helper import info
        from .conf import use_application
        from . import batch, domain
        from .case import arrangements
        ops, args = self.opargs
        if len(args) > 0:
            name = args[0]
        else:
            name = os.path.basename(os.getcwd())
        # import application packages.
        for modname in self.env.modnames:
            use_application(modname)
        # get scheduler.
        scheduler = getattr(batch, ops.scheduler)
        # get partition number and domain type.
        npart = ops.npart
        if npart != None:
            if scheduler == batch.Scheduler:
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
            'scheduler': scheduler,
            'npart': npart, 'domaintype': domaintype,
        }
        func = arrangements[name]
        if ops.use_profiler:
            cProfile.runctx('func(submit=False, **funckw)', globals(), locals(),
                ops.profiler_dat)
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
    Submit arrangement to batch scheduler.
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
        opg.add_option('--postpone', action='store_true',
            dest='postpone', default=False,
            help='Postpone feeding into scheduler.',
        )
        op.add_option_group(opg)
        self.opg_batch = opg

    def __call__(self):
        import os
        from .conf import use_application
        from . import batch
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
        # get scheduler class.
        scheduler = getattr(batch, ops.scheduler)
        # submit to arrangement.
        arrangements[name](submit=True, postpone=ops.postpone,
            envar=self.envar,
            runlevel=ops.runlevel,
            resources=resources, scheduler=scheduler, npart=ops.npart,
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
