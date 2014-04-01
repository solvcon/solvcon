# -*- coding: UTF-8 -*-
# Copyright (C) 2010-2011 by Yung-Yu Chen.  All rights reserved.

from solvcon.command import SolverLog

class BenchResult(dict):
    COMM_KEYS = ['ibcam', 'ibcsoln', 'ibcdsoln']
    CALC_KEYS = ['update', 'calcsolt', 'calcsoln', 'calcdsoln', 'calccfl']
    BOUND_KEYS = ['bcsoln', 'bcdsoln']
    MARCH_KEYS = COMM_KEYS + CALC_KEYS + BOUND_KEYS
    def __init__(self, *args, **kw):
        self.dirname = kw.pop('dirname')
        super(BenchResult, self).__init__(*args, **kw)
        # read time data.
        times = self._read_times(self.dirname)
        # performance.
        self.perf = times.pop('perf')
        # categorized time.
        self.alltime = 0.0
        for name in 'comm', 'calc', 'bound':
            names = getattr(self, name.upper()+'_KEYS')
            stime = sum([times[key] for key in names])
            setattr(self, name, stime)
            self.alltime += stime
        # store raw times.
        self.update(times)
    @classmethod
    def _read_times(cls, dirname):
        import sys, os, glob
        from solvcon.anchor import MarchStatAnchor
        if not os.path.isdir(dirname):
            raise IOError('%s is not a directory'%dirname)
        # load data.
        sys.stdout.write('reading data from %s ... ' % dirname)
        nfn = len(glob.glob(os.path.join(dirname, 'solvcon.solver*.log')))
        datas = list()
        for idx in range(nfn):
            fn = 'solvcon.solver'
            fn += '.log' if nfn == 1 else '%d.log'%idx
            datas.append(open(os.path.join(dirname, fn)).readlines())
        sys.stdout.write('%d files done.\n' % len(datas))
        # accumulate time.
        times = dict()
        ifn = 0
        for lines in datas:
            for key in ['march'] + cls.MARCH_KEYS:
                arr, dmp, dmp = MarchStatAnchor.parse(lines, key, diff=False)
                times.setdefault(key, 0.0)
                times[key] += arr[-1,0]
            ifn += 1
            sys.stdout.write('.')
            sys.stdout.flush()
            if ifn != 0 and ifn%80 == 0:
                sys.stdout.write('\n')
        if ifn % 80 != 0:
            sys.stdout.write('\n')
        # performance.
        pfn = os.path.join(dirname, os.path.basename(dirname)+'_perf.txt')
        times['perf'] = [float(line.split()[0]) for line in
            open(pfn).readlines()[1:]]
        return times

class pjcf(SolverLog):
    """
    Aggregate marching time for pjcf cases.
    """
    SYMBS = '+x*o^s'

    min_args = 0
    def __init__(self, env):
        from optparse import OptionGroup
        super(pjcf, self).__init__(env)
        op = self.op

        opg = OptionGroup(op, 'Show Runtime')
        opg.add_option('--droot', action='store',
            dest='droot', default='run/benchmark',
            help='Data root.',
        )
        opg.add_option('--cpn', action='store', type=int,
            dest='cpn', default=4,
            help='Number of cores per node.',
        )
        opg.add_option('--show', action='store_true',
            dest='show', default=False,
            help='Show plot in screen.',
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
    def _init_mpl(self):
        import matplotlib
        ops, args = self.opargs
        matplotlib.rcParams.update({
            'backend': ops.backend,
            'axes.labelsize': 'large',
            'figure.figsize': (7., 5.5),
            'figure.subplot.top': 0.92,
            'figure.subplot.bottom': 0.12,
            'figure.subplot.right': 0.93,
            'figure.subplot.left': 0.12,
        })
        matplotlib.use(ops.backend)

    def _get_data(self, fn):
        import os
        ops, args = self.opargs
        return BenchResult(dirname=os.path.join(ops.droot, fn))
    def _get_data_all(self):
        import os, sys
        import cPickle as pickle
        ops, args = self.opargs
        dfn = os.path.join(ops.droot, os.path.basename(ops.droot)+'.pickle')
        # load if pickled.
        if os.path.exists(dfn):
            buc = pickle.load(open(dfn, 'rb'))
            sys.stdout.write('loaded from %s.\n' % dfn)
            return buc
        # get data.
        buc = dict()
        for msh, npclst in (
                (112, [0, 1, 4]),
                (50, [4, 8, 11]),
                (35, [11, 20, 34]),
                (28, [34, 66, 128, 256]),
        ):
            cnr = buc.setdefault(msh, dict())
            for npc in npclst:
                cnr[npc] = self._get_data('pjcf_%d_p%d' % (
                    msh, npc))
        # store.
        pickle.dump(buc, open(dfn, 'wb'), protocol=-1)
        return buc

    def _plot_weak(self, buc, ax, lloc=None):
        from numpy import array
        xdata = array([1033920, 11842380, 34214667, 66791129],
            dtype='float64')/1.e6
        isym = 0
        ydata = [buc[msh][npc].perf[-2] for msh, npc in [
            (112, 1), (50, 11), (35, 34), (28, 66)]]
        ax.plot(xdata, ydata, '-o')
        ax.set_ylim([0, max(ydata)*1.1])
        ax.set_title('Weak Scaling')
        ax.set_xlabel('Total mesh size (million element)')
        ax.set_ylabel('Performance per node (Meps)')
        ax.set_xticks(xdata)
        ax.set_xticklabels(['%.2f'%val for val in xdata])
        ax.grid()

    def _splot_strong(self, buc, ax, rel, lloc=None):
        from numpy import array
        ops, args = self.opargs
        cpn = ops.cpn
        mshmap = {112: 1033920, 50: 11842380, 35: 34214667, 28: 66791129}
        pref = buc[112][0].perf[3]
        if rel:
            ideal = [1, cpn, 34*cpn, 128*cpn, 256*cpn]
        else:
            ideal = [1, pref*cpn, pref*34*cpn, pref*128*cpn, pref*256*cpn]
        ax.plot(array([0, 1, 34, 128, 256])*cpn, ideal, '--k', label='Ideal')
        # calculate and plot data.
        isym = 0
        for msh in 112, 50, 35, 28:
            xdata = array(sorted(buc[msh].keys()))
            if rel:
                ydata = [buc[msh][npc].perf[3]/pref for npc in xdata]
                xdata *= cpn
                if xdata[0] == 0:
                    xdata[0] = 1
                ax.loglog(xdata, ydata, '-'+self.SYMBS[isym],
                    label='%d M elems'%int(mshmap[msh]/1.e6), basex=2, basey=2)
            else:
                ydata = [buc[msh][npc].perf[3] for npc in xdata]
                ax.plot(xdata*cpn, ydata, '-'+self.SYMBS[isym],
                    label='%d M elems'%int(mshmap[msh]/1.e6))
            isym += 1
        ax.set_xlim([0, 1024])
        ax.set_title('Strong Scaling')
        ax.set_xlabel('Number of cores')
        if rel:
            ax.set_ylabel('Speed Up (times)')
        else:
            ax.set_ylabel('Performance (Meps)')
        ax.legend(loc=lloc)
    def _plot_strong(self, buc, ax, lloc=None):
        self._splot_strong(buc, ax, False, lloc=lloc)
    def _plot_speedup(self, buc, ax, lloc=None):
        self._splot_strong(buc, ax, True, lloc=lloc)

    def _plot_eff(self, buc, ax, lloc=None):
        from numpy import array
        ops, args = self.opargs
        cpn = ops.cpn
        mshmap = {112: 1033920, 50: 11842380, 35: 34214667, 28: 66791129}
        pref = buc[112][0].perf[3]
        # calculate and plot data.
        isym = 0
        for msh in 112, 50, 35, 28:
            xdata = array(sorted(buc[msh].keys()))
            ydata = [buc[msh][npc].perf[-2]/pref*100/cpn for npc in xdata]
            xdata *= cpn
            if xdata[0] == 0:
                xdata[0] = 1
                ydata[0] *= cpn
            ax.semilogx(xdata, ydata, '-'+self.SYMBS[isym],
                label='%d M elems'%int(mshmap[msh]/1.e6), basex=2)
            isym += 1
        ax.set_xlim([0, 1024])
        ax.set_ylim([0, 105])
        ax.set_title('Parallel Efficiency')
        ax.set_xlabel('Number of cores')
        ax.set_ylabel('Efficiency (%)')
        ax.grid()
        ax.legend(loc=lloc)

    def __call__(self):
        import os, sys
        ops, args = self.opargs
        self._init_mpl()
        from matplotlib import pyplot as plt
        buc = self._get_data_all()
        # plot weak.
        for pname in 'weak', 'strong', 'speedup', 'eff':
            fig = plt.figure()
            func = getattr(self, '_plot_'+pname)
            func(buc, fig.add_subplot(1, 1, 1), lloc=ops.lloc)
            figfn = os.path.join(ops.droot, 'pjcf_%s.png'%pname)
            fig.savefig(figfn)
            sys.stdout.write('Figure %s written.\n' % figfn)
