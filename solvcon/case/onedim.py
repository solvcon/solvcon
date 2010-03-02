# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
One-dimensional simulation cases.
"""

from .core import BaseCase, Hook

class OnedimCase(BaseCase):
    """
    @cvar simulations: simulations belongs to this class.
    @ctype simulations: dict
    """

    defdict = {
        # execution related.
        'execution.dnx': None,
    }

    def init(self, force=False):
        """
        Load block and initialize solver from the geometry information in the
        block and conditions in the self case.  If parallel run is specified
        (throught domaintype), split the domain and perform corresponding tasks.
        """
        preres = super(OnedimCase, self).init(force=force)
        solvertype = self.solver.solvertype

        xgrid, xmtrl = self._init_grid()
        svr = self.solver.solvertype(xgrid, xmtrl,
            neq=self.execution.neq,
        )
        svr.bind()
        svr.init()
        self.solver.solverobj = svr

        self._have_init = preres and True
        return self._have_init

    def _init_grid(self):
        """
        Initialize grid.

        @return: x grid.
        @rtype: numpy.ndarray
        """
        raise NotImplementedError

    def _run(self):
        """
        Run the simulation case; time marching.

        @return: nothing.
        """
        import sys
        # start log.
        self._log_start('run', msg=' '+self.io.basefn)
        self.info("\n")
        # prepare for time marching.
        aCFL = 0.0
        self.execution.step_current = 0
        # hook: init.
        self._runhooks('preloop')
        self._log_start('loop_march', postmsg='\n')
        while self.execution.step_current < self.execution.steps_run:
            # hook: premarch.
            self._runhooks('premarch')
            cCFL = self.solver.solverobj.march(
                self.execution.step_current*self.execution.time_increment,
                self.execution.time_increment)
            # process CFL.
            istep = self.execution.step_current
            aCFL += cCFL
            mCFL = aCFL/(istep+1)
            self.execution.cCFL = cCFL
            self.execution.aCFL = aCFL
            self.execution.mCFL = mCFL
            # increment to next time step.
            self.execution.step_current += 1
            # hook: postmarch.
            self._runhooks('postmarch')
            sys.stdout.flush()
        self._log_start('loop_march')
        # hook: final.
        self._runhooks('postloop')
        # end log.
        self._log_end('run')

    def run(self):
        """
        Determine how to run the case according to do animation or not, and run
        the case.

        @return: nothing.
        """
        from matplotlib import pyplot as plt
        from matplotlib import rcParams
        # no matter how hooks say, use inner runner if specified.
        if self.execution.run_inner:
            self._run_inner()
        else:
            # if there's any figure in hooks, do gui plot.
            figs = [hk.fig for hk in self.execution.runhooks if
                hasattr(hk, 'fig') and hk.fig != None]
            if len(figs) and rcParams['backend'] == 'TkAgg':
                fig = figs[0]
                fig.canvas.manager.window.after(100, self._run)
                plt.show()
            else:
                self._run()

class OnedimHook(Hook):
    """
    Base type for hooks needing a OnedimCase.

    @ivar psteps: the interval number of steps between printing.
    @itype psteps: int
    """
    def __init__(self, case, **kw):
        assert isinstance(case, OnedimCase)
        super(OnedimHook, self).__init__(case, **kw)

class Initializer(OnedimHook):
    """
    Initializer for one-dimensional cases.  It's an abstract class.  You should
    override init() method.
    """

    def preloop(self):
        raise NotImplementedError

class Calculator(OnedimHook):
    """
    Calculator for one-dimensional cases.  It's an abstract class.  You should
    override _calculate() method.
    """

    def _calculate(self):
        raise NotImplementedError

class Plotter(OnedimHook):
    """
    Abstract Hook class providing necessary framework to plot on one single 
    figure of matplotlib.

    @ivar imgfn: image output filename; also indicate output if true.
    @itype imgfn: str
    @ivar fig: contains the matplotlib figure that needs to be draw.
    @itype fig: matplotlib.figure.Figure
    @ivar outdir: output directory.
    @itype outdir: str
    @ivar _flags: flags for showing on plot.  The keys can be accessed with
        flag_ properties.
    @itype _flags: dict
    """

    def __init__(self, case, imgfn=None, flags=None, **kw):
        """
        Initialize figure for plotting and other variables.  Prepare output
        directory.

        @param case: case object.
        @type case: OnedimCase
        @keyword imgfn: image output filename.
        @type imgfn: str
        """
        import os
        from glob import glob
        from matplotlib import pyplot as plt
        self.imgfn = imgfn
        self._flags = flags if flags else {}
        super(Plotter, self).__init__(case, **kw)
        # prepare figure.
        self.fig = plt.figure()
        # prepare output directory.
        info = self.info
        self.outdir = self.case.io.basedir
        if self.imgfn:
            # remove previously generated .png files.
            ptn = os.path.join(self.outdir, '*.png')
            info('Unlink %s ...' % ptn)
            for fn in glob(ptn):
                os.unlink(fn)
            info('\n')

    def __getattr__(self, key):
        if key.startswith('flag_'):
            return self._flags[key[5:]]
        else:
            raise AttributeError
    def __setattr__(self, key, value):
        if key.startswith('flag_'):
            self._flags[key[5:]] = value
        else:
            super(Plotter, self).__setattr__(key, value)

    @staticmethod
    def _reltext(ax, relx, rely, *args, **kw):
        """
        Create text at relative coordinate within an ax.

        @param ax: axes object.
        @type ax: matplotlib.axes.AxesSubplot
        @param relx: relative x coordinate.
        @type relx: float
        @param rely: relative y coordinate.
        @type rely: float
        @return: nothing.
        """
        assert relx >= 0 and relx <= 1
        assert rely >= 0 and rely <= 1
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        x = xmin + (xmax-xmin)*relx
        y = ymin + (ymax-ymin)*rely
        return ax.text(x, y, *args, **kw)

    @staticmethod
    def _sync_legend_linewidth(lg):
        from matplotlib import rcParams
        lg.get_frame().set_linewidth(rcParams['axes.linewidth'])
        return lg

class Movie(Plotter):
    """
    Abstract Hook class providing necessary framework to plot movie on one 
    single figure of matplotlib, and create movie using mplayer/mencoder.

    @ivar width: width of image.
    @itype width: int
    @ivar height: height of image.
    @itype height: int
    @ivar fps: frames per second.
    @itype fps: int
    @ivar snapshots: list for steps that to take snapshots; first element
        indicate the snapshot identifying name.
    @itype snapshots: list
    @ivar imgtmpl: image output filename template.
    @itype imgtmpl: str
    @ivar snptmpl: snapshot output filename template.
    @itype snptmpl: str
    """
    def __init__(self, case, **kw):
        """
        @keyword width: width of image.
        @type width: int
        @keyword height: height of image.
        @type height: int
        @keyword fps: frames per second.
        @type fps: int
        @param snapshots: list for steps that to take snapshots; first element
            indicate the snapshot identifying name.
        @type snapshots: list
        """
        from math import log, ceil
        from matplotlib import pyplot as plt
        self.fps = kw.pop('fps', 10)
        self.width = kw.pop('width', 800)
        self.height = kw.pop('height', 600)
        self.snapshots = kw.pop('snapshots', [])
        self.snapshots_exts = kw.pop('snapshots_exts', ('.png',))
        self.vcodec = kw.pop('vcodec', 'msmpeg4v2')
        super(Movie, self).__init__(case, **kw)
        # image filename template.
        digits = int(ceil(log(self.case.execution.steps_run, 10)))
        self.imgtmpl = "%s_%s_%%0%dd" % (
            self.case.io.basefn, self.imgfn, digits)
        if self.snapshots:
            self.snptmpl = "%s_%s_snapshot_%%0%dd" % (
                self.case.io.basefn, self.snapshots[0], digits)
        else:
            self.snptmpl = None
        # check for mencoder.
        if self.imgfn and not self._locate_mencoder():
            raise OSError, 'mencoder not found on system'

    @staticmethod
    def _locate_mencoder():
        """
        Locate mencoder binary.  If not found, return False.

        @return: path to mencoder.
        @rtype: str
        """
        import sys
        import os
        splitter = ';' if sys.platform.startswith('win') else ':'
        for dname in os.environ['PATH'].split(splitter):
            loc = os.path.join(dname, 'mencoder')
            if os.path.exists(loc):
                return loc
        return False

    def _redraw(self):
        """
        Update figure and save (if flagged) it into file.

        @return: nothing.
        """
        import os
        self.fig.canvas.draw()
        # save to png for movie.
        if self.imgfn:
            istep = self.case.execution.step_current
            self._makedir(self.outdir)
            self.fig.savefig(os.path.join(
                self.outdir, (self.imgtmpl+'.png')%istep))
        # save to eps/png for snapshots.
        if self.snapshots:
            istep = self.case.execution.step_current
            if istep in self.snapshots:
                self._makedir(self.outdir)
                for ext in self.snapshots_exts:
                    self.fig.savefig(os.path.join(
                        self.outdir, (self.snptmpl+ext)%istep))

    def _encode_movie(self):
        """
        Invoke mplayer/mencoder to generate movie file.

        @return: nothing.
        """
        import os
        import subprocess
        if not self.imgfn:  # don't encode if no imgfn supplied.
            return
        outputfn = os.path.join(self.case.io.outputdir, self.imgfn+'.avi')
        command = (
            'mencoder',
            'mf://%s/%s_%s_*.png' % (
                self.case.io.outputdir, self.case.io.casename, self.imgfn),
            '-mf',
            'type=png:w=%d:h=%d:fps=%d' % (self.width, self.height, self.fps),
            '-ovc',
            'lavc',
            '-lavcopts',
            'vcodec=%s' % self.vcodec,
            #'vcodec=msmpeg4v2',
            #'vcodec=libx264',
            #'vcodec=mpeg4',
            '-oac',
            'copy',
            '-o',
            outputfn
        )
        subprocess.check_call(command)
        self.info("\nMovie output to %s .\n" % outputfn)
