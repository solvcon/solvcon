#!/usr/bin/python
#
# sodtubecmdp.py
#
# use command pattern to rewrite sodtube1d.py
#

class SodTube():
    """The INVOKER class"""
    @classmethod
    def execute(cls, command):
        command.execute()

class Command():
    """The COMMAND interface"""
    def __init__(self, obj):
        self._obj = obj

    def execute(self):
        raise NotImplemented

class GenGrid(Command):
    """The COMMAND for getting the grid"""
    def execute(self):
        self._obj.gen_grid()

class GetSolution(Command):
    """
    The COMMAND for getting the solution
    """
    def execute(self):
        self._obj.get_solution()

class CalAnalyticSolution(Command):
    """
    The COMMAND for caculating the solution

    TODO:
    this could be a micro command
        cal_analytic_solution
    to execute the sub commands like
        cal_solution_region1
        cal_solution_region2
        cal_solution_region3
        cal_solution_region4
        cal_solution_region5
    """
    def execute(self):
        self._obj.cal_analytic_solution()

class DumpResult(Command):
    """The COMMAND for dumping the result"""
    def execute(self):
        self._obj.dump_result()

class DumpGrid(Command):
    """The COMMAND for dumping the grid points"""
    def execute(self):
        self._obj.dump_grid()

class DumpAnalyticRegion4(Command):
    """The COMMAND for dumping the analytic solution in the region 4"""
    def execute(self):
        self._obj.dump_analytic_region4()


class Solver():
    """The RECEIVER class"""
    def __init__(self):
        self._grid = ()
        self._u = ()
        self._result = []

    def gen_grid(self):
        xstep = 100
        xstart = -5050
        xstop = 5050
        grid = []
        for x in range(xstart, xstop + xstep, xstep):
            grid.append(float(x)/10000.0)
        self._grid = tuple(grid)
        print("generated grid points")

    def get_solution(self):
        print("get solution")

    def cal_analytic_solution(self):
        print("calculate analytic solution")


    def dump_result(self):
        print("dump result")

    def dump_grid(self):
        for point in self._grid:
            print(point)
        print("grid points were dumped")

    def dump_analytic_region4(self):
        print("dump analytic solution in region4")

class SolutionClient():
    """The CLIENT class"""
    def __init__(self):
        self._solver = Solver()
        self._sodtube = SodTube()

    def invoke(self, cmd):
        cmd = cmd.strip().upper()
        if cmd == "GRID":
            self._sodtube.execute(GenGrid(self._solver))
        elif cmd == "SOLUTION":
            self._sodtube.execute(GetSolution(self._solver))
        elif cmd == "ANALYTIC":
            self._sodtube.execute(CalAnalyticSolution(self._solver))
        elif cmd == "DUMP":
            self._sodtube.execute(DumpResult(self._solver))
        elif cmd == "DUMPGRID":
            self._sodtube.execute(DumpGrid(self._solver))
        elif cmd == "DUMPANALYTICREGION4":
            self._sodtube.execute(DumpAnalyticRegion4(self._solver))
        else:
            print("No such command")

#if __name__ == "__main__":

