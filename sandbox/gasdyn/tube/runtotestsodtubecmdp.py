#!/usr/bin/python
#
# runtotestsodtubecmdp.py
#
# a script to test sodtubecmdp.py
#
# usage: just run ./runtotestsodtubecmdp.py
#

import sodtubecmdp

so = sodtubecmdp.SolutionClient()

so.invoke("grid")
so.invoke("solution")
so.invoke("dump")

