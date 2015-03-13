#
# Regular cron jobs for the solvcon package
#
0 4	* * *	root	[ -x /usr/bin/solvcon_maintenance ] && /usr/bin/solvcon_maintenance
