! Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.
subroutine calc_soln(msh, exn, clvol, sol, soln)
implicit none
include 'ctypes.inc'
! arguments.
type(mesh), intent(in) :: msh
type(execution), intent(in) :: exn
real(FPKIND), intent(in) :: clvol(-msh%ngstcell:msh%ncell-1)
real(FPKIND), intent(in) :: sol(0:exn%neq-1, -msh%ngstcell:msh%ncell-1)
real(FPKIND), intent(out) :: soln(0:exn%neq-1, -msh%ngstcell:msh%ncell-1)
! locals.
real(FPKIND) :: hdt, qdt
integer(4) :: icl
hdt = exn%time_increment / 2.0
do icl = 0, msh%ncell-1
    soln(:,icl) = sol(:,icl) + clvol(icl) * hdt
end do
end subroutine calc_soln
! vim: set ts=4 et:
