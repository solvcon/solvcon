! Copyright (C) 2008-2010 Yung-Yu Chen.
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 2 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

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
