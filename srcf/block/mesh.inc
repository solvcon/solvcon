! Copyright (C) 2008-2010 Yung-Yu Chen <yyc@solvcon.net>.
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

! geometrical informations for mesh.
type mesh
    ! number of axis, number of equations; ndimgeo, neq.
    integer(4) :: ndim
    ! max number of nodes in a face; maxnvf.
    integer(4) :: fcmnd
    ! max number of nodes in a cell; maxnvc.
    integer(4) :: clmnd
    ! max number of faces in a cell; maxnfc.
    integer(4) :: clmfc
    ! number of nodes, faces, cells; nv, nf, nc.
    integer(4) :: nnode, nface, ncell
    ! number of boundary faces; nbf.
    integer(4) :: nbound
    ! number of ghost nodes, faces, cells; nbf, 2*nbf, nbf.
    integer(4) :: ngstnode, ngstface, ngstcell
end type mesh
! vim: set ft=fortran ff=unix et ts=4:
