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

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! subroutine get_faces_from_cells
!
! Extract interier faces from node list of cells.  Subroutine is designed to 
! handle all types of cell.  See block.py for the types to be supported.
! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine get_faces_from_cells(nnd, ncl, max_nfc, &
    max_clnnd, max_clnfc, max_fcnnd, cltpn, clnds, &
    nfc, clfcs, fctpn, fcnds, fccls)
implicit none
! dummy parameters.
integer(4), intent(in) :: nnd, ncl, max_nfc, max_clnnd, max_clnfc, max_fcnnd
integer(4), intent(in) :: cltpn(0:ncl-1)
integer(4), intent(in) :: clnds(0:max_clnnd, 0:ncl-1)
integer(4), intent(out) :: nfc
integer(4), intent(out) :: clfcs(0:max_clnfc, 0:ncl-1)
integer(4), intent(out) :: fctpn(0:max_nfc-1)
integer(4), intent(out) :: fcnds(0:max_fcnnd, 0:max_nfc-1)
integer(4), intent(out) :: fccls(0:3, 0:max_nfc-1)

! local buffers.
integer(4), allocatable :: ndnfc(:), ndfcs(:,:), map(:), map2(:)
! local variables.
integer(4) :: tpnicl, max_ndnfc
integer(4) :: ndstcl(0:max_clnnd), ndsfi(0:max_fcnnd), ndsfj(0:max_fcnnd)
integer(4) :: cond
! iterator.
integer(4) :: icl, ifc, jfc, ind, it, itf, jtf

! extract face definition from the node list of cells.
ifc = -1
loop_cell: do icl = 0, ncl-1
    tpnicl = cltpn(icl) ! type number of cell with index icl.
    ndstcl(:) = clnds(:,icl)
    if (tpnicl .eq. 0) then
    else if (tpnicl .eq. 1) then    ! line/edge.
        ! extract 2 points from a line.
        clfcs(0,icl) = 2    ! set the number of faces.
        fctpn(ifc+1:ifc+2) = 0  ! set face type to point.
        fcnds(0,ifc+1:ifc+2) = 1    ! set the number of nodes.
        ifc = ifc + 1
        clfcs(1,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        ifc = ifc + 1
        clfcs(2,icl) = ifc
        fcnds(1,ifc) = clnds(2,icl)
    else if (tpnicl .eq. 2) then    ! quadrilateral.
        ! extract 3 lines from a triangle.
        clfcs(0,icl) = 4    ! set the number of faces.
        fctpn(ifc+1:ifc+4) = 1  ! set face type to line.
        fcnds(0,ifc+1:ifc+4) = 2    ! set the number of nodes.
        ifc = ifc + 1
        clfcs(1,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(2,icl)
        ifc = ifc + 1
        clfcs(2,icl) = ifc
        fcnds(1,ifc) = clnds(2,icl)
        fcnds(2,ifc) = clnds(3,icl)
        ifc = ifc + 1
        clfcs(3,icl) = ifc
        fcnds(1,ifc) = clnds(3,icl)
        fcnds(2,ifc) = clnds(4,icl)
        ifc = ifc + 1
        clfcs(4,icl) = ifc
        fcnds(1,ifc) = clnds(4,icl)
        fcnds(2,ifc) = clnds(1,icl)
    else if (tpnicl .eq. 3) then    ! triangle.
        ! extract 3 lines from a triangle.
        clfcs(0,icl) = 3    ! set the number of faces.
        fctpn(ifc+1:ifc+3) = 1  ! set face type to line.
        fcnds(0,ifc+1:ifc+3) = 2    ! set the number of nodes.
        ifc = ifc + 1
        clfcs(1,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(2,icl)
        ifc = ifc + 1
        clfcs(2,icl) = ifc
        fcnds(1,ifc) = clnds(2,icl)
        fcnds(2,ifc) = clnds(3,icl)
        ifc = ifc + 1
        clfcs(3,icl) = ifc
        fcnds(1,ifc) = clnds(3,icl)
        fcnds(2,ifc) = clnds(1,icl)
    else if (tpnicl .eq. 4) then    ! hexahedron/brick.
        ! extract 6 quadrilaterals from a hexahedron.
        clfcs(0,icl) = 6    ! set number of face for this cell.
        fctpn(ifc+1:ifc+6) = 2  ! set face type to quadrilateral.
        fcnds(0,ifc+1:ifc+6) = 4    ! set number of nodes for each face.
        ifc = ifc + 1
        clfcs(1,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(4,icl)
        fcnds(3,ifc) = clnds(3,icl)
        fcnds(4,ifc) = clnds(2,icl)
        ifc = ifc + 1
        clfcs(2,icl) = ifc
        fcnds(1,ifc) = clnds(2,icl)
        fcnds(2,ifc) = clnds(3,icl)
        fcnds(3,ifc) = clnds(7,icl)
        fcnds(4,ifc) = clnds(6,icl)
        ifc = ifc + 1
        clfcs(3,icl) = ifc
        fcnds(1,ifc) = clnds(5,icl)
        fcnds(2,ifc) = clnds(6,icl)
        fcnds(3,ifc) = clnds(7,icl)
        fcnds(4,ifc) = clnds(8,icl)
        ifc = ifc + 1
        clfcs(4,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(5,icl)
        fcnds(3,ifc) = clnds(8,icl)
        fcnds(4,ifc) = clnds(4,icl)
        ifc = ifc + 1
        clfcs(5,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(2,icl)
        fcnds(3,ifc) = clnds(6,icl)
        fcnds(4,ifc) = clnds(5,icl)
        ifc = ifc + 1
        clfcs(6,icl) = ifc
        fcnds(1,ifc) = clnds(3,icl)
        fcnds(2,ifc) = clnds(4,icl)
        fcnds(3,ifc) = clnds(8,icl)
        fcnds(4,ifc) = clnds(7,icl)
    else if (tpnicl .eq. 5) then    ! tetrahedron.
        ! extract 4 triangles from a tetrahedron.
        clfcs(0,icl) = 4    ! set number of face for this cell.
        fctpn(ifc+1:ifc+4) = 3  ! set face type to triangle.
        fcnds(0,ifc+1:ifc+4) = 3    ! set number of nodes for each face.
        ifc = ifc + 1
        clfcs(1,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(3,icl)
        fcnds(3,ifc) = clnds(2,icl)
        ifc = ifc + 1
        clfcs(2,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(2,icl)
        fcnds(3,ifc) = clnds(4,icl)
        ifc = ifc + 1
        clfcs(3,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(4,icl)
        fcnds(3,ifc) = clnds(3,icl)
        ifc = ifc + 1
        clfcs(4,icl) = ifc
        fcnds(1,ifc) = clnds(2,icl)
        fcnds(2,ifc) = clnds(3,icl)
        fcnds(3,ifc) = clnds(4,icl)
    else if (tpnicl .eq. 6) then    ! prism/wedge.
        ! extract 2 triangles and 3 quadrilaterals from a tetrahedron.
        clfcs(0,icl) = 5    ! set number of face for this cell.
        fctpn(ifc+1:ifc+2) = 3  ! set face type to triangle.
        fcnds(0,ifc+1:ifc+2) = 3    ! set number of nodes for each face.
        fctpn(ifc+3:ifc+5) = 2  ! set face type to quadrilateral.
        fcnds(0,ifc+3:ifc+5) = 4    ! set number of nodes for each face.
        ifc = ifc + 1
        clfcs(1,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(2,icl)
        fcnds(3,ifc) = clnds(3,icl)
        ifc = ifc + 1
        clfcs(2,icl) = ifc
        fcnds(1,ifc) = clnds(4,icl)
        fcnds(2,ifc) = clnds(6,icl)
        fcnds(3,ifc) = clnds(5,icl)
        ifc = ifc + 1
        clfcs(3,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(4,icl)
        fcnds(3,ifc) = clnds(5,icl)
        fcnds(4,ifc) = clnds(2,icl)
        ifc = ifc + 1
        clfcs(4,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(3,icl)
        fcnds(3,ifc) = clnds(6,icl)
        fcnds(4,ifc) = clnds(4,icl)
        ifc = ifc + 1
        clfcs(5,icl) = ifc
        fcnds(1,ifc) = clnds(2,icl)
        fcnds(2,ifc) = clnds(5,icl)
        fcnds(3,ifc) = clnds(6,icl)
        fcnds(4,ifc) = clnds(3,icl)
    else if (tpnicl .eq. 7) then    ! pyramid.
        ! extract 4 triangles and 1 quadrilateral from a tetrahedron.
        clfcs(0,icl) = 5    ! set number of face for this cell.
        fctpn(ifc+1:ifc+4) = 3  ! set face type to triangle.
        fcnds(0,ifc+1:ifc+4) = 3    ! set number of nodes for each face.
        fctpn(ifc+5) = 2    ! set face type to quadrilateral.
        fcnds(0,ifc+5) = 4      ! set number of nodes for each face.
        ifc = ifc + 1
        clfcs(1,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(5,icl)
        fcnds(3,ifc) = clnds(4,icl)
        ifc = ifc + 1
        clfcs(2,icl) = ifc
        fcnds(1,ifc) = clnds(2,icl)
        fcnds(2,ifc) = clnds(5,icl)
        fcnds(3,ifc) = clnds(1,icl)
        ifc = ifc + 1
        clfcs(3,icl) = ifc
        fcnds(1,ifc) = clnds(3,icl)
        fcnds(2,ifc) = clnds(5,icl)
        fcnds(3,ifc) = clnds(2,icl)
        ifc = ifc + 1
        clfcs(4,icl) = ifc
        fcnds(1,ifc) = clnds(4,icl)
        fcnds(2,ifc) = clnds(5,icl)
        fcnds(3,ifc) = clnds(3,icl)
        ifc = ifc + 1
        clfcs(5,icl) = ifc
        fcnds(1,ifc) = clnds(1,icl)
        fcnds(2,ifc) = clnds(4,icl)
        fcnds(3,ifc) = clnds(3,icl)
        fcnds(4,ifc) = clnds(2,icl)
    end if
end do loop_cell

! build the hash table, to know what faces connect to each node.
! first pass: get the maximum number of faces.
allocate(ndnfc(0:nnd-1))
ndnfc(:) = 0
loop_count_ndnfc: do ifc = 0, max_nfc-1
    do it = 1, fcnds(0,ifc)
        ind = fcnds(it, ifc)   ! node of interest.
        ndnfc(ind) = ndnfc(ind) + 1 ! increment counting.
    end do
end do loop_count_ndnfc
max_ndnfc = maxval(ndnfc)   ! get the maximum.
deallocate(ndnfc)
! second pass: scan again to build hash table.
allocate(ndfcs(0:max_ndnfc, 0:nnd-1))
ndfcs(:,:) = -1
ndfcs(0,:) = 0
loop_count_ndfcs: do ifc = 0, max_nfc-1
    do it = 1, fcnds(0,ifc)
        ind = fcnds(it, ifc)   ! node of interest.
        ndfcs(ndfcs(0,ind)+1, ind) = ifc  ! collect face.
        ndfcs(0, ind) = ndfcs(0, ind) + 1   ! increment counting.
    end do
end do loop_count_ndfcs

! scan for duplicated faces and build mapping.
allocate(map(0:max_nfc-1))
do ifc = 0, max_nfc-1
    map(ifc) = ifc
end do
loop_find_duplicated: do ifc = 0, max_nfc-1
    if (map(ifc) .ne. ifc) cycle
    ndsfi(:) = fcnds(:,ifc)
    loop_connected_faces: do it = 1, ndfcs(0, ndsfi(1))
        jfc = ndfcs(it, ndsfi(1))
        ! test for duplication.
        if (jfc .eq. ifc) cycle
        if (fctpn(jfc) .ne. fctpn(ifc)) cycle
        ndsfj(:) = fcnds(:,jfc)
        cond = ndsfj(0)
        loop_j: do jtf = 1, ndsfj(0)
            loop_i: do itf = 1, ndsfi(0)
                if (ndsfj(jtf) .eq. ndsfi(itf)) then
                    cond = cond-1
                    exit
                end if
            end do loop_i
        end do loop_j
        if (cond .eq. 0) then
            map(jfc) = ifc  ! record duplication.
        end if
    end do loop_connected_faces
end do loop_find_duplicated

! use mapping data to remap nodes in faces, and build renewed mapping.
allocate(map2(0:max_nfc-1))
jfc = 0
do ifc = 0, max_nfc-1
    if (map(ifc) .eq. ifc) then
        fcnds(:,jfc) = fcnds(:,ifc)
        fctpn(jfc) = fctpn(ifc)
        map2(ifc) = jfc
        jfc = jfc + 1
    else
        map2(ifc) = map2(map(ifc))
    end if
end do
nfc = jfc   ! unduplicated number of faces.

! rebuild cellsfaces and face neiboring information, according to
! face mapping.
fccls(:,:) = -1
loop_fccls: do icl = 0, ncl-1
    do it = 1, clfcs(0, icl)
        ifc = clfcs(it, icl)
        jfc = map2(ifc)
        clfcs(it, icl) = jfc
        if (fccls(0, jfc) .eq. -1) then
            fccls(0, jfc) = icl
        else if (fccls(1, jfc) .eq. -1) then
            fccls(1, jfc) = icl
        end if
    end do
end do loop_fccls

deallocate(map2)
deallocate(map)
deallocate(ndfcs)

end subroutine get_faces_from_cells
! vim:set nu et tw=80 ts=4 sw=4 cino=>4:
