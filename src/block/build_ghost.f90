!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! subroutine build_ghost: Build all information for ghost cells by 
! mirroring information from interior cells.  The action includes:
!   1. define indices and build connectivities for ghost nodes, faces, 
!      and cells.  In the same loop, mirror the coordinates of interior 
!      nodes to ghost nodes.
!   2. compute center coordinates for faces for ghost cells.
!   3. compute normal vectors and areas for faces for ghost cells.
!   4. compute center coordinates for ghost cells.
!   5. compute volume for ghost cells.
! NOTE: all the metric, type and connnectivities data passed in this 
!   subroutine are SHARED arrays rather than interior arrays.  The 
!   indices for ghost information should be carefully treated.  All the 
!   ghost indices are negative in shared arrays.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine build_ghost(msh, bndfcs, fctpn, cltpn, clgrp, &
    fcnds, fccls, clnds, clfcs, &
    ndcrd, fccncrd, fcnormal, fcarea, clcncrd, clvol)
implicit none
include 'mesh.inc'
! meta data.
type(mesh), intent(in) :: msh
integer(4), intent(in) :: bndfcs(0:1, 0:msh%ngstcell-1)
integer(4), intent(inout) :: fctpn(-msh%ngstface:msh%nface-1)
integer(4), intent(inout) :: cltpn(-msh%ngstcell:msh%ncell-1)
integer(4), intent(inout) :: clgrp(-msh%ngstcell:msh%ncell-1)
! connectivity.
integer(4), intent(inout) :: fcnds(0:msh%fcmnd, -msh%ngstface:msh%nface-1)
integer(4), intent(inout) :: fccls(0:3, -msh%ngstface:msh%nface-1)
integer(4), intent(inout) :: clnds(0:msh%clmnd, -msh%ngstcell:msh%ncell-1)
integer(4), intent(inout) :: clfcs(0:msh%clmfc, -msh%ngstcell:msh%ncell-1)
! geometry.
real(FPKIND), intent(inout) :: ndcrd(0:msh%ndim-1, -msh%ngstnode:msh%nnode-1)
real(FPKIND), intent(inout) :: fccncrd(0:msh%ndim-1, -msh%ngstface:msh%nface-1)
real(FPKIND), intent(inout) :: fcnormal(0:msh%ndim-1, -msh%ngstface:msh%nface-1)
real(FPKIND), intent(inout) :: fcarea(-msh%ngstface:msh%nface-1)
real(FPKIND), intent(inout) :: clcncrd(0:msh%ndim-1,-msh%ngstcell:msh%ncell-1)
real(FPKIND), intent(inout) :: clvol(-msh%ngstcell:msh%ncell-1)

integer(4), allocatable :: gstndmap(:)

integer(4) :: nnd, nfc
real(FPKIND) :: ndist, vol
real(FPKIND) :: lvec(0:msh%ndim-1), radvec(0:msh%ndim-1, 0:msh%fcmnd-1)

integer(4) :: fcnnd, clnnd, clnfc
integer(4) :: igcl, igfc, ignd
integer(4) :: iicl, ibfc
integer(4) :: ifc, ind, icl
integer(4) :: it, itt

integer(4) :: mk_found

allocate(gstndmap(0:msh%nnode-1))
gstndmap(:) = msh%nnode ! set to the least impossible value.

! create ghost entities and build connectivities and by the way mirror node 
! coordinate.
ignd = -1
igfc = -1
do igcl = -1, -msh%ngstcell, -1
    ibfc = bndfcs(0,-igcl-1)    ! indicate current boundary face.
    iicl = fccls(0,ibfc)    ! indicate current interior cell.
    fcnnd = fcnds(0,ibfc)   ! number of nodes in current boundary face.
    clnnd = clnds(0,iicl)   ! number of nodes in current interior cell.
    clnfc = clfcs(0,iicl)   ! number of faces in current interior cell.
    ! copy cell type.
    cltpn(igcl) = cltpn(iicl)
    ! copy cell group.
    clgrp(igcl) = clgrp(iicl)
    ! process node list in ghost cell.
    clnds(:,igcl) = clnds(:,iicl)   ! copy nodes from current interior cell.
    loop_clnnd: do it = 1, clnnd
        ind = clnds(it,iicl)
        ! try to find the node in boundary face.
        mk_found = 0
        do itt = 1, fcnnd
            if (ind .eq. fcnds(itt,ibfc)) then
                mk_found = 1
                exit
            end if
        end do
        ! if it's not in boundary face, it should be a ghost node.
        if (mk_found .eq. 0) then
            gstndmap(ind) = ignd    ! record map for faces processing.
            clnds(it,igcl) = ignd   ! save to clnds.
            ! mirror coordinate of ghost cell.
            ! NOTE: fcnormal always point outward.
            lvec(:) = fccncrd(:,ibfc) - ndcrd(:,ind)
            ndist = dot_product(lvec(:), fcnormal(:,ibfc))
            ndcrd(:,ignd) = ndcrd(:,ind) + 2*ndist*fcnormal(:,ibfc)
            ! decrement ghost node counter.
            ignd = ignd - 1
        end if
    end do loop_clnnd
    ! set the relating cell as ghost cell.
    fccls(1,ibfc) = igcl
    ! process face list in ghost cell.
    clfcs(:,igcl) = clfcs(:,iicl)   ! copy faces in current interior cell.
    loop_clnfc: do it = 1, clnfc
        ifc = clfcs(it,igcl)        ! indicate face to be processed.
        if (ifc .eq. ibfc) cycle    ! if boundary face then skip.
        fctpn(igfc) = fctpn(ifc)    ! copy face type.
        fccls(0,igfc) = igcl        ! save to fccls.
        clfcs(it,igcl) = igfc       ! save to clfcs.
        ! face-to-node connectivity.
        fcnds(:,igfc) = fcnds(:,ifc)    ! copy from interior face.
        do itt = 1, fcnds(0,igfc)
            ind = fcnds(itt,igfc)
            if (gstndmap(ind) .eq. msh%nnode) cycle ! no map data: on bnd face.
            fcnds(itt,igfc) = gstndmap(ind)     ! save to fcnds.
        end do
        ! decrement ghost face counter.
        igfc = igfc - 1
    end do loop_clnfc
    ! erase node map record.
    do it = 1, clnnd
        gstndmap(clnds(it,iicl)) = msh%nnode
    end do
end do
! tests.
!if (ignd .ne. -ngstnode-1) print*, "trouble about node:", ignd, ngstnode
!if (igfc .ne. -ngstface-1) print*, "trouble about face:", igfc, ngstface
!do it = 0, nnode-1
!    if (gstndmap(it) .ne. nnode) print*, "trouble about map:", it, gstndmap(it)
!end do

! To this place, all ghost connectivities are computed.
! Start compute ghost metrics.

! compute ghost face center coordinate.
fccncrd(:,-msh%ngstface:-1) = 0.d0
do ifc = -1, -msh%ngstface, -1  ! NOTE: indicate ghost face.
    nnd = fcnds(0,ifc)
    do it = 1, nnd
        ind = fcnds(it,ifc)
        fccncrd(:,ifc) = fccncrd(:,ifc) + ndcrd(:,ind)
    end do
    fccncrd(:,ifc) = fccncrd(:,ifc) / nnd
end do

! compute face normal vector and face area.
fcnormal(:,-msh%ngstface:-1) = 0.d0
if (msh%ndim .eq. 2) then
    do ifc = -1, -msh%ngstface, -1  ! NOTE: indicate ghost face.
        nnd = 2 ! face must be a line.
        lvec(:) = ndcrd(:,fcnds(2,ifc)) - ndcrd(:,fcnds(1,ifc))
        fcnormal(0,ifc) = lvec(1)
        fcnormal(1,ifc) = -lvec(0)
        ! compute face area.
        fcarea(ifc) = sqrt(sum(fcnormal(:,ifc)**2))
        ! make face normal unitized.
        fcnormal(:,ifc) = fcnormal(:,ifc)/fcarea(ifc)
    end do
else if (msh%ndim .eq. 3) then
    do ifc = -1, -msh%ngstface, -1  ! NOTE: indicate ghost face.
        nnd = fcnds(0,ifc)
        ! compute radial vector.
        do it = 0, nnd-1    ! NOTE: iterator starts from 0.
            ind = fcnds(it+1,ifc)
            radvec(:,it) = ndcrd(:,ind) - fccncrd(:,ifc)
        end do
        ! compute cross product, summation over all subtriangle.
        ! NOTE: this requires specific ordering in face node definition.
        fcnormal(0,ifc) = &
            radvec(1,nnd-1)*radvec(2,0) - radvec(2,nnd-1)*radvec(1,0)
        fcnormal(1,ifc) = &
            radvec(2,nnd-1)*radvec(0,0) - radvec(0,nnd-1)*radvec(2,0)
        fcnormal(2,ifc) = &
            radvec(0,nnd-1)*radvec(1,0) - radvec(1,nnd-1)*radvec(0,0)
        do it = 1, nnd-1
            fcnormal(0,ifc) = fcnormal(0,ifc) + &
                radvec(1,it-1)*radvec(2,it) - radvec(2,it-1)*radvec(1,it)
            fcnormal(1,ifc) = fcnormal(1,ifc) + &
                radvec(2,it-1)*radvec(0,it) - radvec(0,it-1)*radvec(2,it)
            fcnormal(2,ifc) = fcnormal(2,ifc) + &
                radvec(0,it-1)*radvec(1,it) - radvec(1,it-1)*radvec(0,it)
        end do
        ! compute face area*2.
        fcarea(ifc) = sqrt(sum(fcnormal(:,ifc)**2))
        ! make face normal unitized.
        fcnormal(:,ifc) = fcnormal(:,ifc)/fcarea(ifc)
        ! get face area.
        fcarea(ifc) = fcarea(ifc)/2.d0
    end do
end if

! compute center point coordinate for each ghost cell.
clcncrd(:,-msh%ngstcell:-1) = 0.d0
do icl = -1, -msh%ngstcell, -1
    nnd = clnds(0,icl)
    do it = 1, nnd
        ind = clnds(it, icl)
        clcncrd(:,icl) = clcncrd(:,icl) + ndcrd(:,ind)
    end do
    clcncrd(:,icl) = clcncrd(:,icl)/nnd
end do

! compute volume for each ghost cell.
clvol(-msh%ngstcell:-1) = 0.d0
do icl = -1, -msh%ngstcell, -1
    nfc = clfcs(0,icl)
    do it = 1, nfc
        ifc = clfcs(it, icl)
        ! calculate volume associated with each bounding face.
        lvec(:) = fccncrd(:,ifc) - clcncrd(:,icl)
        vol = dot_product(lvec(:), fcnormal(:,ifc)) * fcarea(ifc)
        ! check if the normal vector for ghost face point outward.
        if (vol .lt. 0.d0) then
            if (ifc .lt. 0) then
                fcnormal(:,ifc) = -fcnormal(:,ifc)
            end if
            vol = -vol
        end if
        ! accumulate the volume for icell.
        clvol(icl) = clvol(icl) + vol
    end do
    clvol(icl) = clvol(icl)/msh%ndim
end do

deallocate(gstndmap)

end subroutine build_ghost
! vim:set nu et tw=80 ts=4 sw=4 cino=>4:
