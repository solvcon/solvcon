! Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! subroutine calc_metric: Calculate all metric information, including:
!   1. center of faces.
!   2. unit normal and area of faces.
!   3. center of cells.
!   4. volume of cells.
!   And fcnds could be reordered.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine calc_metric(msh, ndcrd, fccls, clnds, clfcs, &
    fcnds, fccncrd, fcnormal, fcarea, clcncrd, clvol)
implicit none
include 'mesh.inc'
! input.
type(mesh), intent(in) :: msh
real(FPKIND), intent(in) :: ndcrd(0:msh%ndim-1, 0:msh%nnode)
integer(4), intent(in) :: fccls(0:3, 0:msh%nface-1)
integer(4), intent(in) :: clnds(0:msh%clmnd, 0:msh%ncell-1)
integer(4), intent(in) :: clfcs(0:msh%clmfc, 0:msh%ncell-1)
! output.
integer(4), intent(inout) :: fcnds(0:msh%fcmnd, 0:msh%nface-1)
real(FPKIND), intent(inout) :: fccncrd(0:msh%ndim-1, 0:msh%nface-1)
real(FPKIND), intent(inout) :: fcnormal(0:msh%ndim-1, 0:msh%nface-1)
real(FPKIND), intent(inout) :: fcarea(0:msh%nface-1)
real(FPKIND), intent(inout) :: clcncrd(0:msh%ndim-1, 0:msh%ncell-1)
real(FPKIND), intent(inout) :: clvol(0:msh%ncell-1)

! vectors.
real(FPKIND) :: lvec(0:msh%ndim-1), radvec(0:msh%ndim-1, 0:msh%fcmnd-1)
! scalars.
real(FPKIND) :: vol
! lists.
real(FPKIND) :: nds_tf(0:msh%fcmnd-1)   ! FIXME: wrong data type.
! number holders.
integer(4) :: nnd, nfc
! iterators.
integer(4) :: ind, ifc, icl
integer(4) :: it

! compute face center coordinate.
fccncrd(:,:) = 0.d0
do ifc = 0, msh%nface-1
    nnd = fcnds(0,ifc)
    do it = 1, nnd
        ind = fcnds(it,ifc)
        fccncrd(:,ifc) = fccncrd(:,ifc) + ndcrd(:,ind)
    end do
    fccncrd(:,ifc) = fccncrd(:,ifc) / nnd
end do

! compute face normal vector, and face area.
fcnormal(:,:) = 0.d0
if (msh%ndim .eq. 2) then
    do ifc = 0, msh%nface-1
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
    do ifc = 0, msh%nface-1
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

! compute center point coordinate for each cell.
clcncrd(:,:) = 0.d0
do icl = 0, msh%ncell-1
    nnd = clnds(0,icl)
    do it = 1, nnd
        ind = clnds(it, icl)
        clcncrd(:,icl) = clcncrd(:,icl) + ndcrd(:,ind)
    end do
    clcncrd(:,icl) = clcncrd(:,icl)/nnd
end do

! compute volume for each cell, and judge the orientation of 
! faces by the way.
clvol(:) = 0.d0
do icl = 0, msh%ncell-1
    nfc = clfcs(0,icl)
    do it = 1, nfc
        ifc = clfcs(it, icl)
        ! calculate volume associated with each bounding face.
        lvec(:) = fccncrd(:,ifc) - clcncrd(:,icl)
        vol  = dot_product(lvec(:), fcnormal(:,ifc)) * fcarea(ifc)
        ! check if need to reorder node definition and connecting cell list for 
        ! the face.
        if (vol .lt. 0.d0) then
            if (fccls(0,ifc) .eq. icl) then
                nnd = fcnds(0,ifc)
                nds_tf(0:nnd-1) = fcnds(nnd:1:-1,ifc)
                fcnds(1:nnd,ifc) = nds_tf(0:nnd-1)
                fcnormal(:,ifc) = -fcnormal(:,ifc)
            end if
            vol = -vol
        else
            if (fccls(0,ifc) .ne. icl) then
                nnd = fcnds(0,ifc)
                nds_tf(0:nnd-1) = fcnds(nnd:1:-1,ifc)
                fcnds(1:nnd,ifc) = nds_tf(0:nnd-1)
                fcnormal(:,ifc) = -fcnormal(:,ifc)
            end if
        end if
        ! accumulate the volume for icell.
        clvol(icl) = clvol(icl) + vol
    end do
end do
clvol(:) = clvol(:)/msh%ndim

end subroutine calc_metric
! vim:set nu et tw=80 ts=4 sw=4 cino=>4:
