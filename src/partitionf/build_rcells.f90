! Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! subroutine build_rcells:
!     Build unstructured grid cell relation accroding to face-cell 
!     relation. 
!     Return rcells and rcellno.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine build_rcells(max_clnfc, nface, ncell, clfcs, fccls, rcells, rcellno)
implicit none
! arguments.
integer(4), intent(in) :: max_clnfc, nface, ncell
integer(4), intent(in), dimension(0:max_clnfc, 0:ncell-1) :: clfcs
integer(4), intent(in), dimension(0:3, 0:nface-1) :: fccls
integer(4), intent(out), dimension(0:max_clnfc-1, 0:ncell-1) :: rcells
integer(4), intent(out), dimension(0:ncell-1) :: rcellno
! iterators.
integer(4) :: icl, it, ifc
rcells  = -1
rcellno = 0
do icl = 0, ncell-1
    do it = 0, clfcs(0,icl)-1
        ifc = clfcs(it+1,icl)
        ! is not a face; shouldn't happen
        if( ifc .eq. -1 ) then
            rcells(it,icl) = -1
            cycle
        else if( fccls(0,ifc) .eq. icl ) then
            ! has neibor block
            if( fccls(2,ifc) .ne. -1 ) then
                rcells(it,icl) = -1
            ! is interior
            else
                rcells(it,icl) = fccls(1,ifc)
            end if
        ! I am the neiboring cell
        else if( fccls(1,ifc) .eq. icl ) then
            rcells(it,icl) = fccls(0,ifc)
        end if
        ! count rcell number
        if( rcells(it,icl) .ge. 0 ) then 
            rcellno(icl) = rcellno(icl) + 1
        else
            rcells(it,icl) = -1
        end if
    end do
end do
end subroutine build_rcells
! vim:set nu et tw=72 ts=4 sw=4 cino=>4:
