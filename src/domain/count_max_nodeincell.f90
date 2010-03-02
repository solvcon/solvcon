subroutine count_max_nodeinblock(max_clnnd, ncell, nblk, nnode, &
    clnds, part, max_ndcnt)
implicit none
! arguments.
integer(4), intent(in) :: max_clnnd, ncell, nblk, nnode
integer(4), intent(in), dimension(0:max_clnnd, 0:ncell-1) :: clnds
integer(4), intent(in), dimension(0:ncell-1) :: part
integer(4), intent(out) :: max_ndcnt
! buffers.
integer(4), allocatable, dimension(:,:) :: blkit
integer(4), allocatable, dimension(:) :: blkcls, ndcnt, ndlcnt
! iterators.
integer(4) :: icl, iblk, it, inl, ind

! Sweep 1.
allocate(blkit(0:1, 0:nblk-1))
blkit(:,:) = 0
do icl = 0, ncell-1
    iblk = part(icl)
    blkit(1,iblk) = blkit(1,iblk) + 1
end do
do iblk = 1, nblk-1
    blkit(1,iblk) = blkit(1,iblk) + blkit(1,iblk-1)
end do

! Sweep 2.
blkit(0,0) = 0
blkit(0,1:nblk-1) = blkit(1,0:nblk-2)
allocate(blkcls(0:ncell-1))
do icl = 0, ncell-1
    iblk = part(icl)
    it = blkit(0,iblk)
    blkcls(it) = icl
    blkit(0,iblk) = it + 1
end do

! Sweep 3.
allocate(ndcnt(0:nnode-1))
allocate(ndlcnt(0:nnode-1))
ndcnt(:) = 0
blkit(0,0) = 0
blkit(0,1:nblk-1) = blkit(1,0:nblk-2)
do iblk = 0, nblk-1
    ndlcnt(:) = 0
    do it = blkit(0,iblk), blkit(1,iblk)-1
        icl = blkcls(it)
        do inl = 1, clnds(0,icl)
            ind = clnds(inl,icl)
            ndlcnt(ind) = 1
        end do
    end do
    ndcnt(:) = ndcnt(:) + ndlcnt(:)
end do
max_ndcnt = maxval(ndcnt)   ! get the maximum.

deallocate(ndlcnt)
deallocate(ndcnt)
deallocate(blkcls)
deallocate(blkit)
end subroutine count_max_nodeinblock
! vim:set nu et tw=80 ts=4 sw=4 cino=>4:
