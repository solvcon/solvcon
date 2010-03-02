subroutine calc_dsoln(msh, exn, clcnd, dsol, dsoln)
implicit none
include 'ctypes.inc'
! arguments.
type(mesh), intent(in) :: msh
type(execution), intent(in) :: exn
real(FPKIND), intent(in) :: clcnd(0:msh%ndim-1, -msh%ngstcell:msh%ncell-1)
real(FPKIND), intent(in) :: &
    dsol(0:msh%ndim-1, 0:exn%neq-1, -msh%ngstcell:msh%ncell-1)
real(FPKIND), intent(out) :: &
    dsoln(0:msh%ndim-1, 0:exn%neq-1, -msh%ngstcell:msh%ncell-1)
! locals.
real(FPKIND) :: hdt
integer(4) :: icl, ieq
hdt = exn%time_increment / 2.0
interior_cells: do icl = 0, msh%ncell-1
    do ieq = 0, exn%neq-1
        dsoln(:,ieq,icl) = dsol(:,ieq,icl) + clcnd(:,icl) * hdt
    end do
end do interior_cells
end subroutine calc_dsoln
! vim: set ts=4 et:
