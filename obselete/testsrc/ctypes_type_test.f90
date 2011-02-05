subroutine ctypes_type_test(par_t, ret_t)
use iso_c_binding
implicit none
include 'ctypes.inc'
! arguments.
type(record), intent(in) :: par_t
type(record), intent(out) :: ret_t
! locals.
real*8, pointer, dimension(:) :: locarr
real*8, allocatable, dimension(:) :: testarr
ret_t%idx = -par_t%idx
ret_t%arr(:) = -par_t%arr(:)
call c_f_pointer(par_t%arr_ptr, locarr, [par_t%idx])
locarr(:) = locarr(:) + 200.d0
end subroutine ctypes_type_test
