subroutine ctypes_test(par_i, par_d, n_i, par_arr_i, n_d1, n_d2, par_arr_d)
implicit none
! arguments.
integer*4, intent(in) :: par_i
real*8, intent(in) :: par_d
integer*4, intent(in) :: n_i
integer*4, intent(inout) :: par_arr_i(n_i)
integer*4, intent(in) :: n_d1, n_d2
real*8, intent(inout) :: par_arr_d(n_d1, n_d2)

par_arr_i(:) = par_arr_i(:) + par_i
par_arr_d(:,:) = par_arr_d(:,:) + par_d
par_arr_d(:,1) = par_arr_d(:,1) - par_d
end subroutine ctypes_test
