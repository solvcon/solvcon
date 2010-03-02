! NOTE: it is required to use iso_c_binding while included.
type record
    integer*4 :: idx
    real*8, dimension(10) :: arr
    type(c_ptr) :: arr_ptr
end type record
! vim: set ft=fortran ts=4 et:
