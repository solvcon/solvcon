! geometrical informations for mesh.
type mesh
    ! number of axis, number of equations; ndimgeo, neq.
    integer*4 :: ndim
    ! max number of nodes in a face; maxnvf.
    integer*4 :: fcmnd
    ! max number of nodes in a cell; maxnvc.
    integer*4 :: clmnd
    ! max number of faces in a cell; maxnfc.
    integer*4 :: clmfc
    ! number of nodes, faces, cells; nv, nf, nc.
    integer*4 :: nnode, nface, ncell
    ! number of boundary faces; nbf.
    integer*4 :: nbound
    ! number of ghost nodes, faces, cells; nbf, 2*nbf, nbf.
    integer*4 :: ngstnode, ngstface, ngstcell
end type mesh
! executional informations.
type execution
    integer*4 :: ncore
    integer*4 :: neq
    real*8 :: time, time_increment  ! time, time increment.
end type execution
! vim: set ft=fortran ff=unix et ts=4:
