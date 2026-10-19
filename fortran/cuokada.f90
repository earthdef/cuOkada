! -*- f90 -*-
!
! cuokada: elastic dislocations in a half space, on the cpu and the gpu; the fortran interface
! generated from include/cuokada/cuokada.h by tools/gen_fortran.py; do not edit
!
! Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE
!
! The functions of cuokada.h, with the same arguments and layouts: arrays are row major in c, so
! a fortran array obs(3, nobs) holds the stations, src(7, nsrc) the rectangles, slip(3, nsrc) the
! slips, and out(3, nobs) the sum of the displacements; the matrix out(3, nsrc, F, nobs). The
! cuda functions take device pointers, e.g. c_devloc(a) in cuda fortran.

module cuokada
  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_double, c_float, c_ptr
  implicit none

  interface

    integer(c_int) function cuokada_rect_disp_matrix_d(obs, nobs, src, nsrc, nu, out) bind(c, name="cuokada_rect_disp_matrix_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_double), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_double), intent(in) :: src(*)
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      real(c_double), intent(out) :: out(*)
    end function cuokada_rect_disp_matrix_d

    integer(c_int) function cuokada_rect_disp_sum_d(obs, nobs, src, slip, nsrc, nu, out) bind(c, name="cuokada_rect_disp_sum_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_double), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_double), intent(in) :: src(*)
      real(c_double), intent(in) :: slip(*)
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      real(c_double), intent(out) :: out(*)
    end function cuokada_rect_disp_sum_d

    integer(c_int) function cuokada_rect_disp_pairs_d(obs, src, slip, n, nu, out) bind(c, name="cuokada_rect_disp_pairs_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_double), intent(in) :: obs(*)
      real(c_double), intent(in) :: src(*)
      real(c_double), intent(in) :: slip(*)
      integer(c_int64_t), value :: n
      real(c_double), value :: nu
      real(c_double), intent(out) :: out(*)
    end function cuokada_rect_disp_pairs_d

    integer(c_int) function cuokada_rect_strain_matrix_d( &
        obs, nobs, src, nsrc, nu, out) &
        bind(c, name="cuokada_rect_strain_matrix_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_double), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_double), intent(in) :: src(*)
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      real(c_double), intent(out) :: out(*)
    end function cuokada_rect_strain_matrix_d

    integer(c_int) function cuokada_rect_strain_sum_d( &
        obs, nobs, src, slip, nsrc, nu, out) &
        bind(c, name="cuokada_rect_strain_sum_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_double), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_double), intent(in) :: src(*)
      real(c_double), intent(in) :: slip(*)
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      real(c_double), intent(out) :: out(*)
    end function cuokada_rect_strain_sum_d

    integer(c_int) function cuokada_rect_strain_pairs_d(obs, src, slip, n, nu, out) bind(c, name="cuokada_rect_strain_pairs_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_double), intent(in) :: obs(*)
      real(c_double), intent(in) :: src(*)
      real(c_double), intent(in) :: slip(*)
      integer(c_int64_t), value :: n
      real(c_double), value :: nu
      real(c_double), intent(out) :: out(*)
    end function cuokada_rect_strain_pairs_d

    integer(c_int) function cuokada_tri_disp_matrix_d( &
        obs, nobs, src, nsrc, nu, halfspace, out) &
        bind(c, name="cuokada_tri_disp_matrix_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_double), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_double), intent(in) :: src(*)
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      integer(c_int), value :: halfspace
      real(c_double), intent(out) :: out(*)
    end function cuokada_tri_disp_matrix_d

    integer(c_int) function cuokada_tri_disp_sum_d( &
        obs, nobs, src, slip, nsrc, nu, halfspace, out) &
        bind(c, name="cuokada_tri_disp_sum_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_double), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_double), intent(in) :: src(*)
      real(c_double), intent(in) :: slip(*)
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      integer(c_int), value :: halfspace
      real(c_double), intent(out) :: out(*)
    end function cuokada_tri_disp_sum_d

    integer(c_int) function cuokada_tri_disp_pairs_d( &
        obs, src, slip, n, nu, halfspace, out) &
        bind(c, name="cuokada_tri_disp_pairs_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_double), intent(in) :: obs(*)
      real(c_double), intent(in) :: src(*)
      real(c_double), intent(in) :: slip(*)
      integer(c_int64_t), value :: n
      real(c_double), value :: nu
      integer(c_int), value :: halfspace
      real(c_double), intent(out) :: out(*)
    end function cuokada_tri_disp_pairs_d

    integer(c_int) function cuokada_tri_strain_matrix_d( &
        obs, nobs, src, nsrc, nu, halfspace, out) &
        bind(c, name="cuokada_tri_strain_matrix_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_double), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_double), intent(in) :: src(*)
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      integer(c_int), value :: halfspace
      real(c_double), intent(out) :: out(*)
    end function cuokada_tri_strain_matrix_d

    integer(c_int) function cuokada_tri_strain_sum_d( &
        obs, nobs, src, slip, nsrc, nu, halfspace, out) &
        bind(c, name="cuokada_tri_strain_sum_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_double), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_double), intent(in) :: src(*)
      real(c_double), intent(in) :: slip(*)
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      integer(c_int), value :: halfspace
      real(c_double), intent(out) :: out(*)
    end function cuokada_tri_strain_sum_d

    integer(c_int) function cuokada_tri_strain_pairs_d( &
        obs, src, slip, n, nu, halfspace, out) &
        bind(c, name="cuokada_tri_strain_pairs_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_double), intent(in) :: obs(*)
      real(c_double), intent(in) :: src(*)
      real(c_double), intent(in) :: slip(*)
      integer(c_int64_t), value :: n
      real(c_double), value :: nu
      integer(c_int), value :: halfspace
      real(c_double), intent(out) :: out(*)
    end function cuokada_tri_strain_pairs_d

    integer(c_int) function cuokada_rect_disp_matrix_f(obs, nobs, src, nsrc, nu, out) bind(c, name="cuokada_rect_disp_matrix_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_float), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_float), intent(in) :: src(*)
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      real(c_float), intent(out) :: out(*)
    end function cuokada_rect_disp_matrix_f

    integer(c_int) function cuokada_rect_disp_sum_f(obs, nobs, src, slip, nsrc, nu, out) bind(c, name="cuokada_rect_disp_sum_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_float), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_float), intent(in) :: src(*)
      real(c_float), intent(in) :: slip(*)
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      real(c_float), intent(out) :: out(*)
    end function cuokada_rect_disp_sum_f

    integer(c_int) function cuokada_rect_disp_pairs_f(obs, src, slip, n, nu, out) bind(c, name="cuokada_rect_disp_pairs_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_float), intent(in) :: obs(*)
      real(c_float), intent(in) :: src(*)
      real(c_float), intent(in) :: slip(*)
      integer(c_int64_t), value :: n
      real(c_float), value :: nu
      real(c_float), intent(out) :: out(*)
    end function cuokada_rect_disp_pairs_f

    integer(c_int) function cuokada_rect_strain_matrix_f( &
        obs, nobs, src, nsrc, nu, out) &
        bind(c, name="cuokada_rect_strain_matrix_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_float), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_float), intent(in) :: src(*)
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      real(c_float), intent(out) :: out(*)
    end function cuokada_rect_strain_matrix_f

    integer(c_int) function cuokada_rect_strain_sum_f( &
        obs, nobs, src, slip, nsrc, nu, out) &
        bind(c, name="cuokada_rect_strain_sum_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_float), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_float), intent(in) :: src(*)
      real(c_float), intent(in) :: slip(*)
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      real(c_float), intent(out) :: out(*)
    end function cuokada_rect_strain_sum_f

    integer(c_int) function cuokada_rect_strain_pairs_f(obs, src, slip, n, nu, out) bind(c, name="cuokada_rect_strain_pairs_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_float), intent(in) :: obs(*)
      real(c_float), intent(in) :: src(*)
      real(c_float), intent(in) :: slip(*)
      integer(c_int64_t), value :: n
      real(c_float), value :: nu
      real(c_float), intent(out) :: out(*)
    end function cuokada_rect_strain_pairs_f

    integer(c_int) function cuokada_tri_disp_matrix_f( &
        obs, nobs, src, nsrc, nu, halfspace, out) &
        bind(c, name="cuokada_tri_disp_matrix_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_float), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_float), intent(in) :: src(*)
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      integer(c_int), value :: halfspace
      real(c_float), intent(out) :: out(*)
    end function cuokada_tri_disp_matrix_f

    integer(c_int) function cuokada_tri_disp_sum_f( &
        obs, nobs, src, slip, nsrc, nu, halfspace, out) &
        bind(c, name="cuokada_tri_disp_sum_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_float), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_float), intent(in) :: src(*)
      real(c_float), intent(in) :: slip(*)
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      integer(c_int), value :: halfspace
      real(c_float), intent(out) :: out(*)
    end function cuokada_tri_disp_sum_f

    integer(c_int) function cuokada_tri_disp_pairs_f( &
        obs, src, slip, n, nu, halfspace, out) &
        bind(c, name="cuokada_tri_disp_pairs_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_float), intent(in) :: obs(*)
      real(c_float), intent(in) :: src(*)
      real(c_float), intent(in) :: slip(*)
      integer(c_int64_t), value :: n
      real(c_float), value :: nu
      integer(c_int), value :: halfspace
      real(c_float), intent(out) :: out(*)
    end function cuokada_tri_disp_pairs_f

    integer(c_int) function cuokada_tri_strain_matrix_f( &
        obs, nobs, src, nsrc, nu, halfspace, out) &
        bind(c, name="cuokada_tri_strain_matrix_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_float), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_float), intent(in) :: src(*)
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      integer(c_int), value :: halfspace
      real(c_float), intent(out) :: out(*)
    end function cuokada_tri_strain_matrix_f

    integer(c_int) function cuokada_tri_strain_sum_f( &
        obs, nobs, src, slip, nsrc, nu, halfspace, out) &
        bind(c, name="cuokada_tri_strain_sum_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_float), intent(in) :: obs(*)
      integer(c_int64_t), value :: nobs
      real(c_float), intent(in) :: src(*)
      real(c_float), intent(in) :: slip(*)
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      integer(c_int), value :: halfspace
      real(c_float), intent(out) :: out(*)
    end function cuokada_tri_strain_sum_f

    integer(c_int) function cuokada_tri_strain_pairs_f( &
        obs, src, slip, n, nu, halfspace, out) &
        bind(c, name="cuokada_tri_strain_pairs_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      real(c_float), intent(in) :: obs(*)
      real(c_float), intent(in) :: src(*)
      real(c_float), intent(in) :: slip(*)
      integer(c_int64_t), value :: n
      real(c_float), value :: nu
      integer(c_int), value :: halfspace
      real(c_float), intent(out) :: out(*)
    end function cuokada_tri_strain_pairs_f

    integer(c_int) function cuokada_cuda_rect_disp_matrix_d( &
        obs, nobs, src, nsrc, nu, out, stream) &
        bind(c, name="cuokada_cuda_rect_disp_matrix_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_rect_disp_matrix_d

    integer(c_int) function cuokada_cuda_rect_disp_sum_d( &
        obs, nobs, src, slip, nsrc, nu, out, stream) &
        bind(c, name="cuokada_cuda_rect_disp_sum_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_rect_disp_sum_d

    integer(c_int) function cuokada_cuda_rect_disp_pairs_d( &
        obs, src, slip, n, nu, out, stream) &
        bind(c, name="cuokada_cuda_rect_disp_pairs_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: n
      real(c_double), value :: nu
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_rect_disp_pairs_d

    integer(c_int) function cuokada_cuda_rect_strain_matrix_d( &
        obs, nobs, src, nsrc, nu, out, stream) &
        bind(c, name="cuokada_cuda_rect_strain_matrix_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_rect_strain_matrix_d

    integer(c_int) function cuokada_cuda_rect_strain_sum_d( &
        obs, nobs, src, slip, nsrc, nu, out, stream) &
        bind(c, name="cuokada_cuda_rect_strain_sum_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_rect_strain_sum_d

    integer(c_int) function cuokada_cuda_rect_strain_pairs_d( &
        obs, src, slip, n, nu, out, stream) &
        bind(c, name="cuokada_cuda_rect_strain_pairs_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: n
      real(c_double), value :: nu
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_rect_strain_pairs_d

    integer(c_int) function cuokada_cuda_tri_disp_matrix_d( &
        obs, nobs, src, nsrc, nu, halfspace, out, stream) &
        bind(c, name="cuokada_cuda_tri_disp_matrix_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      integer(c_int), value :: halfspace
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_tri_disp_matrix_d

    integer(c_int) function cuokada_cuda_tri_disp_sum_d( &
        obs, nobs, src, slip, nsrc, nu, halfspace, out, stream) &
        bind(c, name="cuokada_cuda_tri_disp_sum_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      integer(c_int), value :: halfspace
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_tri_disp_sum_d

    integer(c_int) function cuokada_cuda_tri_disp_pairs_d( &
        obs, src, slip, n, nu, halfspace, out, stream) &
        bind(c, name="cuokada_cuda_tri_disp_pairs_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: n
      real(c_double), value :: nu
      integer(c_int), value :: halfspace
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_tri_disp_pairs_d

    integer(c_int) function cuokada_cuda_tri_strain_matrix_d( &
        obs, nobs, src, nsrc, nu, halfspace, out, stream) &
        bind(c, name="cuokada_cuda_tri_strain_matrix_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      integer(c_int), value :: halfspace
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_tri_strain_matrix_d

    integer(c_int) function cuokada_cuda_tri_strain_sum_d( &
        obs, nobs, src, slip, nsrc, nu, halfspace, out, stream) &
        bind(c, name="cuokada_cuda_tri_strain_sum_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: nsrc
      real(c_double), value :: nu
      integer(c_int), value :: halfspace
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_tri_strain_sum_d

    integer(c_int) function cuokada_cuda_tri_strain_pairs_d( &
        obs, src, slip, n, nu, halfspace, out, stream) &
        bind(c, name="cuokada_cuda_tri_strain_pairs_d")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: n
      real(c_double), value :: nu
      integer(c_int), value :: halfspace
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_tri_strain_pairs_d

    integer(c_int) function cuokada_cuda_rect_disp_matrix_f( &
        obs, nobs, src, nsrc, nu, out, stream) &
        bind(c, name="cuokada_cuda_rect_disp_matrix_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_rect_disp_matrix_f

    integer(c_int) function cuokada_cuda_rect_disp_sum_f( &
        obs, nobs, src, slip, nsrc, nu, out, stream) &
        bind(c, name="cuokada_cuda_rect_disp_sum_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_rect_disp_sum_f

    integer(c_int) function cuokada_cuda_rect_disp_pairs_f( &
        obs, src, slip, n, nu, out, stream) &
        bind(c, name="cuokada_cuda_rect_disp_pairs_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: n
      real(c_float), value :: nu
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_rect_disp_pairs_f

    integer(c_int) function cuokada_cuda_rect_strain_matrix_f( &
        obs, nobs, src, nsrc, nu, out, stream) &
        bind(c, name="cuokada_cuda_rect_strain_matrix_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_rect_strain_matrix_f

    integer(c_int) function cuokada_cuda_rect_strain_sum_f( &
        obs, nobs, src, slip, nsrc, nu, out, stream) &
        bind(c, name="cuokada_cuda_rect_strain_sum_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_rect_strain_sum_f

    integer(c_int) function cuokada_cuda_rect_strain_pairs_f( &
        obs, src, slip, n, nu, out, stream) &
        bind(c, name="cuokada_cuda_rect_strain_pairs_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: n
      real(c_float), value :: nu
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_rect_strain_pairs_f

    integer(c_int) function cuokada_cuda_tri_disp_matrix_f( &
        obs, nobs, src, nsrc, nu, halfspace, out, stream) &
        bind(c, name="cuokada_cuda_tri_disp_matrix_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      integer(c_int), value :: halfspace
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_tri_disp_matrix_f

    integer(c_int) function cuokada_cuda_tri_disp_sum_f( &
        obs, nobs, src, slip, nsrc, nu, halfspace, out, stream) &
        bind(c, name="cuokada_cuda_tri_disp_sum_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      integer(c_int), value :: halfspace
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_tri_disp_sum_f

    integer(c_int) function cuokada_cuda_tri_disp_pairs_f( &
        obs, src, slip, n, nu, halfspace, out, stream) &
        bind(c, name="cuokada_cuda_tri_disp_pairs_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: n
      real(c_float), value :: nu
      integer(c_int), value :: halfspace
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_tri_disp_pairs_f

    integer(c_int) function cuokada_cuda_tri_strain_matrix_f( &
        obs, nobs, src, nsrc, nu, halfspace, out, stream) &
        bind(c, name="cuokada_cuda_tri_strain_matrix_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      integer(c_int), value :: halfspace
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_tri_strain_matrix_f

    integer(c_int) function cuokada_cuda_tri_strain_sum_f( &
        obs, nobs, src, slip, nsrc, nu, halfspace, out, stream) &
        bind(c, name="cuokada_cuda_tri_strain_sum_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      integer(c_int64_t), value :: nobs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: nsrc
      real(c_float), value :: nu
      integer(c_int), value :: halfspace
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_tri_strain_sum_f

    integer(c_int) function cuokada_cuda_tri_strain_pairs_f( &
        obs, src, slip, n, nu, halfspace, out, stream) &
        bind(c, name="cuokada_cuda_tri_strain_pairs_f")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: obs
      type(c_ptr), value :: src
      type(c_ptr), value :: slip
      integer(c_int64_t), value :: n
      real(c_float), value :: nu
      integer(c_int), value :: halfspace
      type(c_ptr), value :: out
      type(c_ptr), value :: stream
    end function cuokada_cuda_tri_strain_pairs_f

    integer(c_int) function cuokada_cuda_device_count(count) bind(c, name="cuokada_cuda_device_count")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      integer(c_int), intent(out) :: count
    end function cuokada_cuda_device_count

    integer(c_int) function cuokada_cuda_set_device(device) bind(c, name="cuokada_cuda_set_device")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      integer(c_int), value :: device
    end function cuokada_cuda_set_device

    integer(c_int) function cuokada_cuda_get_device(device) bind(c, name="cuokada_cuda_get_device")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      integer(c_int), intent(out) :: device
    end function cuokada_cuda_get_device

    integer(c_int) function cuokada_cuda_malloc(ptr, bytes) bind(c, name="cuokada_cuda_malloc")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), intent(out) :: ptr
      integer(c_int64_t), value :: bytes
    end function cuokada_cuda_malloc

    integer(c_int) function cuokada_cuda_free(ptr) bind(c, name="cuokada_cuda_free")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: ptr
    end function cuokada_cuda_free

    integer(c_int) function cuokada_cuda_malloc_host(ptr, bytes) bind(c, name="cuokada_cuda_malloc_host")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), intent(out) :: ptr
      integer(c_int64_t), value :: bytes
    end function cuokada_cuda_malloc_host

    integer(c_int) function cuokada_cuda_free_host(ptr) bind(c, name="cuokada_cuda_free_host")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: ptr
    end function cuokada_cuda_free_host

    integer(c_int) function cuokada_cuda_memcpy(dst, src, bytes, stream) bind(c, name="cuokada_cuda_memcpy")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: dst
      type(c_ptr), value :: src
      integer(c_int64_t), value :: bytes
      type(c_ptr), value :: stream
    end function cuokada_cuda_memcpy

    integer(c_int) function cuokada_cuda_synchronize(stream) bind(c, name="cuokada_cuda_synchronize")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: stream
    end function cuokada_cuda_synchronize

    integer(c_int) function cuokada_cuda_stream_wait(stream, other) bind(c, name="cuokada_cuda_stream_wait")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: stream
      type(c_ptr), value :: other
    end function cuokada_cuda_stream_wait

    integer(c_int) function cuokada_cuda_memory_kind(ptr, kind) bind(c, name="cuokada_cuda_memory_kind")
      import :: c_int, c_int64_t, c_double, c_float, c_ptr
      type(c_ptr), value :: ptr
      integer(c_int), intent(out) :: kind
    end function cuokada_cuda_memory_kind

    integer(c_int) function cuokada_has_cuda() bind(c, name="cuokada_has_cuda")
      import :: c_int
    end function cuokada_has_cuda

  end interface

end module cuokada

! end of file
