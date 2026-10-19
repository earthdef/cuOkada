! -*- f90 -*-
!
! the fortran interface: uplift on a grid from two patches, the sum against the matrix
!
! Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

program test_fortran
  use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int64_t
  use cuokada
  implicit none

  integer, parameter :: nx = 21, ny = 11, nobs = nx * ny, nsrc = 2
  real(c_double) :: obs(3, nobs), src(7, nsrc), slip(3, nsrc)
  real(c_double) :: u(3, nobs), g(3, nsrc, 3, nobs), uz
  integer(c_int) :: status
  integer :: i, j, k, failures

  ! a grid at the surface, and two thrusts along a trench
  do j = 1, ny
    do i = 1, nx
      k = (j - 1) * nx + i
      obs(:, k) = [-50.0d0 + 5.0d0 * (i - 1), -25.0d0 + 5.0d0 * (j - 1), 0.0d0]
    end do
  end do
  src(:, 1) = [-10.0d0, 0.0d0, 10.0d0, 20.0d0, 15.0d0, 0.0d0, 15.0d0]
  src(:, 2) = [10.0d0, 0.0d0, 12.0d0, 20.0d0, 15.0d0, 10.0d0, 20.0d0]
  slip(:, 1) = [0.0d0, 2.0d0, 0.0d0]
  slip(:, 2) = [0.5d0, 3.0d0, 0.0d0]

  status = cuokada_rect_disp_sum_d(obs, int(nobs, c_int64_t), src, slip, int(nsrc, c_int64_t), 0.25d0, u)
  status = status + cuokada_rect_disp_matrix_d(obs, int(nobs, c_int64_t), src, int(nsrc, c_int64_t), 0.25d0, g)

  failures = 0
  do k = 1, nobs
    do i = 1, 3
      if (abs(u(i, k) - sum(g(:, :, i, k) * slip)) > 1d-12 * (1 + abs(u(i, k)))) failures = failures + 1
    end do
  end do
  uz = maxval(u(3, :))
  if (status /= 0 .or. failures /= 0 .or. .not. (uz > 0)) then
    print '(a, i0, a, i0)', 'FAIL: status ', status, ', failures ', failures
    stop 1
  end if
  print '(a, f8.4, a, i0)', 'PASS: max uplift ', uz, ', cuda ', cuokada_has_cuda()
end program test_fortran

! end of file
