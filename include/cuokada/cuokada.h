/* -*- c -*-
 *
 * cuokada: elastic dislocations in a half space, on the cpu and the gpu; the c interface
 *
 * Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE
 *
 * Coordinates are x east, y north, z up, with stations at z <= 0; lengths in any one unit (the
 * singularity tolerances are 1e-6 of it, as in Okada's dc3d), displacements in the unit of slip.
 * All arrays are contiguous, row major.
 *
 * Sources:
 *   rect: [x, y, depth, length, width, strike, dip] per patch: the center of the patch, the
 *         depth of the center (> 0), the strike clockwise from north and the dip to its right,
 *         in degrees; slip [strike, dip, tensile]: left lateral, reverse and opening positive,
 *         as in Okada (1992)
 *   tri:  [x0, y0, z0, x1, y1, z1, x2, y2, z2] per triangle; slip [strike, dip, tensile] in the
 *         triangle frame of Nikkhoo and Walter (2015); halfspace = 0 for a full space
 *
 * Fields: disp [ux, uy, uz], strain [exx, eyy, ezz, exy, exz, eyz]; F = 3 or 6 components
 *
 * Modes:
 *   matrix: out[nobs, F, nsrc, 3], the response of each station to unit slips of each source
 *   sum:    out[nobs, F], the response of each station to all sources with their slip[nsrc, 3]
 *   pairs:  out[n, F], the response of station i to source i with slip i
 *
 * Every function returns 0 on success; the cuda ones return the cudaError_t of the launch.
 * Stations on a rectangle edge get zeros (as dc3d), above the surface nan.
 */

#ifndef CUOKADA_H
#define CUOKADA_H

#include <stdint.h>

#if defined(_WIN32)
#define CUOKADA_API __declspec(dllexport)
#else
#define CUOKADA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// the cpu backend
CUOKADA_API int cuokada_rect_disp_matrix_d(const double *obs, int64_t nobs, const double *src, int64_t nsrc, double nu, double *out);
CUOKADA_API int cuokada_rect_disp_sum_d(const double *obs, int64_t nobs, const double *src, const double *slip, int64_t nsrc, double nu, double *out);
CUOKADA_API int cuokada_rect_disp_pairs_d(const double *obs, const double *src, const double *slip, int64_t n, double nu, double *out);
CUOKADA_API int cuokada_rect_strain_matrix_d(const double *obs, int64_t nobs, const double *src, int64_t nsrc, double nu, double *out);
CUOKADA_API int cuokada_rect_strain_sum_d(const double *obs, int64_t nobs, const double *src, const double *slip, int64_t nsrc, double nu, double *out);
CUOKADA_API int cuokada_rect_strain_pairs_d(const double *obs, const double *src, const double *slip, int64_t n, double nu, double *out);
CUOKADA_API int cuokada_tri_disp_matrix_d(const double *obs, int64_t nobs, const double *src, int64_t nsrc, double nu, int halfspace, double *out);
CUOKADA_API int cuokada_tri_disp_sum_d(const double *obs, int64_t nobs, const double *src, const double *slip, int64_t nsrc, double nu, int halfspace, double *out);
CUOKADA_API int cuokada_tri_disp_pairs_d(const double *obs, const double *src, const double *slip, int64_t n, double nu, int halfspace, double *out);
CUOKADA_API int cuokada_tri_strain_matrix_d(const double *obs, int64_t nobs, const double *src, int64_t nsrc, double nu, int halfspace, double *out);
CUOKADA_API int cuokada_tri_strain_sum_d(const double *obs, int64_t nobs, const double *src, const double *slip, int64_t nsrc, double nu, int halfspace, double *out);
CUOKADA_API int cuokada_tri_strain_pairs_d(const double *obs, const double *src, const double *slip, int64_t n, double nu, int halfspace, double *out);

CUOKADA_API int cuokada_rect_disp_matrix_f(const float *obs, int64_t nobs, const float *src, int64_t nsrc, float nu, float *out);
CUOKADA_API int cuokada_rect_disp_sum_f(const float *obs, int64_t nobs, const float *src, const float *slip, int64_t nsrc, float nu, float *out);
CUOKADA_API int cuokada_rect_disp_pairs_f(const float *obs, const float *src, const float *slip, int64_t n, float nu, float *out);
CUOKADA_API int cuokada_rect_strain_matrix_f(const float *obs, int64_t nobs, const float *src, int64_t nsrc, float nu, float *out);
CUOKADA_API int cuokada_rect_strain_sum_f(const float *obs, int64_t nobs, const float *src, const float *slip, int64_t nsrc, float nu, float *out);
CUOKADA_API int cuokada_rect_strain_pairs_f(const float *obs, const float *src, const float *slip, int64_t n, float nu, float *out);
CUOKADA_API int cuokada_tri_disp_matrix_f(const float *obs, int64_t nobs, const float *src, int64_t nsrc, float nu, int halfspace, float *out);
CUOKADA_API int cuokada_tri_disp_sum_f(const float *obs, int64_t nobs, const float *src, const float *slip, int64_t nsrc, float nu, int halfspace, float *out);
CUOKADA_API int cuokada_tri_disp_pairs_f(const float *obs, const float *src, const float *slip, int64_t n, float nu, int halfspace, float *out);
CUOKADA_API int cuokada_tri_strain_matrix_f(const float *obs, int64_t nobs, const float *src, int64_t nsrc, float nu, int halfspace, float *out);
CUOKADA_API int cuokada_tri_strain_sum_f(const float *obs, int64_t nobs, const float *src, const float *slip, int64_t nsrc, float nu, int halfspace, float *out);
CUOKADA_API int cuokada_tri_strain_pairs_f(const float *obs, const float *src, const float *slip, int64_t n, float nu, int halfspace, float *out);

// the cuda backend: device (or managed) pointers, asynchronous on {stream}, 0 for the default stream
CUOKADA_API int cuokada_cuda_rect_disp_matrix_d(const double *obs, int64_t nobs, const double *src, int64_t nsrc, double nu, double *out, void *stream);
CUOKADA_API int cuokada_cuda_rect_disp_sum_d(const double *obs, int64_t nobs, const double *src, const double *slip, int64_t nsrc, double nu, double *out, void *stream);
CUOKADA_API int cuokada_cuda_rect_disp_pairs_d(const double *obs, const double *src, const double *slip, int64_t n, double nu, double *out, void *stream);
CUOKADA_API int cuokada_cuda_rect_strain_matrix_d(const double *obs, int64_t nobs, const double *src, int64_t nsrc, double nu, double *out, void *stream);
CUOKADA_API int cuokada_cuda_rect_strain_sum_d(const double *obs, int64_t nobs, const double *src, const double *slip, int64_t nsrc, double nu, double *out, void *stream);
CUOKADA_API int cuokada_cuda_rect_strain_pairs_d(const double *obs, const double *src, const double *slip, int64_t n, double nu, double *out, void *stream);
CUOKADA_API int cuokada_cuda_tri_disp_matrix_d(const double *obs, int64_t nobs, const double *src, int64_t nsrc, double nu, int halfspace, double *out, void *stream);
CUOKADA_API int cuokada_cuda_tri_disp_sum_d(const double *obs, int64_t nobs, const double *src, const double *slip, int64_t nsrc, double nu, int halfspace, double *out, void *stream);
CUOKADA_API int cuokada_cuda_tri_disp_pairs_d(const double *obs, const double *src, const double *slip, int64_t n, double nu, int halfspace, double *out, void *stream);
CUOKADA_API int cuokada_cuda_tri_strain_matrix_d(const double *obs, int64_t nobs, const double *src, int64_t nsrc, double nu, int halfspace, double *out, void *stream);
CUOKADA_API int cuokada_cuda_tri_strain_sum_d(const double *obs, int64_t nobs, const double *src, const double *slip, int64_t nsrc, double nu, int halfspace, double *out, void *stream);
CUOKADA_API int cuokada_cuda_tri_strain_pairs_d(const double *obs, const double *src, const double *slip, int64_t n, double nu, int halfspace, double *out, void *stream);

CUOKADA_API int cuokada_cuda_rect_disp_matrix_f(const float *obs, int64_t nobs, const float *src, int64_t nsrc, float nu, float *out, void *stream);
CUOKADA_API int cuokada_cuda_rect_disp_sum_f(const float *obs, int64_t nobs, const float *src, const float *slip, int64_t nsrc, float nu, float *out, void *stream);
CUOKADA_API int cuokada_cuda_rect_disp_pairs_f(const float *obs, const float *src, const float *slip, int64_t n, float nu, float *out, void *stream);
CUOKADA_API int cuokada_cuda_rect_strain_matrix_f(const float *obs, int64_t nobs, const float *src, int64_t nsrc, float nu, float *out, void *stream);
CUOKADA_API int cuokada_cuda_rect_strain_sum_f(const float *obs, int64_t nobs, const float *src, const float *slip, int64_t nsrc, float nu, float *out, void *stream);
CUOKADA_API int cuokada_cuda_rect_strain_pairs_f(const float *obs, const float *src, const float *slip, int64_t n, float nu, float *out, void *stream);
CUOKADA_API int cuokada_cuda_tri_disp_matrix_f(const float *obs, int64_t nobs, const float *src, int64_t nsrc, float nu, int halfspace, float *out, void *stream);
CUOKADA_API int cuokada_cuda_tri_disp_sum_f(const float *obs, int64_t nobs, const float *src, const float *slip, int64_t nsrc, float nu, int halfspace, float *out, void *stream);
CUOKADA_API int cuokada_cuda_tri_disp_pairs_f(const float *obs, const float *src, const float *slip, int64_t n, float nu, int halfspace, float *out, void *stream);
CUOKADA_API int cuokada_cuda_tri_strain_matrix_f(const float *obs, int64_t nobs, const float *src, int64_t nsrc, float nu, int halfspace, float *out, void *stream);
CUOKADA_API int cuokada_cuda_tri_strain_sum_f(const float *obs, int64_t nobs, const float *src, const float *slip, int64_t nsrc, float nu, int halfspace, float *out, void *stream);
CUOKADA_API int cuokada_cuda_tri_strain_pairs_f(const float *obs, const float *src, const float *slip, int64_t n, float nu, int halfspace, float *out, void *stream);

/* whether the library has the cuda backend */
CUOKADA_API int cuokada_has_cuda(void);

/* gpu memory, for codes without cuda arrays of their own; each returns its cudaError_t */
CUOKADA_API int cuokada_cuda_device_count(int *count);
CUOKADA_API int cuokada_cuda_set_device(int device);
CUOKADA_API int cuokada_cuda_get_device(int *device);
CUOKADA_API int cuokada_cuda_malloc(void **ptr, int64_t bytes);
CUOKADA_API int cuokada_cuda_free(void *ptr);
/* page-locked host memory, for fast copies */
CUOKADA_API int cuokada_cuda_malloc_host(void **ptr, int64_t bytes);
CUOKADA_API int cuokada_cuda_free_host(void *ptr);
/* a copy in any direction, asynchronous on {stream} */
CUOKADA_API int cuokada_cuda_memcpy(void *dst, const void *src, int64_t bytes, void *stream);
CUOKADA_API int cuokada_cuda_synchronize(void *stream);
/* make {stream} wait for the work queued so far on {other}, without blocking the host */
CUOKADA_API int cuokada_cuda_stream_wait(void *stream, void *other);
/* where {ptr} lives: 0 host, 1 page-locked host, 2 device, 3 managed */
CUOKADA_API int cuokada_cuda_memory_kind(const void *ptr, int *kind);

#ifdef __cplusplus
}
#endif

#endif

/* end of file */
