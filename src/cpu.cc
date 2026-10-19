// -*- c++ -*-
//
// the cpu backend of the c interface
//
// Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

#include <cuokada/cuokada.h>
#include <cuokada/kernels.h>

namespace {

using namespace cuokada;

template <class E, typename T>
int matrix(const T * obs, int64_t nobs, const T * src, int64_t nsrc, T nu, T * out) {
#pragma omp parallel for collapse(2) schedule(dynamic, 64)
    for (int64_t i = 0; i < nobs; ++i)
        for (int64_t j = 0; j < nsrc; ++j)
            matrix_entry<E>(i, j, obs, src, nsrc, nu, out);
    return 0;
}

template <class E, typename T>
int sum(const T * obs, int64_t nobs, const T * src, const T * slip, int64_t nsrc, T nu, T * out) {
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t i = 0; i < nobs; ++i)
        sum_entry<E>(i, obs, src, slip, nsrc, nu, out);
    return 0;
}

template <class E, typename T>
int pairs(const T * obs, const T * src, const T * slip, int64_t n, T nu, T * out) {
#pragma omp parallel for schedule(dynamic, 64)
    for (int64_t i = 0; i < n; ++i)
        pair_entry<E>(i, obs, src, slip, nu, out);
    return 0;
}

} // namespace

#define CUOKADA_RECT(field, F, p, T)                                                                         \
    int cuokada_rect_##field##_matrix_##p(const T * obs, int64_t nobs, const T * src, int64_t nsrc, T nu,   \
                                          T * out) {                                                        \
        return matrix<Rect<T, F>>(obs, nobs, src, nsrc, nu, out);                                           \
    }                                                                                                       \
    int cuokada_rect_##field##_sum_##p(const T * obs, int64_t nobs, const T * src, const T * slip,         \
                                       int64_t nsrc, T nu, T * out) {                                       \
        return sum<Rect<T, F>>(obs, nobs, src, slip, nsrc, nu, out);                                        \
    }                                                                                                       \
    int cuokada_rect_##field##_pairs_##p(const T * obs, const T * src, const T * slip, int64_t n, T nu,     \
                                         T * out) {                                                         \
        return pairs<Rect<T, F>>(obs, src, slip, n, nu, out);                                               \
    }

#define CUOKADA_TRI(field, F, p, T)                                                                          \
    int cuokada_tri_##field##_matrix_##p(const T * obs, int64_t nobs, const T * src, int64_t nsrc, T nu,    \
                                         int halfspace, T * out) {                                          \
        return halfspace ? matrix<Tri<T, F, true>>(obs, nobs, src, nsrc, nu, out)                           \
                         : matrix<Tri<T, F, false>>(obs, nobs, src, nsrc, nu, out);                         \
    }                                                                                                       \
    int cuokada_tri_##field##_sum_##p(const T * obs, int64_t nobs, const T * src, const T * slip,          \
                                      int64_t nsrc, T nu, int halfspace, T * out) {                         \
        return halfspace ? sum<Tri<T, F, true>>(obs, nobs, src, slip, nsrc, nu, out)                        \
                         : sum<Tri<T, F, false>>(obs, nobs, src, slip, nsrc, nu, out);                      \
    }                                                                                                       \
    int cuokada_tri_##field##_pairs_##p(const T * obs, const T * src, const T * slip, int64_t n, T nu,      \
                                        int halfspace, T * out) {                                           \
        return halfspace ? pairs<Tri<T, F, true>>(obs, src, slip, n, nu, out)                               \
                         : pairs<Tri<T, F, false>>(obs, src, slip, n, nu, out);                             \
    }

extern "C" {

CUOKADA_RECT(disp, DISP, d, double)
CUOKADA_RECT(strain, STRAIN, d, double)
CUOKADA_RECT(disp, DISP, f, float)
CUOKADA_RECT(strain, STRAIN, f, float)
CUOKADA_TRI(disp, DISP, d, double)
CUOKADA_TRI(strain, STRAIN, d, double)
CUOKADA_TRI(disp, DISP, f, float)
CUOKADA_TRI(strain, STRAIN, f, float)

#if !defined(CUOKADA_HAVE_CUDA)

int cuokada_has_cuda(void) { return 0; }

// without cuda, there are no devices and all memory is the host's
int cuokada_cuda_device_count(int * count) {
    *count = 0;
    return 0;
}
int cuokada_cuda_set_device(int) { return 100; }
int cuokada_cuda_get_device(int *) { return 100; }
int cuokada_cuda_malloc(void **, int64_t) { return 100; }
int cuokada_cuda_free(void *) { return 100; }
int cuokada_cuda_malloc_host(void **, int64_t) { return 100; }
int cuokada_cuda_free_host(void *) { return 100; }
int cuokada_cuda_memcpy(void *, const void *, int64_t, void *) { return 100; }
int cuokada_cuda_synchronize(void *) { return 0; }
int cuokada_cuda_stream_wait(void *, void *) { return 0; }
int cuokada_cuda_memory_kind(const void *, int * kind) {
    *kind = 0;
    return 0;
}

// without cuda, every cuda entry point fails with cudaErrorNoDevice
#define CUOKADA_NOCUDA_SHAPE(shape, field, p, T, ...)                                                        \
    int cuokada_cuda_##shape##_##field##_matrix_##p(const T *, int64_t, const T *, int64_t, T, __VA_ARGS__  \
                                                    T *, void *) {                                          \
        return 100;                                                                                         \
    }                                                                                                       \
    int cuokada_cuda_##shape##_##field##_sum_##p(const T *, int64_t, const T *, const T *, int64_t, T,     \
                                                 __VA_ARGS__ T *, void *) {                                 \
        return 100;                                                                                         \
    }                                                                                                       \
    int cuokada_cuda_##shape##_##field##_pairs_##p(const T *, const T *, const T *, int64_t, T,            \
                                                   __VA_ARGS__ T *, void *) {                               \
        return 100;                                                                                         \
    }

CUOKADA_NOCUDA_SHAPE(rect, disp, d, double, )
CUOKADA_NOCUDA_SHAPE(rect, strain, d, double, )
CUOKADA_NOCUDA_SHAPE(rect, disp, f, float, )
CUOKADA_NOCUDA_SHAPE(rect, strain, f, float, )
CUOKADA_NOCUDA_SHAPE(tri, disp, d, double, int, )
CUOKADA_NOCUDA_SHAPE(tri, strain, d, double, int, )
CUOKADA_NOCUDA_SHAPE(tri, disp, f, float, int, )
CUOKADA_NOCUDA_SHAPE(tri, strain, f, float, int, )

#endif

} // extern "C"

// end of file
