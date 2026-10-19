// -*- c++ -*-
//
// the cuda backend of the c interface
//
// Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

#include <cuda_runtime.h>
#include <cuokada/cuokada.h>
#include <cuokada/kernels.h>

namespace {

using namespace cuokada;

constexpr int BLOCK = 128;

inline unsigned int blocks(int64_t n) {
    return static_cast<unsigned int>((n + BLOCK - 1) / BLOCK);
}

template <class E, typename T>
__global__ void matrix_kernel(const T * obs, int64_t nobs, const T * src, int64_t nsrc, T nu, T * out) {
    const int64_t k = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= nobs * nsrc) return;
    // a warp along the sources writes contiguously; along the stations it shares one source,
    // so that only the warps on near vertical patches take the double path
    if constexpr (E::MIXED) {
        matrix_entry<E>(k % nobs, k / nobs, obs, src, nsrc, nu, out);
    } else {
        matrix_entry<E>(k / nsrc, k % nsrc, obs, src, nsrc, nu, out);
    }
}

template <class E, typename T>
__global__ void sum_kernel(const T * obs, int64_t nobs, const T * src, const T * slip, int64_t nsrc, T nu, T * out) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nobs) return;
    sum_entry<E>(i, obs, src, slip, nsrc, nu, out);
}

template <class E, typename T>
__global__ void pairs_kernel(const T * obs, const T * src, const T * slip, int64_t n, T nu, T * out) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    pair_entry<E>(i, obs, src, slip, nu, out);
}

template <class E, typename T>
int matrix(const T * obs, int64_t nobs, const T * src, int64_t nsrc, T nu, T * out, void * stream) {
    if (nobs * nsrc == 0) return 0;
    matrix_kernel<E><<<blocks(nobs * nsrc), BLOCK, 0, cudaStream_t(stream)>>>(obs, nobs, src, nsrc, nu, out);
    return cudaGetLastError();
}

template <class E, typename T>
int sum(const T * obs, int64_t nobs, const T * src, const T * slip, int64_t nsrc, T nu, T * out, void * stream) {
    if (nobs == 0) return 0;
    sum_kernel<E><<<blocks(nobs), BLOCK, 0, cudaStream_t(stream)>>>(obs, nobs, src, slip, nsrc, nu, out);
    return cudaGetLastError();
}

template <class E, typename T>
int pairs(const T * obs, const T * src, const T * slip, int64_t n, T nu, T * out, void * stream) {
    if (n == 0) return 0;
    pairs_kernel<E><<<blocks(n), BLOCK, 0, cudaStream_t(stream)>>>(obs, src, slip, n, nu, out);
    return cudaGetLastError();
}

} // namespace

#define CUOKADA_RECT(field, F, p, T)                                                                         \
    int cuokada_cuda_rect_##field##_matrix_##p(const T * obs, int64_t nobs, const T * src, int64_t nsrc,    \
                                               T nu, T * out, void * stream) {                              \
        return matrix<Rect<T, F>>(obs, nobs, src, nsrc, nu, out, stream);                                   \
    }                                                                                                       \
    int cuokada_cuda_rect_##field##_sum_##p(const T * obs, int64_t nobs, const T * src, const T * slip,    \
                                            int64_t nsrc, T nu, T * out, void * stream) {                   \
        return sum<Rect<T, F>>(obs, nobs, src, slip, nsrc, nu, out, stream);                                \
    }                                                                                                       \
    int cuokada_cuda_rect_##field##_pairs_##p(const T * obs, const T * src, const T * slip, int64_t n,     \
                                              T nu, T * out, void * stream) {                               \
        return pairs<Rect<T, F>>(obs, src, slip, n, nu, out, stream);                                       \
    }

#define CUOKADA_TRI(field, F, p, T)                                                                          \
    int cuokada_cuda_tri_##field##_matrix_##p(const T * obs, int64_t nobs, const T * src, int64_t nsrc,     \
                                              T nu, int halfspace, T * out, void * stream) {                \
        return halfspace ? matrix<Tri<T, F, true>>(obs, nobs, src, nsrc, nu, out, stream)                   \
                         : matrix<Tri<T, F, false>>(obs, nobs, src, nsrc, nu, out, stream);                 \
    }                                                                                                       \
    int cuokada_cuda_tri_##field##_sum_##p(const T * obs, int64_t nobs, const T * src, const T * slip,     \
                                           int64_t nsrc, T nu, int halfspace, T * out, void * stream) {     \
        return halfspace ? sum<Tri<T, F, true>>(obs, nobs, src, slip, nsrc, nu, out, stream)                \
                         : sum<Tri<T, F, false>>(obs, nobs, src, slip, nsrc, nu, out, stream);              \
    }                                                                                                       \
    int cuokada_cuda_tri_##field##_pairs_##p(const T * obs, const T * src, const T * slip, int64_t n,      \
                                             T nu, int halfspace, T * out, void * stream) {                 \
        return halfspace ? pairs<Tri<T, F, true>>(obs, src, slip, n, nu, out, stream)                       \
                         : pairs<Tri<T, F, false>>(obs, src, slip, n, nu, out, stream);                     \
    }

extern "C" {

int cuokada_has_cuda(void) { return 1; }

int cuokada_cuda_device_count(int * count) { return cudaGetDeviceCount(count); }

int cuokada_cuda_set_device(int device) { return cudaSetDevice(device); }

int cuokada_cuda_get_device(int * device) { return cudaGetDevice(device); }

int cuokada_cuda_malloc(void ** ptr, int64_t bytes) { return cudaMalloc(ptr, size_t(bytes)); }

int cuokada_cuda_free(void * ptr) { return cudaFree(ptr); }

int cuokada_cuda_malloc_host(void ** ptr, int64_t bytes) { return cudaMallocHost(ptr, size_t(bytes)); }

int cuokada_cuda_free_host(void * ptr) { return cudaFreeHost(ptr); }

int cuokada_cuda_memcpy(void * dst, const void * src, int64_t bytes, void * stream) {
    return cudaMemcpyAsync(dst, src, size_t(bytes), cudaMemcpyDefault, cudaStream_t(stream));
}

int cuokada_cuda_synchronize(void * stream) {
    return stream ? cudaStreamSynchronize(cudaStream_t(stream)) : cudaDeviceSynchronize();
}

int cuokada_cuda_stream_wait(void * stream, void * other) {
    cudaEvent_t event;
    cudaError_t status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (status != cudaSuccess) return status;
    status = cudaEventRecord(event, cudaStream_t(other));
    if (status == cudaSuccess) status = cudaStreamWaitEvent(cudaStream_t(stream), event, 0);
    cudaEventDestroy(event);
    return status;
}

int cuokada_cuda_memory_kind(const void * ptr, int * kind) {
    cudaPointerAttributes a;
    const cudaError_t status = cudaPointerGetAttributes(&a, ptr);
    if (status != cudaSuccess) {
        cudaGetLastError();
        *kind = 0;
        return status;
    }
    switch (a.type) {
    case cudaMemoryTypeHost: *kind = 1; break;
    case cudaMemoryTypeDevice: *kind = 2; break;
    case cudaMemoryTypeManaged: *kind = 3; break;
    default: *kind = 0;
    }
    return 0;
}

CUOKADA_RECT(disp, DISP, d, double)
CUOKADA_RECT(strain, STRAIN, d, double)
CUOKADA_RECT(disp, DISP, f, float)
CUOKADA_RECT(strain, STRAIN, f, float)
CUOKADA_TRI(disp, DISP, d, double)
CUOKADA_TRI(strain, STRAIN, d, double)
CUOKADA_TRI(disp, DISP, f, float)
CUOKADA_TRI(strain, STRAIN, f, float)

} // extern "C"

// end of file
