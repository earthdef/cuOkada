# -*- python -*-
#
# gpu arrays of other libraries in and out: cupy and jax, through __cuda_array_interface__ and
# dlpack; each test is skipped without its library
#
# Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

import os

import numpy

from helpers import cuokada, patches, triangles, stations, relerr, run

# keep jax from taking most of the gpu
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

rng = numpy.random.default_rng(5)
P = patches(10, rng)
T = triangles(P)
OBS = stations(rng, 100, depth=5.0)
SLIP = rng.normal(size=(len(T), 3))


def _gpu():
    return cuokada.has_cuda() and cuokada.device_count() > 0


def _cupy():
    try:
        import cupy
    except ImportError:
        return None
    return cupy if _gpu() else None


def _jax():
    try:
        import jax
    except ImportError:
        return None
    if not _gpu() or jax.default_backend() != "gpu":
        return None
    jax.config.update("jax_enable_x64", True)
    return jax


class DLPackOnly:
    # an array that offers only dlpack, as some libraries do
    def __init__(self, a):
        self.a = a

    def __dlpack__(self, **kw):
        return self.a.__dlpack__(**kw)

    def __dlpack_device__(self):
        return self.a.__dlpack_device__()


def test_cupy():
    cupy = _cupy()
    if cupy is None:
        print("  no cupy, skipped")
        return
    cpu = cuokada.tri_disp_sum(OBS, T, SLIP)
    # through __cuda_array_interface__: the result comes back as cupy
    out = cuokada.tri_disp_sum(cupy.asarray(OBS), cupy.asarray(T), cupy.asarray(SLIP))
    assert isinstance(out, cupy.ndarray) and relerr(cupy.asnumpy(out), cpu) < 1e-10
    # through dlpack only
    out = cuokada.tri_disp_sum(*(DLPackOnly(cupy.asarray(a)) for a in (OBS, T, SLIP)))
    assert isinstance(out, cuokada.DeviceArray) and relerr(out.copy_to_host(), cpu) < 1e-10
    # a DeviceArray out to cupy, through both
    assert relerr(cupy.asnumpy(cupy.asarray(out)), cpu) < 1e-10
    assert relerr(cupy.asnumpy(cupy.from_dlpack(out)), cpu) < 1e-10


def test_cupy_stream():
    # inputs written on another stream, without synchronizing: the launch waits for the stream
    # that cupy reports, its current one
    cupy = _cupy()
    if cupy is None:
        print("  no cupy, skipped")
        return
    cpu = cuokada.rect_disp_matrix(OBS, P)
    p = cupy.asarray(P)
    cupy.cuda.Device().synchronize()
    for _ in range(5):
        stream = cupy.cuda.Stream(non_blocking=True)
        with stream:
            # nan stations, then a long product on the stream, and only then the real ones
            obs = cupy.full((len(OBS), 3), cupy.nan)
            stream.synchronize()
            big = cupy.ones((1500, 1500))
            for _ in range(4):
                big = big @ big / 1500
            obs[...] = cupy.asarray(OBS) + 0 * big[0, 0]
            assert obs.__cuda_array_interface__.get("stream") == stream.ptr
            out = cuokada.rect_disp_matrix(obs, p, out=cuokada.device_array((len(OBS), 3, len(P), 3)))
        assert relerr(out.copy_to_host(), cpu) < 1e-10


def test_jax():
    jax = _jax()
    if jax is None:
        print("  no jax on the gpu, skipped")
        return
    import jax.numpy as jnp

    cpu = cuokada.rect_strain_matrix(OBS, P)
    for wrap in (lambda a: a, DLPackOnly):
        out = cuokada.rect_strain_matrix(wrap(jnp.asarray(OBS)), wrap(jnp.asarray(P)))
        assert isinstance(out, cuokada.DeviceArray)
        # a DeviceArray out to jax, through dlpack
        res = jnp.from_dlpack(out)
        assert res.dtype == jnp.float64 and relerr(numpy.asarray(res), cpu) < 1e-10


if __name__ == "__main__":
    run(dict(globals()))

# end of file
