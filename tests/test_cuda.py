# -*- python -*-
#
# the gpu against the cpu, for every shape, field and mode; cuokada's gpu arrays and host copies
#
# Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

import numpy

from helpers import cuokada, patches, triangles, stations, relerr, run

rng = numpy.random.default_rng(3)


def _gpu():
    return cuokada.has_cuda() and cuokada.device_count() > 0


def _numba():
    try:
        from numba import cuda
    except ImportError:
        return None
    return cuda if cuda.is_available() else None


def test_cpu_gpu():
    if not _gpu():
        print("  no cuda, skipped")
        return
    P = patches(20, rng)
    sources = {"rect": P, "tri": triangles(P)}
    obs = numpy.concatenate([stations(rng, 200), stations(rng, 100, depth=8.0)])
    # the gpu fuses multiply-adds, which okada's cancellations amplify
    for dtype, tol in ((numpy.float64, 1e-10), (numpy.float32, 1e-3)):
        o = obs.astype(dtype)
        d_obs = cuokada.to_device(o)
        for shape, src in sources.items():
            src = src.astype(dtype)
            slip = rng.normal(size=(len(src), 3)).astype(dtype)
            idx = rng.integers(0, len(src), len(o))
            d_src, d_slip = cuokada.to_device(src), cuokada.to_device(slip)
            d_psrc, d_pslip = cuokada.to_device(src[idx]), cuokada.to_device(slip[idx])
            for field in ("disp", "strain"):
                f = lambda mode: getattr(cuokada, f"{shape}_{field}_{mode}")  # noqa: E731
                cases = (
                    (f("matrix")(o, src), f("matrix")(d_obs, d_src)),
                    (f("sum")(o, src, slip), f("sum")(d_obs, d_src, d_slip)),
                    (f("pairs")(o, src[idx], slip[idx]), f("pairs")(d_obs, d_psrc, d_pslip)),
                )
                for mode, (cpu, gpu) in zip(("matrix", "sum", "pairs"), cases):
                    assert isinstance(gpu, cuokada.DeviceArray)
                    gpu = gpu.copy_to_host()
                    assert gpu.dtype == dtype
                    assert relerr(gpu, cpu) < tol, (shape, field, mode, dtype, relerr(gpu, cpu))


def test_host_copies():
    # numpy arrays with device="cuda" are copied in, and the result comes back
    if not _gpu():
        print("  no cuda, skipped")
        return
    P = patches(10, rng)
    obs = stations(rng, 100, depth=5.0)
    slip = rng.normal(size=(len(P), 3))
    cpu = cuokada.rect_strain_matrix(obs, P)
    gpu = cuokada.rect_strain_matrix(obs, P, device="cuda")
    assert isinstance(gpu, numpy.ndarray) and relerr(gpu, cpu) < 1e-10
    # into a given host array, from page-locked memory
    o, p = cuokada.pinned_array(obs.shape), cuokada.pinned_array(P.shape)
    o[:], p[:] = obs, P
    assert cuokada.memory_kind(o) == "pinned"
    out = cuokada.pinned_array((len(obs), 3))
    res = cuokada.rect_disp_sum(o, p, slip, out=out, device="cuda")
    assert res is out and relerr(out, cuokada.rect_disp_sum(obs, P, slip)) < 1e-10


def test_device_array():
    if not _gpu():
        print("  no cuda, skipped")
        return
    a = rng.normal(size=(7, 3)).astype(numpy.float32)
    d = cuokada.to_device(a)
    assert d.shape == (7, 3) and d.dtype == numpy.float32 and d.nbytes == a.nbytes
    assert cuokada.memory_kind(d) == "device"
    assert cuokada.memory_kind(a) == "host"
    assert (d.copy_to_host() == a).all()
    e = cuokada.device_array((7, 3), numpy.float32).copy_from_host(2 * a)
    assert (e.copy_to_host() == 2 * a).all()
    assert cuokada.device_array(0).copy_to_host().size == 0


def test_numba_arrays():
    # the arrays of another library, in and out
    cuda = _numba()
    if not _gpu() or cuda is None:
        print("  no numba, skipped")
        return
    P = patches(10, rng)
    obs = stations(rng, 100, depth=5.0)
    out = cuokada.tri_disp_matrix(cuda.to_device(obs), cuda.to_device(triangles(P)))
    assert type(out).__module__.split(".")[0] == "numba"
    assert relerr(out.copy_to_host(), cuokada.tri_disp_matrix(obs, triangles(P))) < 1e-10


def test_managed():
    # numpy arrays in managed memory run on the gpu in place
    cuda = _numba()
    if not _gpu() or cuda is None:
        print("  no numba, skipped")
        return
    P = patches(10, rng)
    obs = stations(rng, 100, depth=5.0)
    m_obs = cuda.managed_array(obs.shape, dtype=numpy.float64)
    m_src = cuda.managed_array(P.shape, dtype=numpy.float64)
    m_out = cuda.managed_array((len(obs), 3, len(P), 3), dtype=numpy.float64)
    m_obs[:] = obs
    m_src[:] = P
    assert cuokada.memory_kind(numpy.asarray(m_obs)) == "managed"
    out = numpy.asarray(m_out)
    res = cuokada.rect_disp_matrix(numpy.asarray(m_obs), numpy.asarray(m_src), out=out, device="cuda")
    cuokada.synchronize()
    assert res is out and relerr(out, cuokada.rect_disp_matrix(obs, P)) < 1e-12


if __name__ == "__main__":
    run(dict(globals()))

# end of file
