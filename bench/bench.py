# -*- python -*-
#
# timing of the matrix of rectangles: cuokada on the cpu and the gpu, okada4py, cutde
#
# Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE
#
# usage: OMP_NUM_THREADS=8 python bench.py [patches] [stations] [--gpu: skip the cpu]

import os
import sys
import time

import numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from helpers import cuokada, patches, triangles, stations  # noqa: E402


def best(f, repeat=3):
    # the fastest of a few runs
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gpu_only = "--gpu" in sys.argv
    npatch = int(args[0]) if len(args) > 0 else 400
    nobs = int(args[1]) if len(args) > 1 else 2500
    rng = numpy.random.default_rng(0)
    P = patches(npatch, rng)
    T = triangles(P)
    obs = stations(rng, nobs, extent=50.0)
    rows = {}
    if not gpu_only:
        for dtype in (numpy.float64, numpy.float32):
            name = numpy.dtype(dtype).name
            o, p, t = obs.astype(dtype), P.astype(dtype), T.astype(dtype)
            rows[f"cuokada rect cpu {name}"] = best(lambda: cuokada.rect_disp_matrix(o, p))
            rows[f"cuokada tri  cpu {name}"] = best(lambda: cuokada.tri_disp_matrix(o, t))
        try:
            import cutde.halfspace as HS

            rows["cutde (2 tri) cpu float64"] = best(lambda: HS.disp_matrix(obs, T, 0.25))
        except ImportError:
            pass
        try:
            import okada4py as ok

            xyz = [numpy.ascontiguousarray(obs[:, i]) for i in range(3)]

            def okada():
                for j in range(npatch):
                    for s in range(3):
                        slip = [numpy.zeros(1) for _ in range(3)]
                        slip[s][0] = 1.0
                        ok.okada92(*xyz, *[P[j : j + 1, k].copy() for k in (0, 1, 2, 3, 4, 6, 5)], *slip, 30e9, 0.25)

            rows["okada4py cpu float64"] = best(okada, 1)
        except ImportError:
            pass
    if cuokada.has_cuda() and cuokada.device_count() > 0:
        # raise the gpu clocks from idle before timing
        d_obs, d_p = cuokada.to_device(obs), cuokada.to_device(P)
        out = cuokada.device_array((nobs, 3, npatch, 3))
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 2:
            cuokada.rect_disp_matrix(d_obs, d_p, out=out)
            cuokada.synchronize()

        def run(f, *args, **kw):
            f(*args, **kw)
            cuokada.synchronize()

        for dtype in (numpy.float64, numpy.float32):
            name = numpy.dtype(dtype).name
            for shape, src in (("rect", P), ("tri ", T)):
                f = getattr(cuokada, f"{shape.strip()}_disp_matrix")
                o, s = obs.astype(dtype), src.astype(dtype)
                # on the gpu, inputs and output there
                d_o, d_s = cuokada.to_device(o), cuokada.to_device(s)
                d_out = cuokada.device_array((nobs, 3, len(s), 3), dtype)
                rows[f"cuokada {shape} gpu {name}"] = best(lambda: run(f, d_o, d_s, out=d_out), 5)
                # from and to page-locked host memory
                p_o, p_s = cuokada.pinned_array(o.shape, dtype), cuokada.pinned_array(s.shape, dtype)
                p_o[:], p_s[:] = o, s
                p_out = cuokada.pinned_array((nobs, 3, len(s), 3), dtype)
                rows[f"cuokada {shape} gpu {name}, pinned copies"] = best(lambda: f(p_o, p_s, out=p_out, device="cuda"), 5)
                # from and to ordinary host memory
                rows[f"cuokada {shape} gpu {name}, pageable copies"] = best(lambda: f(o, s, device="cuda"), 5)
    n = npatch * nobs * 3
    threads = os.environ.get("OMP_NUM_THREADS", "all")
    print(f"displacement matrix: {npatch} rectangles x {nobs} stations x 3 slips, omp threads {threads}")
    for name, t in rows.items():
        print(f"  {name:44s} {t:9.4f} s  {n / t / 1e6:9.1f} M rectangle-slips/s")


if __name__ == "__main__":
    main()

# end of file
