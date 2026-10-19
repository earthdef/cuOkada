# -*- python -*-
#
# the triangle kernels on the gpu, matrix and sum, displacement and strain, inputs and output on
# the gpu; and the displacement matrix from and to host memory
#
# Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE
#
# usage: python tri_gpu.py

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from helpers import cuokada, patches, triangles, stations  # noqa: E402

REPEAT = 5


def timeit(launch):
    # the mean of a few runs, after one more
    launch()
    cuokada.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        launch()
    cuokada.synchronize()
    return (time.perf_counter() - t0) / REPEAT


def main():
    rng = np.random.default_rng(0)
    T = triangles(patches(400, rng))
    obs = stations(rng, 2500, extent=50.0)
    slip = rng.normal(size=(len(T), 3))
    # raise the gpu clocks from idle
    d_o, d_t = cuokada.to_device(obs), cuokada.to_device(T)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2:
        cuokada.tri_disp_matrix(d_o, d_t)
        cuokada.synchronize()
    rows = {}
    for dtype in (np.float64, np.float32):
        name = np.dtype(dtype).name
        o, t, s = (a.astype(dtype) for a in (obs, T, slip))
        d_obs, d_tris, d_slip = (cuokada.to_device(a) for a in (o, t, s))
        for field, nf in (("disp", 3), ("strain", 6)):
            out = cuokada.device_array((len(o), nf, len(t), 3), dtype)
            f = getattr(cuokada, f"tri_{field}_matrix")
            rows[f"matrix {field} {name}"] = timeit(lambda: f(d_obs, d_tris, out=out))
            outs = cuokada.device_array((len(o), nf), dtype)
            g = getattr(cuokada, f"tri_{field}_sum")
            rows[f"sum    {field} {name}"] = timeit(lambda: g(d_obs, d_tris, d_slip, out=outs))
        rows[f"matrix disp {name}, from and to the host"] = timeit(lambda: cuokada.tri_disp_matrix(o, t, device="cuda"))
    for k, v in rows.items():
        print(f"{k:40s} {v * 1e3:9.2f} ms")


if __name__ == "__main__":
    main()

# end of file
