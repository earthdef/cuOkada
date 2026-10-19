# -*- python -*-
#
# shared by the tests: the package from the build directory, sources and stations
#
# Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

import os
import sys

import numpy

# prefer the build directory, then an installed package
_build = os.environ.get("CUOKADA_BUILD", os.path.join(os.path.dirname(__file__), "..", "build"))
if os.path.isdir(os.path.join(_build, "cuokada")):
    sys.path.insert(0, os.path.abspath(_build))

import cuokada  # noqa: E402


def patches(n, rng, top=None, **fixed):
    # random rectangles [x, y, depth, length, width, strike, dip], with their top at {top} if given
    P = numpy.empty((n, 7))
    P[:, 0] = rng.uniform(-20, 20, n)
    P[:, 1] = rng.uniform(-20, 20, n)
    P[:, 3] = rng.uniform(2, 15, n)
    P[:, 4] = rng.uniform(2, 10, n)
    P[:, 5] = rng.uniform(0, 360, n)
    P[:, 6] = rng.uniform(5, 90, n)
    for k, v in fixed.items():
        P[:, {"length": 3, "width": 4, "strike": 5, "dip": 6}[k]] = v
    half = P[:, 4] / 2 * numpy.sin(numpy.radians(P[:, 6]))
    P[:, 2] = half + (rng.uniform(0.5, 15, n) if top is None else top)
    return P


def triangles(P):
    # the two triangles of each rectangle, (2n, 3, 3): all first triangles, then all second
    st, dp = numpy.radians(P[:, 5]), numpy.radians(P[:, 6])
    s = numpy.stack([numpy.sin(st), numpy.cos(st), 0 * st], -1)
    d = numpy.stack([numpy.cos(st) * numpy.cos(dp), -numpy.sin(st) * numpy.cos(dp), -numpy.sin(dp)], -1)
    c = numpy.stack([P[:, 0], P[:, 1], -P[:, 2]], -1)
    hl, hw = (P[:, 3] / 2)[:, None], (P[:, 4] / 2)[:, None]
    C = numpy.stack([c - hl * s - hw * d, c + hl * s - hw * d, c + hl * s + hw * d, c - hl * s + hw * d], 1)
    # counterclockwise seen from above: the normal points up, so the slip signs are okada's
    return numpy.ascontiguousarray(numpy.concatenate([C[:, [0, 2, 1]], C[:, [0, 3, 2]]], 0))


def stations(rng, n=400, extent=40.0, depth=0.0):
    # random stations at the surface, or down to {depth}
    obs = numpy.empty((n, 3))
    obs[:, :2] = rng.uniform(-extent, extent, (n, 2))
    obs[:, 2] = -rng.uniform(0, depth, n) if depth > 0 else 0.0
    return obs


def symmetric(G):
    # the strain [xx, yy, zz, xy, xz, yz] of displacement gradients G[..., i, j] = du_i/dx_j
    return numpy.stack(
        [G[..., 0, 0], G[..., 1, 1], G[..., 2, 2],
         (G[..., 0, 1] + G[..., 1, 0]) / 2, (G[..., 0, 2] + G[..., 2, 0]) / 2, (G[..., 1, 2] + G[..., 2, 1]) / 2],
        -1,
    )


def relerr(a, b):
    # the largest difference relative to the largest value
    return float(numpy.abs(a - b).max() / numpy.abs(b).max())


def run(namespace):
    # run the tests of a module as a script
    tests = [(k, v) for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    for name, test in tests:
        test()
        print(f"  {name}: ok")
    print(f"{len(tests)} tests passed")


# end of file
