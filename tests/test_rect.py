# -*- python -*-
#
# rectangles: against okada4py, their two triangles, and their own finite differences
#
# Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

import numpy

from helpers import cuokada, patches, triangles, stations, symmetric, relerr, run

rng = numpy.random.default_rng(1)


def okada4py(obs, P, nu=0.25):
    # displacement (nobs, 3, n, 3) and gradients (nobs, 3, 3, n, 3) from okada4py, a patch at a time
    import okada4py as ok

    n = len(P)
    U = numpy.empty((len(obs), 3, n, 3))
    D = numpy.empty((len(obs), 3, 3, n, 3))
    xyz = [numpy.ascontiguousarray(obs[:, i]) for i in range(3)]
    for j in range(n):
        for s in range(3):
            slip = [numpy.zeros(1) for _ in range(3)]
            slip[s][0] = 1.0
            geometry = [P[j : j + 1, k].copy() for k in (0, 1, 2, 3, 4, 6, 5)]
            u, d, *_ = ok.okada92(*xyz, *geometry, *slip, 30e9, nu)
            U[:, :, j, s] = u.reshape(-1, 3)
            D[:, :, :, j, s] = d.reshape(-1, 3, 3)
    return U, D


def _have_okada4py():
    try:
        import okada4py  # noqa: F401
    except ImportError:
        return False
    return True


def test_okada4py():
    # displacements and strains at and below the surface, for random, vertical and shallow patches
    if not _have_okada4py():
        print("  okada4py not installed, skipped")
        return
    obs = numpy.concatenate([stations(rng, 300), stations(rng, 100, depth=8.0)])
    for P in (patches(20, rng), patches(10, rng, dip=90.0), patches(10, rng, dip=10.0), patches(10, rng, top=0.1)):
        U, D = okada4py(obs, P)
        E = symmetric(numpy.moveaxis(D, (1, 2), (-2, -1)))
        E = numpy.moveaxis(E, -1, 1)
        # near vertical patches lose digits to the 1/cos(dip)^2 of okada's part B, in both codes
        assert relerr(cuokada.rect_disp_matrix(obs, P), U) < 1e-8
        assert relerr(cuokada.rect_strain_matrix(obs, P), E) < 1e-8


def test_triangles():
    # a rectangle is its two triangles, counterclockwise seen from above
    obs = numpy.concatenate([stations(rng, 200), stations(rng, 100, depth=8.0)])
    for P in (patches(10, rng), patches(10, rng, dip=90.0), patches(10, rng, top=0.0)):
        T = triangles(P)
        n = len(P)
        for field in ("disp", "strain"):
            R = getattr(cuokada, f"rect_{field}_matrix")(obs, P)
            G = getattr(cuokada, f"tri_{field}_matrix")(obs, T)
            G = G[:, :, :n] + G[:, :, n:]
            assert relerr(R, G) < 1e-8, (field, relerr(R, G))


def test_finite_differences():
    # the strain against central differences of the displacement, below the surface
    P = patches(5, rng)
    obs = stations(rng, 100, depth=8.0)
    obs[:, 2] = numpy.minimum(obs[:, 2], -1e-3)
    h = 1e-5
    G = numpy.empty((len(obs), 3, 3, len(P), 3))
    for j in range(3):
        step = numpy.zeros(3)
        step[j] = h
        up = cuokada.rect_disp_matrix(obs + step, P)
        um = cuokada.rect_disp_matrix(obs - step, P)
        G[:, :, j] = (up - um) / (2 * h)
    E = numpy.moveaxis(symmetric(numpy.moveaxis(G, (1, 2), (-2, -1))), -1, 1)
    assert relerr(cuokada.rect_strain_matrix(obs, P), E) < 1e-7


def test_modes():
    # sum and pairs against the matrix, and single against double precision
    P = patches(30, rng)
    obs = stations(rng, 200, depth=5.0)
    slip = rng.normal(size=(len(P), 3))
    for field in ("disp", "strain"):
        M = getattr(cuokada, f"rect_{field}_matrix")(obs, P)
        S = getattr(cuokada, f"rect_{field}_sum")(obs, P, slip)
        assert relerr(S, numpy.einsum("ocjs,js->oc", M, slip)) < 1e-12
        idx = rng.integers(0, len(P), len(obs))
        A = getattr(cuokada, f"rect_{field}_pairs")(obs, P[idx], slip[idx])
        assert relerr(A, numpy.einsum("ocs,os->oc", M[numpy.arange(len(obs)), :, idx], slip[idx])) < 1e-12
        M32 = getattr(cuokada, f"rect_{field}_matrix")(obs.astype(numpy.float32), P.astype(numpy.float32))
        assert M32.dtype == numpy.float32
        assert relerr(M32, M) < 1e-4, (field, relerr(M32, M))


def test_single_steep():
    # single precision near vertical, where part B divides by cos(dip)^2 and switches to double
    obs = numpy.concatenate([stations(rng, 200, extent=50.0), stations(rng, 100, depth=8.0)])
    for dip in (80.0, 84.0, 86.0, 88.0, 89.9, 89.999, 90.0):
        P = patches(10, rng, dip=dip)
        for field in ("disp", "strain"):
            f = getattr(cuokada, f"rect_{field}_matrix")
            err = relerr(f(obs.astype(numpy.float32), P.astype(numpy.float32)), f(obs, P))
            assert err < 1e-4, (dip, field, err)


def test_single_edge_extension():
    # single precision on the strike extension of the top edge, where r + xi would cancel
    P = numpy.array([[0.0, 0.0, 10.0, 12.0, 8.0, 0.0, 40.0]])
    cd, sd = numpy.cos(numpy.radians(40.0)), numpy.sin(numpy.radians(40.0))
    beyond = numpy.linspace(5.0, 40.0, 50)
    obs = numpy.stack([-4.0 * cd + 0 * beyond, -(6.0 + beyond), -(10.0 - 4.0 * sd) + 0 * beyond], -1)
    # off the line by about 0.1, outside the singular zone of dc3d (r + xi < 1e-6)
    obs += rng.normal(scale=0.1, size=obs.shape)
    for field in ("disp", "strain"):
        f = getattr(cuokada, f"rect_{field}_matrix")
        err = relerr(f(obs.astype(numpy.float32), P.astype(numpy.float32)), f(obs, P))
        assert err < 1e-4, (field, err)


def test_above_the_surface():
    U = cuokada.rect_disp_matrix(numpy.array([[0.0, 0.0, 1.0]]), patches(1, rng))
    assert numpy.isnan(U).all()


if __name__ == "__main__":
    run(dict(globals()))

# end of file
