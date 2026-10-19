# -*- python -*-
#
# triangles: against cutde, in a full and a half space
#
# Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

import numpy

from helpers import cuokada, patches, triangles, stations, relerr, run

rng = numpy.random.default_rng(2)


def _cutde():
    try:
        import cutde.fullspace
        import cutde.halfspace
    except ImportError:
        return None
    return {True: cutde.halfspace, False: cutde.fullspace}


def test_cutde():
    # matrix, sum and pairs, for displacement and strain, against the same functions of cutde
    cutde = _cutde()
    if cutde is None:
        print("  cutde not installed, skipped")
        return
    T = triangles(patches(15, rng))
    obs = numpy.concatenate([stations(rng, 200), stations(rng, 100, depth=8.0)])
    slip = rng.normal(size=(len(T), 3))
    idx = rng.integers(0, len(T), len(obs))
    for halfspace, module in cutde.items():
        for field in ("disp", "strain"):
            M = getattr(cuokada, f"tri_{field}_matrix")(obs, T, halfspace=halfspace)
            assert relerr(M, getattr(module, f"{field}_matrix")(obs, T, 0.25)) < 1e-12
            S = getattr(cuokada, f"tri_{field}_sum")(obs, T, slip, halfspace=halfspace)
            assert relerr(S, getattr(module, f"{field}_free")(obs, T, slip, 0.25)) < 1e-12
            A = getattr(cuokada, f"tri_{field}_pairs")(obs, T[idx], slip[idx], halfspace=halfspace)
            assert relerr(A, getattr(module, field)(obs, T[idx], slip[idx], 0.25)) < 1e-12


if __name__ == "__main__":
    run(dict(globals()))

# end of file
