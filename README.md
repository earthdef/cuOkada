# cuokada

Elastic dislocation Green's functions, displacements and strains, on the cpu and the gpu, with
C++, Fortran and Python interfaces:

- **rectangles** in a half space, after Okada (1992), written from the paper as templated
  host/device C++;
- **triangles** in a full or a half space, after Nikkhoo and Walter (2015), ported from
  [cutde](https://github.com/tbenthompson/cutde) (MIT).

Its core is a few header-only C++ files that compile with a host compiler, nvcc or nvrtc. On
top of them is a C interface over raw pointers, so C, C++, Fortran and CUDA codes can call it
without Python or any framework. Python bindings are optional.

## Conventions

Coordinates are x east, y north, z up, with stations at z <= 0. Lengths can be in any unit;
the singularity tolerances are 1e-6 of that unit, as in Okada's `dc3d`. Displacements come out
in the unit of slip. All arrays are contiguous and row major.

| source | numbers per source | slip `[s, d, t]` |
|---|---|---|
| `rect` | `[x, y, depth, length, width, strike, dip]`: the center of the patch and its depth (> 0), the strike clockwise from north and the dip to its right, in degrees | left-lateral, reverse and opening are positive, as in Okada (1992) |
| `tri` | `[x0, y0, z0, x1, y1, z1, x2, y2, z2]` | along the strike, dip and normal of the triangle frame, set by the vertex order; see [Triangles](#triangles) |

| field | components |
|---|---|
| `disp` | `[ux, uy, uz]` |
| `strain` | `[exx, eyy, ezz, exy, exz, eyz]` |

| mode | output | use |
|---|---|---|
| `matrix` | `[nobs, F, nsrc, 3]` | Green's functions: unit slips of each source |
| `sum` | `[nobs, F]` | forward models: all sources with their slips |
| `pairs` | `[n, F]` | station `i` with source `i`, e.g. many geometries at once |

Stations on a rectangle edge get zeros (as in `dc3d`); stations above the surface get NaN.

## Triangles

The order of the vertices `(v0, v1, v2)` sets the frame of a triangle, as in Nikkhoo and
Walter (2015):

- the normal **n** follows the right-hand rule: it points to the side from which the vertices
  run counterclockwise;
- the strike **s** = **ẑ** × **n** is horizontal. For a horizontal triangle, it is north when
  **n** points up and south when **n** points down;
- the dip **d** = **n** × **s** points up the dip when **n** points up.

The slip `[strike, dip, tensile]` is the displacement of the block on the side of **n**
relative to the other block, along **s**, **d** and **n**.

**Okada's signs.** Order the vertices counterclockwise as seen from above, so that **n** points
up, into the hanging wall. For a vertical triangle, point **n** to the right of the strike you
want. Then strike-slip is left-lateral, dip-slip is reverse (thrust) and tensile is opening,
as for rectangles. With the corners of a rectangle taken as top start, top end, bottom end
and bottom start (`c0, c1, c2, c3`, along strike, then down dip), the two triangles
`(c0, c2, c1)` and `(c0, c3, c2)` reproduce it exactly. The opposite order flips **n** and
**s**: strike-slip and opening keep their meaning, but dip-slip changes sign.

A mesh from csi, gmsh or similar may come with either orientation. To make every normal point
up (orient vertical triangles by hand, as their normals are horizontal):

```python
n = numpy.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
down = n[:, 2] < 0
tris[down] = tris[down][:, [0, 2, 1]]
```

**Full or half space.** `halfspace = 1` (the default in Python) puts a free surface at z = 0;
all vertices must be at z <= 0, which is not checked. `halfspace = 0` gives a full space,
with triangles and stations anywhere. Rectangles are half space only, as in Okada; for a
rectangle in a full space, use its two triangles with `halfspace = 0`.

```python
G = cuokada.tri_disp_matrix(obs, tris)                     # half space, (nobs, 3, ntri, 3)
G = cuokada.tri_disp_matrix(obs, tris, halfspace=False)    # full space
u = cuokada.tri_strain_sum(obs, tris, slips)               # all triangles with their slips
```

```c
cuokada_tri_disp_matrix_d(obs, nobs, tris, ntri, 0.25, 1, G);    /* half space */
```

**Singular points.** Stations exactly on an edge or a vertex get NaN. Stations inside the
triangle, in its plane, get the average of the two sides.

**Cost.** In a half space, each triangle takes three angular dislocations, their image and a
free-surface correction, so it costs far more than an Okada rectangle; see
[Performance](#performance).

**Why a port of cutde.** cutde is a Python package. On the gpu, it compiles its kernels at
runtime from mako templates through pycuda or OpenCL, and copies its arrays between the host
and the gpu on every call. cuokada ports its triangle formulas unchanged, for:

- **easy integration:** the same header-only C++ and C interface as the rectangles, callable
  from C, C++, Fortran, CUDA and Python, with no pycuda;
- **arrays in place:** device or managed pointers, with no copies to and from the host, on a
  given CUDA stream;
- **single precision:** the constants in the formulas take the precision of the kernel, so
  float32 stays in float, and runs much faster on the gpu than cutde's kernels, with the same
  accuracy.

The port is generated by [tools/port_cutde.py](tools/port_cutde.py) from an installed cutde,
and the tests compare the two to 1e-12 in every mode. To pick up a fix in cutde, update it,
rerun the script and the tests. cutde's hierarchical-matrix tools (block-wise products and
adaptive cross approximation) are not ported; use cutde itself for those.

## Precision

Single precision computes in float throughout, except where Okada's formulas would lose too
many digits in float. There the rectangles switch to double automatically:

- **near-vertical patches**, with cos(dip) < 0.1 (dip above about 84.3°), where part B divides
  by cos²(dip). Exactly vertical patches have their own closed form and stay in float;
- **stations close to the extension of a patch edge**, where Okada's corner terms nearly
  cancel: (R + ξ)/R or (R + η)/R below 1e-4.

On the gpu, a warp shares one patch in the single precision rectangle matrix, so only the
warps on such patches pay for double. The single precision cuda kernels use fast math
(`CUOKADA_FAST_MATH`); it costs no accuracy once those cases run in double.

Okada's sums R + ξ, R + η and R + d are computed without cancellation, as (η² + q²)/(R − ξ)
and so on, when the second term is negative.

## Interfaces

C ([cuokada.h](include/cuokada/cuokada.h)):

```c
#include <cuokada/cuokada.h>
// the displacements at nobs stations from nsrc rectangles with their slips
cuokada_rect_disp_sum_d(obs, nobs, rects, slips, nsrc, 0.25, u);
// on the gpu, with device or managed pointers, asynchronous on a stream
cuokada_cuda_rect_disp_sum_d(d_obs, nobs, d_rects, d_slips, nsrc, 0.25, d_u, stream);
```

Every function comes in double (`_d`) and single (`_f`) precision and returns 0 on success; the
cuda ones return the `cudaError_t` of the launch.

Codes without cuda arrays of their own can manage gpu memory through the same interface:

```c
void *d_obs;
cuokada_cuda_malloc(&d_obs, nobs * 3 * sizeof(double));
cuokada_cuda_memcpy(d_obs, obs, nobs * 3 * sizeof(double), stream);   /* in any direction */
/* ... */
cuokada_cuda_synchronize(stream);
cuokada_cuda_free(d_obs);
```

with `cuokada_cuda_malloc_host` for page-locked host memory, `cuokada_cuda_set_device`,
`cuokada_cuda_device_count` and `cuokada_cuda_memory_kind` (host, page-locked, device or
managed).

Fortran ([cuokada.f90](fortran/cuokada.f90)), through `iso_c_binding`:

```fortran
use cuokada
real(c_double) :: obs(3, nobs), rects(7, nsrc), slips(3, nsrc), u(3, nobs)
status = cuokada_rect_disp_sum_d(obs, int(nobs, c_int64_t), rects, slips, int(nsrc, c_int64_t), 0.25d0, u)
```

The gpu memory functions are there too, with `c_loc` for host arrays.

Python:

```python
import cuokada
G = cuokada.rect_disp_matrix(obs, rects)                  # numpy: on the cpu
G = cuokada.rect_disp_matrix(obs, rects, device="cuda")   # numpy in and out, computed on the gpu

d_obs, d_rects = cuokada.to_device(obs), cuokada.to_device(rects)
G = cuokada.rect_disp_matrix(d_obs, d_rects)              # a DeviceArray, left on the gpu
G.copy_to_host()
```

Arrays of other libraries run on the gpu as they are, without copies:

- those with a `__cuda_array_interface__` (cupy, numba, torch, pycuda, jax); the result comes
  back in the same library for cupy, numba and torch, as a `DeviceArray` otherwise;
- those with only dlpack, on a cuda device;
- numpy arrays in managed memory, such as AlTar's.

The launch waits for the stream that each input reports; cupy, for one, reports its current
stream, so call cuokada inside the same `with stream:`. A `DeviceArray` goes to other libraries
through `__cuda_array_interface__` (`cupy.asarray`, `torch.as_tensor`) or dlpack
(`jax.numpy.from_dlpack`, `torch.from_dlpack`). `pinned_array` gives numpy arrays in
page-locked memory, for faster copies. cuokada needs none of these libraries.

C++ and CUDA codes can also call the evaluators in
[kernels.h](include/cuokada/kernels.h) from their own kernels.

## Build

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX    # finds cuda, openmp, fortran, pybind11
cmake --build build -j2
ctest --test-dir build
cmake --install build --prefix $CONDA_PREFIX
```

Options: `CUOKADA_CUDA`, `CUOKADA_FAST_MATH`, `CUOKADA_PYTHON`, `CUOKADA_FORTRAN`,
`CUOKADA_TESTS` (all `ON`), and
`CMAKE_CUDA_ARCHITECTURES` (default `native`). `pip install .` builds just the Python module.

Two files are generated by scripts in [tools/](tools); edit the scripts, not the files, and
rerun them:

```bash
python tools/port_cutde.py include/cuokada/tde.h                            # from an installed cutde
python tools/gen_fortran.py include/cuokada/cuokada.h > fortran/cuokada.f90  # after changing the c interface
```

## Tests

- **rect:** against okada4py, for displacement and strain at and below the surface; against
  its own two triangles; strain against finite differences of displacement; single against
  double precision near vertical and near the extension of an edge.
- **tri:** against cutde, in a full and a half space, in every mode.
- **cuda:** against the cpu, in every mode and both precisions, with cuokada's gpu arrays; the
  copies from and to host and page-locked memory; numba's arrays and managed memory, if
  numba is installed.
- **interop:** cupy and jax arrays in and out, through `__cuda_array_interface__` and dlpack,
  and inputs written on another stream; each part runs if its library is installed.
- **C and Fortran:** the interfaces, and the gpu memory functions from C.

In double precision, agreement is at round-off. The exception is near-vertical rectangles,
where the 1/cos²(dip) terms of Okada's part B cost digits in any implementation. In single
precision, rectangles stay within 4e-5 of double, relative to the peak. Triangles stay within
about 1e-3, worst far from the source, as in cutde.

## Performance

A 400 × 2500 × 3 displacement matrix of rectangles with random dips (5–90°), or of their
800 triangles.

On the cpu of a laptop (16 cores):

| | threads | float64 | float32 |
|---|---|---|---|
| okada4py | 1 | 5.2 s | |
| cuokada rect | 1 | 0.41 s | 0.36 s |
| cuokada rect | 8 | 0.065 s | 0.068 s |
| cutde, two triangles | 8 | 1.84 s | |
| cuokada tri | 8 | 1.33 s | 1.16 s |

On a V100 (PCIe, 16 GB):

| | float64 | float32 |
|---|---|---|
| rect, on the gpu | 1.3 ms | 0.7 ms |
| rect, from and to pinned memory | 15 ms | 7.7 ms |
| rect, from and to pageable memory | 26 ms | 13 ms |
| tri, on the gpu | 91 ms | 4.9 ms |
| tri, from and to pinned memory | 114 ms | 18 ms |
| tri, from and to pageable memory | 143 ms | 29 ms |

"On the gpu" has the inputs and the output there already. The other rows copy the inputs in
and the matrix out, 36 to 144 MB, which takes most of their time in float32. The V100 runs
double precision at half its single-precision rate. Most gaming and workstation cards run it
at 1/64, where float32 is the better choice.

The cpu timings vary by up to about 40% between runs on the laptop; the gpu timings by a few
percent. Run them with `python bench/bench.py` (`--gpu` for the gpu rows only) and
`python bench/tri_gpu.py`, for the triangles in every mode.

**CPU or GPU?**

- **The cpu is enough for small or occasional work.** The Green's functions of a few hundred
  patches take tens of milliseconds. The gpu adds the copies, and a one-time CUDA start-up in
  each program.
- **The gpu pays off for heavy work.** That means repeated calls (inversions that resample
  the geometry with `pairs`, forward models in MCMC, many scenarios) and triangle meshes: in
  float32, 40 to 60 times faster than the laptop's cpu even with the copies. It also pays off for
  codes whose data already live on the gpu, such as AlTar's CUDA models or a gpu tsunami
  solver, where the copies disappear.
- **Precision.** On gaming and workstation cards, use float32; it is safe for rectangles, as
  the steep and near-edge cases switch to double. On data-center cards (V100, A100, H100),
  float64 costs little.
- **Copies.** Keep large matrices on the gpu where you can, use `sum` when only the total is
  needed, and pinned memory for what comes back. `sum` runs one thread per station, so it
  needs thousands of stations to fill a large gpu.

## References

- Okada, Y. (1992), Internal deformation due to shear and tensile faults in a half-space,
  BSSA 82(2), 1018–1040.
- Nikkhoo, M., and T. R. Walter (2015), Triangular dislocation: an analytical, artefact-free
  solution, GJI 201(2), 1119–1141.
- Thompson, T. B., cutde, https://github.com/tbenthompson/cutde.

## AI assistance

This project was developed with the assistance of AI, using Claude Code (Anthropic), for its
code, tests and documentation. Its results are checked by the tests above, against okada4py,
cutde and finite differences.

## License

Copyright (c) 2026 California Institute of Technology. MIT; see [LICENSE](LICENSE), which also
carries the cutde notice for [tde.h](include/cuokada/tde.h).
