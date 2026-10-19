# -*- python -*-
#
# cuokada: elastic dislocations in a half space, on the cpu and the gpu
#
# Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

"""
Displacements and strains of rectangular (Okada 1992) and triangular (Nikkhoo and Walter 2015)
dislocations in an elastic half space, on the cpu (OpenMP) and the gpu (CUDA).

Coordinates are x east, y north, z up, with stations at z <= 0.

Sources:
    rect: (n, 7) [x, y, depth, length, width, strike, dip]: the center of the patch and its depth,
          the strike clockwise from north and the dip to its right in degrees; slip
          [strike, dip, tensile], left lateral, reverse and opening positive
    tri:  (n, 3, 3) vertices; slip [strike, dip, tensile] in the triangle frame of Nikkhoo and
          Walter (2015)

Fields: disp (3) [ux, uy, uz], strain (6) [exx, eyy, ezz, exy, exz, eyz]

Modes:
    *_matrix(obs, src):       (nobs, F, nsrc, 3), the response to unit slips of each source
    *_sum(obs, src, slip):    (nobs, F), the response to all sources with their slips
    *_pairs(obs, src, slip):  (n, F), station i with source i

Devices:
    gpu arrays (cuokada's DeviceArray, anything with a __cuda_array_interface__: cupy, numba,
    torch, pycuda; or a dlpack array on a cuda device: jax) run on the gpu, without copies, and so
    does device="cuda". The launch waits for the stream each input reports. There, numpy arrays in managed memory are
    used in place; other numpy arrays are copied to the gpu, and the result comes back to the
    host. Otherwise numpy arrays run on the cpu. Every array has the same dtype, float64 or
    float32. On the gpu, the launch is asynchronous on {stream}; results on the host are ready.

Gpu memory, without other libraries:
    to_device(a), device_array(shape, dtype), DeviceArray.copy_to_host(), pinned_array(shape,
    dtype) for page-locked host memory, synchronize(), set_device(n), device_count(). Other
    libraries take a DeviceArray through __cuda_array_interface__ or dlpack (from_dlpack).
"""

import numpy

from . import _cuokada as _lib

__all__ = []

FIELDS = {"disp": 3, "strain": 6}
SOURCES = {"rect": 7, "tri": 9}


def has_cuda():
    """Whether the library was built with the cuda backend"""
    return bool(_lib.has_cuda())


def _check(status, what):
    if status != 0:
        raise RuntimeError(f"cuokada.{what}: cuda error {status}")


def device_count():
    """The number of gpus"""
    status, n = _lib.cuda_device_count()
    _check(status, "device_count")
    return n


def set_device(device):
    """Run on gpu {device} from now on, in this thread"""
    _check(_lib.cuda_set_device(int(device)), "set_device")


def synchronize(stream=None):
    """Wait for the work on {stream}, or on the whole gpu"""
    _check(_lib.cuda_synchronize(_stream(stream)), "synchronize")


KINDS = ("host", "pinned", "device", "managed")


def memory_kind(a):
    """Where the data of array {a} lives: host, pinned, device or managed"""
    info, _ = _interface(a)
    if not has_cuda() or info["data"][0] == 0:
        return "host"
    status, kind = _lib.cuda_memory_kind(info["data"][0])
    return KINDS[kind]


class DeviceArray:
    """
    A contiguous array in gpu memory, freed with the array; other libraries read it through its
    __cuda_array_interface__
    """

    def __init__(self, shape, dtype=numpy.float64):
        self.shape = tuple(int(n) for n in numpy.atleast_1d(shape))
        self.dtype = numpy.dtype(dtype)
        self.size = int(numpy.prod(self.shape))
        self.nbytes = self.size * self.dtype.itemsize
        self.ptr = 0
        # the stream of the last write, for the readers to wait on
        self.stream = None
        status, self.device = _lib.cuda_get_device()
        _check(status, "device_array")
        if self.nbytes:
            status, self.ptr = _lib.cuda_malloc(self.nbytes)
            _check(status, "device_array")

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": self.shape,
            "typestr": self.dtype.str,
            "data": (self.ptr, False),
            "strides": None,
            "stream": self.stream,
            "version": 3,
        }

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        if copy:
            raise BufferError("DeviceArray.__dlpack__: no copies")
        if dl_device is not None and tuple(dl_device) != self.__dlpack_device__():
            raise BufferError("DeviceArray.__dlpack__: on another device")
        # the reader's stream waits for the last write; -1 asks for no synchronization
        reader = 1 if stream is None else stream
        if self.stream is not None and reader != -1 and reader != self.stream:
            _check(_lib.cuda_stream_wait(reader, self.stream), "__dlpack__")
        return _lib.dlpack_export(self.ptr, list(self.shape), 8 * self.dtype.itemsize, self.device, self)

    def __dlpack_device__(self):
        return (2, self.device)

    def copy_from_host(self, a, stream=None):
        """Copy the host array {a} in"""
        a = numpy.ascontiguousarray(a, dtype=self.dtype)
        if a.size != self.size:
            raise ValueError(f"copy_from_host: {a.size} values into {self.size}")
        if self.nbytes:
            _check(_lib.cuda_memcpy(self.ptr, a.ctypes.data, self.nbytes, _stream(stream)), "copy_from_host")
            synchronize(stream)
        return self

    def copy_to_host(self, out=None, stream=None):
        """Copy out, into a new numpy array or the contiguous host array {out}"""
        if out is None:
            out = numpy.empty(self.shape, dtype=self.dtype)
        if out.dtype != self.dtype or out.size != self.size or not out.flags.c_contiguous:
            raise ValueError("copy_to_host: out must be contiguous, with the same dtype and size")
        if self.nbytes:
            _check(_lib.cuda_memcpy(out.ctypes.data, self.ptr, self.nbytes, _stream(stream)), "copy_to_host")
            synchronize(stream)
        return out

    def __del__(self):
        if self.ptr:
            _lib.cuda_free(self.ptr)
            self.ptr = 0

    def __repr__(self):
        return f"DeviceArray(shape={self.shape}, dtype={self.dtype})"


def device_array(shape, dtype=numpy.float64):
    """An uninitialized array on the gpu"""
    return DeviceArray(shape, dtype)


def to_device(a, dtype=None, stream=None):
    """A copy of the host array {a} on the gpu"""
    a = numpy.asarray(a, dtype=dtype)
    return DeviceArray(a.shape, a.dtype).copy_from_host(a, stream)


class _Pinned:
    # page-locked host memory, freed with the numpy array that views it
    def __init__(self, shape, dtype):
        dtype = numpy.dtype(dtype)
        nbytes = max(int(numpy.prod(shape)) * dtype.itemsize, 1)
        status, self.ptr = _lib.cuda_malloc_host(nbytes)
        _check(status, "pinned_array")
        self.__array_interface__ = {"shape": tuple(shape), "typestr": dtype.str, "data": (self.ptr, False), "version": 3}

    def __del__(self):
        if self.ptr:
            _lib.cuda_free_host(self.ptr)
            self.ptr = 0


def pinned_array(shape, dtype=numpy.float64):
    """An uninitialized numpy array in page-locked host memory, for fast copies"""
    shape = tuple(int(n) for n in numpy.atleast_1d(shape))
    return numpy.asarray(_Pinned(shape, dtype))


def _interface(a):
    # the array interface of {a}, on the gpu or the host
    cai = getattr(a, "__cuda_array_interface__", None)
    return (cai, True) if cai is not None else (numpy.asarray(a).__array_interface__, False)


def _pointer(a, dtype, shape, name, device):
    # the address of {a}, after checking its dtype, shape and layout
    info, gpu = _interface(a)
    if gpu and device == "cpu":
        raise ValueError(f"{name}: a gpu array on the cpu")
    if numpy.dtype(info["typestr"]) != dtype:
        raise ValueError(f"{name}: dtype {numpy.dtype(info['typestr'])}, expected {dtype}")
    if numpy.prod(info["shape"]) != numpy.prod(shape) or info["shape"][0] != shape[0]:
        raise ValueError(f"{name}: shape {tuple(info['shape'])}, expected {shape}")
    strides = info.get("strides")
    if strides is not None:
        # c contiguous
        expected = numpy.empty(info["shape"], dtype=dtype).strides
        if tuple(strides) != expected and numpy.prod(info["shape"]) > 1:
            raise ValueError(f"{name}: not contiguous")
    return info["data"][0]


def _gpu_empty(like, shape, dtype):
    # an output on the gpu, from the library of {like}, or cuokada's
    library = type(like).__module__.split(".")[0]
    if library == "torch":
        import torch

        return torch.empty(shape, dtype=getattr(torch, dtype.name), device=like.device)
    if library == "numba":
        from numba import cuda

        return cuda.device_array(shape, dtype=dtype)
    if library == "cupy":
        import cupy

        return cupy.empty(shape, dtype=dtype)
    return DeviceArray(shape, dtype)


class _DLPackView:
    # a gpu array of another library, seen through dlpack and kept alive by its capsule
    def __init__(self, a, stream):
        try:
            self.capsule = a.__dlpack__(stream=stream)
        except TypeError:
            self.capsule = a.__dlpack__()
        ptr, shape, strides, code, bits, lanes, _, _ = _lib.dlpack_import(self.capsule)
        if code != 2 or lanes != 1 or bits not in (32, 64):
            raise TypeError("cuokada: dlpack arrays must be float32 or float64")
        dtype = numpy.dtype(f"float{bits}")
        if strides is not None:
            strides = tuple(n * dtype.itemsize for n in strides)
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": dtype.str,
            "data": (ptr, True),
            "strides": strides,
            "version": 3,
        }


def _adopt(a, stream):
    # a dlpack array on a cuda device, as an array with a __cuda_array_interface__
    if a is None or hasattr(a, "__cuda_array_interface__") or not hasattr(a, "__dlpack__"):
        return a
    try:
        kind, _ = a.__dlpack_device__()
    except (AttributeError, TypeError):
        return a
    # cuda and cuda managed memory
    return _DLPackView(a, _stream(stream) or 1) if kind in (2, 13) else a


def _wait_for(a, stream):
    # the launch on {stream} waits for the stream that {a} reports, if any
    other = getattr(a, "__cuda_array_interface__", {}).get("stream")
    if other is not None and other != (_stream(stream) or 1):
        _check(_lib.cuda_stream_wait(_stream(stream), other), "stream_wait")


def _on_host(a):
    # whether {a} is in host memory that the gpu cannot use in place
    return not hasattr(a, "__cuda_array_interface__") and memory_kind(a) in ("host", "pinned")


def _to_gpu(a, dtype, stream):
    # {a} as the gpu can use it: gpu and managed arrays as they are, host arrays copied
    if a is None or hasattr(a, "__cuda_array_interface__"):
        return a
    a = numpy.ascontiguousarray(a, dtype=dtype)
    return to_device(a, stream=stream) if _on_host(a) else a


def _stream(stream):
    # a stream as an integer handle
    if stream is None or isinstance(stream, int):
        return stream or 0
    if hasattr(stream, "ptr"):
        return int(stream.ptr)
    return int(stream.handle)


def _call(shape, field, mode, obs, src, slip, nu, halfspace, out, device, stream):
    obs, src, slip = (_adopt(a, stream) for a in (obs, src, slip))
    # pick the device: gpu arrays run on the gpu
    if device is None:
        device = "cuda" if hasattr(obs, "__cuda_array_interface__") else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device {device!r}: expected 'cpu' or 'cuda'")
    gpu = device == "cuda"
    if gpu and not has_cuda():
        raise RuntimeError("cuokada was built without cuda")
    info, _ = _interface(obs)
    dtype = numpy.dtype(info["typestr"])
    if dtype not in (numpy.float64, numpy.float32):
        dtype = numpy.dtype(numpy.float64)
    # on the cpu, convert the inputs; on the gpu, copy those in host memory
    target = out
    if gpu:
        host = _on_host(obs)
        obs, src, slip = (_to_gpu(a, dtype, stream) for a in (obs, src, slip))
    else:
        obs = numpy.ascontiguousarray(obs, dtype=dtype)
        src = numpy.ascontiguousarray(src, dtype=dtype)
        if slip is not None:
            slip = numpy.ascontiguousarray(slip, dtype=dtype)
    nf, ns = FIELDS[field], SOURCES[shape]
    nobs = _interface(obs)[0]["shape"][0]
    nsrc = _interface(src)[0]["shape"][0]
    if mode == "pairs" and nobs != nsrc:
        raise ValueError(f"pairs: {nobs} stations and {nsrc} sources")
    shape_out = {"matrix": (nobs, nf, nsrc, 3), "sum": (nobs, nf), "pairs": (nobs, nf)}[mode]
    if not gpu:
        out = numpy.empty(shape_out, dtype=dtype) if out is None else out
    elif out is None:
        out = DeviceArray(shape_out, dtype) if host else _gpu_empty(obs, shape_out, dtype)
    elif _on_host(out):
        out = DeviceArray(shape_out, dtype)

    p_obs = _pointer(obs, dtype, (nobs, 3), "obs", device)
    p_src = _pointer(src, dtype, (nsrc, ns), "src", device)
    p_out = _pointer(out, dtype, shape_out, "out", device)
    args = [p_obs]
    if mode in ("matrix", "sum"):
        args.append(nobs)
    args.append(p_src)
    if mode in ("sum", "pairs"):
        if slip is None:
            raise ValueError(f"{mode}: no slip")
        args.append(_pointer(slip, dtype, (nsrc, 3), "slip", device))
    args.append(nsrc)
    args.append(float(nu))
    if shape == "tri":
        args.append(int(bool(halfspace)))
    args.append(p_out)
    if gpu:
        args.append(_stream(stream))
    name = ("cuda_" if gpu else "") + f"{shape}_{field}_{mode}_" + ("d" if dtype == numpy.float64 else "f")
    if gpu:
        for a in (obs, src, slip, out):
            _wait_for(a, stream)
    _check(getattr(_lib, name)(*args), name)
    if isinstance(out, DeviceArray):
        out.stream = _stream(stream) or 1
    # results for the host come back
    if gpu and target is not None and out is not target:
        return out.copy_to_host(target, stream)
    if gpu and target is None and host:
        return out.copy_to_host(stream=stream)
    return out


def _make(shape, field, mode):
    # the public function for one shape, field and mode
    hs = shape == "tri"
    if mode == "matrix":

        def f(obs, src, nu=0.25, *, halfspace=True, out=None, device=None, stream=None):
            return _call(shape, field, mode, obs, src, None, nu, halfspace, out, device, stream)

    else:

        def f(obs, src, slip, nu=0.25, *, halfspace=True, out=None, device=None, stream=None):
            return _call(shape, field, mode, obs, src, slip, nu, halfspace, out, device, stream)

    f.__name__ = f.__qualname__ = f"{shape}_{field}_{mode}"
    what = {"matrix": "per unit slip of each source", "sum": "of all sources", "pairs": "of source i at station i"}
    f.__doc__ = f"The {field} {what[mode]}, for {shape} sources" + (
        "; halfspace=False for a full space" if hs else ""
    )
    return f


for _shape in SOURCES:
    for _field in FIELDS:
        for _mode in ("matrix", "sum", "pairs"):
            _f = _make(_shape, _field, _mode)
            globals()[_f.__name__] = _f
            __all__.append(_f.__name__)
__all__ += ["has_cuda", "device_count", "set_device", "synchronize", "memory_kind", "DeviceArray",
            "device_array", "to_device", "pinned_array"]

# end of file
