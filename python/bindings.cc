// -*- c++ -*-
//
// the python bindings of the c interface: pointers as integers, without the gil
//
// Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

#include <cstdint>
#include <cstring>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuokada/cuokada.h>

namespace py = pybind11;

namespace {

// pointers cross as integers, everything else as is
template <typename A>
struct arg {
    using type = A;
    static A conv(type a) { return a; }
};

template <typename T>
struct arg<T *> {
    using type = std::uintptr_t;
    static T * conv(type a) { return reinterpret_cast<T *>(a); }
};

template <typename R, typename... A>
auto wrap(R (*f)(A...)) {
    return [f](typename arg<A>::type... a) {
        py::gil_scoped_release release;
        return f(arg<A>::conv(a)...);
    };
}

// the dlpack structures, as in dlpack.h, whose abi is stable
struct DLDevice {
    std::int32_t device_type;
    std::int32_t device_id;
};
struct DLDataType {
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
};
struct DLTensor {
    void * data;
    DLDevice device;
    std::int32_t ndim;
    DLDataType dtype;
    std::int64_t * shape;
    std::int64_t * strides;
    std::uint64_t byte_offset;
};
struct DLManagedTensor {
    DLTensor dl_tensor;
    void * manager_ctx;
    void (*deleter)(DLManagedTensor *);
};
struct DLPackVersion {
    std::uint32_t major;
    std::uint32_t minor;
};
struct DLManagedTensorVersioned {
    DLPackVersion version;
    void * manager_ctx;
    void (*deleter)(DLManagedTensorVersioned *);
    std::uint64_t flags;
    DLTensor dl_tensor;
};
constexpr std::int32_t kDLCUDA = 2;
constexpr std::uint8_t kDLFloat = 2;

// an exported tensor, holding a reference to the array that owns the memory
struct Exported {
    DLManagedTensor tensor;
    std::vector<std::int64_t> shape;
    PyObject * owner;
};

void exported_deleter(DLManagedTensor * t) {
    auto * e = static_cast<Exported *>(t->manager_ctx);
    py::gil_scoped_acquire gil;
    Py_XDECREF(e->owner);
    delete e;
}

// a capsule nobody consumed still owns its tensor
void capsule_destructor(PyObject * capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto * t = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor"));
        if (t && t->deleter) t->deleter(t);
    }
}

} // namespace

PYBIND11_MODULE(_cuokada, m) {
    m.doc() = "the c interface of cuokada, with pointers as integers";
    m.def("has_cuda", &cuokada_has_cuda);
    // gpu memory; outputs come back as (status, value)
    m.def("cuda_device_count", [] {
        int n = 0;
        int status = cuokada_cuda_device_count(&n);
        return py::make_tuple(status, n);
    });
    m.def("cuda_set_device", &cuokada_cuda_set_device);
    m.def("cuda_get_device", [] {
        int device = 0;
        int status = cuokada_cuda_get_device(&device);
        return py::make_tuple(status, device);
    });
    m.def("cuda_stream_wait", wrap(&cuokada_cuda_stream_wait));
    // a dlpack capsule for a contiguous float array on gpu {device}, kept alive by {owner}
    m.def("dlpack_export", [](std::uintptr_t ptr, std::vector<std::int64_t> shape, int bits, int device,
                              py::object owner) {
        auto * e = new Exported;
        e->shape = std::move(shape);
        e->owner = owner.release().ptr();
        DLTensor & t = e->tensor.dl_tensor;
        t.data = reinterpret_cast<void *>(ptr);
        t.device = {kDLCUDA, device};
        t.ndim = static_cast<std::int32_t>(e->shape.size());
        t.dtype = {kDLFloat, static_cast<std::uint8_t>(bits), 1};
        t.shape = e->shape.data();
        t.strides = nullptr;
        t.byte_offset = 0;
        e->tensor.manager_ctx = e;
        e->tensor.deleter = exported_deleter;
        return py::reinterpret_steal<py::object>(PyCapsule_New(&e->tensor, "dltensor", capsule_destructor));
    });
    // the address, shape, strides (in elements, or None), dtype and device of a dlpack capsule;
    // the capsule is not consumed, so its producer frees the tensor when it goes away
    m.def("dlpack_import", [](py::capsule capsule) {
        const char * name = PyCapsule_GetName(capsule.ptr());
        DLTensor * t = nullptr;
        if (name && std::strcmp(name, "dltensor") == 0) {
            t = &static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule.ptr(), name))->dl_tensor;
        } else if (name && std::strcmp(name, "dltensor_versioned") == 0) {
            t = &static_cast<DLManagedTensorVersioned *>(PyCapsule_GetPointer(capsule.ptr(), name))->dl_tensor;
        } else {
            throw py::value_error("dlpack_import: not an unconsumed dlpack capsule");
        }
        py::tuple shape(t->ndim);
        for (int i = 0; i < t->ndim; ++i) shape[i] = t->shape[i];
        py::object strides = py::none();
        if (t->strides) {
            py::tuple st(t->ndim);
            for (int i = 0; i < t->ndim; ++i) st[i] = t->strides[i];
            strides = st;
        }
        auto data = reinterpret_cast<std::uintptr_t>(t->data) + t->byte_offset;
        return py::make_tuple(data, shape, strides, int(t->dtype.code), int(t->dtype.bits), int(t->dtype.lanes),
                              t->device.device_type, t->device.device_id);
    });
    m.def("cuda_malloc", [](std::int64_t bytes) {
        void * p = nullptr;
        int status = cuokada_cuda_malloc(&p, bytes);
        return py::make_tuple(status, reinterpret_cast<std::uintptr_t>(p));
    });
    m.def("cuda_malloc_host", [](std::int64_t bytes) {
        void * p = nullptr;
        int status = cuokada_cuda_malloc_host(&p, bytes);
        return py::make_tuple(status, reinterpret_cast<std::uintptr_t>(p));
    });
    m.def("cuda_free", wrap(&cuokada_cuda_free));
    m.def("cuda_free_host", wrap(&cuokada_cuda_free_host));
    m.def("cuda_memcpy", wrap(&cuokada_cuda_memcpy));
    m.def("cuda_synchronize", wrap(&cuokada_cuda_synchronize));
    m.def("cuda_memory_kind", [](std::uintptr_t p) {
        int kind = 0;
        int status = cuokada_cuda_memory_kind(reinterpret_cast<const void *>(p), &kind);
        return py::make_tuple(status, kind);
    });
    m.def("rect_disp_matrix_d", wrap(&cuokada_rect_disp_matrix_d));
    m.def("rect_disp_sum_d", wrap(&cuokada_rect_disp_sum_d));
    m.def("rect_disp_pairs_d", wrap(&cuokada_rect_disp_pairs_d));
    m.def("rect_strain_matrix_d", wrap(&cuokada_rect_strain_matrix_d));
    m.def("rect_strain_sum_d", wrap(&cuokada_rect_strain_sum_d));
    m.def("rect_strain_pairs_d", wrap(&cuokada_rect_strain_pairs_d));
    m.def("tri_disp_matrix_d", wrap(&cuokada_tri_disp_matrix_d));
    m.def("tri_disp_sum_d", wrap(&cuokada_tri_disp_sum_d));
    m.def("tri_disp_pairs_d", wrap(&cuokada_tri_disp_pairs_d));
    m.def("tri_strain_matrix_d", wrap(&cuokada_tri_strain_matrix_d));
    m.def("tri_strain_sum_d", wrap(&cuokada_tri_strain_sum_d));
    m.def("tri_strain_pairs_d", wrap(&cuokada_tri_strain_pairs_d));
    m.def("rect_disp_matrix_f", wrap(&cuokada_rect_disp_matrix_f));
    m.def("rect_disp_sum_f", wrap(&cuokada_rect_disp_sum_f));
    m.def("rect_disp_pairs_f", wrap(&cuokada_rect_disp_pairs_f));
    m.def("rect_strain_matrix_f", wrap(&cuokada_rect_strain_matrix_f));
    m.def("rect_strain_sum_f", wrap(&cuokada_rect_strain_sum_f));
    m.def("rect_strain_pairs_f", wrap(&cuokada_rect_strain_pairs_f));
    m.def("tri_disp_matrix_f", wrap(&cuokada_tri_disp_matrix_f));
    m.def("tri_disp_sum_f", wrap(&cuokada_tri_disp_sum_f));
    m.def("tri_disp_pairs_f", wrap(&cuokada_tri_disp_pairs_f));
    m.def("tri_strain_matrix_f", wrap(&cuokada_tri_strain_matrix_f));
    m.def("tri_strain_sum_f", wrap(&cuokada_tri_strain_sum_f));
    m.def("tri_strain_pairs_f", wrap(&cuokada_tri_strain_pairs_f));
    m.def("cuda_rect_disp_matrix_d", wrap(&cuokada_cuda_rect_disp_matrix_d));
    m.def("cuda_rect_disp_sum_d", wrap(&cuokada_cuda_rect_disp_sum_d));
    m.def("cuda_rect_disp_pairs_d", wrap(&cuokada_cuda_rect_disp_pairs_d));
    m.def("cuda_rect_strain_matrix_d", wrap(&cuokada_cuda_rect_strain_matrix_d));
    m.def("cuda_rect_strain_sum_d", wrap(&cuokada_cuda_rect_strain_sum_d));
    m.def("cuda_rect_strain_pairs_d", wrap(&cuokada_cuda_rect_strain_pairs_d));
    m.def("cuda_tri_disp_matrix_d", wrap(&cuokada_cuda_tri_disp_matrix_d));
    m.def("cuda_tri_disp_sum_d", wrap(&cuokada_cuda_tri_disp_sum_d));
    m.def("cuda_tri_disp_pairs_d", wrap(&cuokada_cuda_tri_disp_pairs_d));
    m.def("cuda_tri_strain_matrix_d", wrap(&cuokada_cuda_tri_strain_matrix_d));
    m.def("cuda_tri_strain_sum_d", wrap(&cuokada_cuda_tri_strain_sum_d));
    m.def("cuda_tri_strain_pairs_d", wrap(&cuokada_cuda_tri_strain_pairs_d));
    m.def("cuda_rect_disp_matrix_f", wrap(&cuokada_cuda_rect_disp_matrix_f));
    m.def("cuda_rect_disp_sum_f", wrap(&cuokada_cuda_rect_disp_sum_f));
    m.def("cuda_rect_disp_pairs_f", wrap(&cuokada_cuda_rect_disp_pairs_f));
    m.def("cuda_rect_strain_matrix_f", wrap(&cuokada_cuda_rect_strain_matrix_f));
    m.def("cuda_rect_strain_sum_f", wrap(&cuokada_cuda_rect_strain_sum_f));
    m.def("cuda_rect_strain_pairs_f", wrap(&cuokada_cuda_rect_strain_pairs_f));
    m.def("cuda_tri_disp_matrix_f", wrap(&cuokada_cuda_tri_disp_matrix_f));
    m.def("cuda_tri_disp_sum_f", wrap(&cuokada_cuda_tri_disp_sum_f));
    m.def("cuda_tri_disp_pairs_f", wrap(&cuokada_cuda_tri_disp_pairs_f));
    m.def("cuda_tri_strain_matrix_f", wrap(&cuokada_cuda_tri_strain_matrix_f));
    m.def("cuda_tri_strain_sum_f", wrap(&cuokada_cuda_tri_strain_sum_f));
    m.def("cuda_tri_strain_pairs_f", wrap(&cuokada_cuda_tri_strain_pairs_f));
}

// end of file
