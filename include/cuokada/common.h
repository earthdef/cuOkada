// -*- c++ -*-
//
// cuokada: elastic dislocations in a half space, on the cpu and the gpu
//
// Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

#pragma once

// the same source compiles for the host, nvcc and nvrtc
#if defined(__CUDACC__)
#define CUOKADA_HD __host__ __device__
#else
#define CUOKADA_HD
#endif

#if !defined(__CUDACC_RTC__)
#include <cmath>
#include <cstdint>
#include <limits>
#endif

namespace cuokada {

#if !defined(__CUDACC_RTC__)
using std::acos;
using std::atan;
using std::atan2;
using std::cos;
using std::fabs;
using std::log;
using std::sin;
using std::sqrt;
using std::tan;
#endif

// small vectors, as in cutde
template <typename T> struct V2 { T x, y; };
template <typename T> struct V3 { T x, y, z; };
template <typename T> struct V6 { T x, y, z, a, b, c; };

// the machine epsilon and a quiet nan of each precision
template <typename T> CUOKADA_HD inline T eps();
template <> CUOKADA_HD inline double eps<double>() { return 2.220446049250313e-16; }
template <> CUOKADA_HD inline float eps<float>() { return 1.1920929e-07f; }

#if defined(__CUDACC_RTC__)
template <typename T> CUOKADA_HD inline T nan() { return T(__longlong_as_double(0x7ff8000000000000ULL)); }
#else
template <typename T> CUOKADA_HD inline T nan() { return T(NAN); }
#endif

// the number of components of each field
enum Field { DISP = 3, STRAIN = 6 };

} // namespace cuokada

// end of file
