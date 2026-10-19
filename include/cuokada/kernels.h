// -*- c++ -*-
//
// the per-source evaluators and the work of one thread in each mode, shared by the backends
//
// Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

#pragma once

#include "common.h"
#include "okada.h"
#include "tde.h"

namespace cuokada {

// rectangles, as in okada::rect: 7 numbers per source
template <typename T, int F>
struct Rect {
    static constexpr int SRC = okada::RECT;
    static constexpr int NF = F;
    // single precision switches near vertical patches to double, so its cost varies by source
    static constexpr bool MIXED = sizeof(T) < sizeof(double);

    // per unit strike, dip and tensile slip, out[comp * 3 + slip]
    CUOKADA_HD static void unit(const T * obs, const T * src, T nu, T * out) {
        okada::rect<F>(obs, src, nu, out);
    }

    CUOKADA_HD static void slip(const T * obs, const T * src, const T * slip, T nu, T * out) {
        T g[F * 3];
        okada::rect<F>(obs, src, nu, g);
        for (int c = 0; c < F; ++c)
            out[c] = g[c * 3] * slip[0] + g[c * 3 + 1] * slip[1] + g[c * 3 + 2] * slip[2];
    }
};

// triangles: the three vertices, 9 numbers per source; slip along the strike, dip and normal of
// the triangle frame of Nikkhoo and Walter (2015)
template <typename T, int F, bool HS>
struct Tri {
    static constexpr int SRC = 9;
    static constexpr int NF = F;
    static constexpr bool MIXED = false;

    CUOKADA_HD static void eval(const T * obs, const T * src, V3<T> b, T nu, T * out) {
        const V3<T> o = {obs[0], obs[1], obs[2]};
        const V3<T> t0 = {src[0], src[1], src[2]};
        const V3<T> t1 = {src[3], src[4], src[5]};
        const V3<T> t2 = {src[6], src[7], src[8]};
        if constexpr (F == DISP) {
            const V3<T> u = HS ? tde::disp_hs<T>(o, t0, t1, t2, b, nu) : tde::disp_fs<T>(o, t0, t1, t2, b, nu);
            out[0] = u.x;
            out[1] = u.y;
            out[2] = u.z;
        } else {
            const V6<T> e = HS ? tde::strain_hs<T>(o, t0, t1, t2, b, nu) : tde::strain_fs<T>(o, t0, t1, t2, b, nu);
            out[0] = e.x;
            out[1] = e.y;
            out[2] = e.z;
            out[3] = e.a;
            out[4] = e.b;
            out[5] = e.c;
        }
    }

    CUOKADA_HD static void unit(const T * obs, const T * src, T nu, T * out) {
        T v[F];
        for (int s = 0; s < 3; ++s) {
            // the triangle frame orders its slip as (normal, strike, dip)
            V3<T> b = {T(s == 2), T(s == 0), T(s == 1)};
            eval(obs, src, b, nu, v);
            for (int c = 0; c < F; ++c)
                out[c * 3 + s] = v[c];
        }
    }

    CUOKADA_HD static void slip(const T * obs, const T * src, const T * slip, T nu, T * out) {
        eval(obs, src, V3<T>{slip[2], slip[0], slip[1]}, nu, out);
    }
};

// matrix: out[nobs, F, nsrc, 3], one (station, source) pair
template <class E, typename T>
CUOKADA_HD inline void matrix_entry(int64_t i, int64_t j, const T * obs, const T * src, int64_t nsrc, T nu, T * out) {
    T g[E::NF * 3];
    E::unit(obs + 3 * i, src + E::SRC * j, nu, g);
    for (int c = 0; c < E::NF; ++c)
        for (int s = 0; s < 3; ++s)
            out[((i * E::NF + c) * nsrc + j) * 3 + s] = g[c * 3 + s];
}

// sum: out[nobs, F], all sources at one station, with a compensated sum
template <class E, typename T>
CUOKADA_HD inline void sum_entry(int64_t i, const T * obs, const T * src, const T * slip, int64_t nsrc, T nu, T * out) {
    T acc[E::NF], cmp[E::NF], v[E::NF];
    for (int c = 0; c < E::NF; ++c) {
        acc[c] = 0;
        cmp[c] = 0;
    }
    for (int64_t j = 0; j < nsrc; ++j) {
        E::slip(obs + 3 * i, src + E::SRC * j, slip + 3 * j, nu, v);
        for (int c = 0; c < E::NF; ++c) {
            const T y = v[c] - cmp[c];
            const T t = acc[c] + y;
            cmp[c] = (t - acc[c]) - y;
            acc[c] = t;
        }
    }
    for (int c = 0; c < E::NF; ++c)
        out[i * E::NF + c] = acc[c];
}

// pairs: out[n, F], station i with source i
template <class E, typename T>
CUOKADA_HD inline void pair_entry(int64_t i, const T * obs, const T * src, const T * slip, T nu, T * out) {
    E::slip(obs + 3 * i, src + E::SRC * i, slip + 3 * i, nu, out + E::NF * i);
}

} // namespace cuokada

// end of file
