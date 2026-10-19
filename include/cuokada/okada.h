// -*- c++ -*-
//
// rectangular dislocations in a half space, after Okada (1992), BSSA 82(2), 1018-1040
//
// Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

#pragma once

#include "common.h"

namespace cuokada::okada {

// the 12 outputs of dc3d: u, du/dx, du/dy, du/dz, each as (ux, uy, uz), in okada's frame
constexpr int NOUT = 12;

// the constants of a source: elastic and dip (dccon0)
template <typename T>
struct Source {
    T alp1, alp2, alp3, alp4, alp5;
    T sd, cd, sdsd, cdcd, sdcd, s2d, c2d;

    CUOKADA_HD Source(T alpha, T dip) {
        const T pi2 = T(6.283185307179586);
        alp1 = (1 - alpha) / 2;
        alp2 = alpha / 2;
        alp3 = (1 - alpha) / alpha;
        alp4 = 1 - alpha;
        alp5 = alpha;
        sd = sin(dip * pi2 / 360);
        cd = cos(dip * pi2 / 360);
        // a vertical fault
        if (fabs(cd) < T(1e-6)) {
            cd = 0;
            sd = sd > 0 ? T(1) : T(-1);
        }
        sdsd = sd * sd;
        cdcd = cd * cd;
        sdcd = sd * cd;
        s2d = 2 * sdcd;
        c2d = cdcd - sdsd;
    }
};

// below this (r + xi) / r or (r + et) / r, single precision corner terms cancel too much
constexpr double ILL_RATIO = 1e-4;
// the status of a single precision evaluation that should be redone in double
constexpr int ILL = 3;

// the station-dependent constants at one corner (dccon2)
template <typename T>
struct Corner {
    T xi, et, q;
    T xi2, et2, q2, r, r2, r3, r5, y, d, tt;
    T alx, ale, x11, y11, x32, y32;
    T ey, ez, fy, fz, gy, gz, hy, hz;
    // in single precision, too close to an edge extension for the corner terms to cancel
    bool ill = false;

    CUOKADA_HD Corner(T xi_, T et_, T q_, const Source<T> & s, int kxi, int ket) {
        const T eps = T(1e-6);
        xi = fabs(xi_) < eps ? T(0) : xi_;
        et = fabs(et_) < eps ? T(0) : et_;
        q = fabs(q_) < eps ? T(0) : q_;
        xi2 = xi * xi;
        et2 = et * et;
        q2 = q * q;
        r2 = xi2 + et2 + q2;
        r = sqrt(r2);
        if (r == 0) {
            return;
        }
        r3 = r * r2;
        r5 = r3 * r2;
        y = et * s.cd + q * s.sd;
        d = et * s.sd - q * s.cd;
        tt = q == 0 ? T(0) : atan(xi * et / (q * r));
        if (kxi == 1) {
            alx = -log(r - xi);
            x11 = 0;
            x32 = 0;
        } else {
            // r + xi without the cancellation on the strike extension of an edge, where xi < 0
            T rxi = xi < 0 ? (et2 + q2) / (r - xi) : r + xi;
            alx = log(rxi);
            x11 = 1 / (r * rxi);
            x32 = (r + rxi) * x11 * x11 / r;
            ill = ill || rxi < T(ILL_RATIO) * r;
        }
        if (ket == 1) {
            ale = -log(r - et);
            y11 = 0;
            y32 = 0;
        } else {
            // r + et without the cancellation on the dip extension of an edge, where et < 0
            T ret = et < 0 ? (xi2 + q2) / (r - et) : r + et;
            ale = log(ret);
            y11 = 1 / (r * ret);
            y32 = (r + ret) * y11 * y11 / r;
            ill = ill || ret < T(ILL_RATIO) * r;
        }
        ey = s.sd / r - y * q / r3;
        ez = s.cd / r + d * q / r3;
        fy = d / r3 + xi2 * y32 * s.sd;
        fz = y / r3 + xi2 * y32 * s.cd;
        gy = 2 * x11 * s.sd - y * q * x32;
        gz = 2 * x11 * s.cd + d * q * x32;
        hy = d * q * x32 + xi * q * y32 * s.sd;
        hz = y * q * x32 + xi * q * y32 * s.cd;
    }
};

// the part A of the displacement and its derivatives, per unit slip (ua)
template <bool Grad, typename T>
CUOKADA_HD inline void ua(const Corner<T> & c, const Source<T> & s, T du[3][NOUT]) {
    const T xi = c.xi, et = c.et, q = c.q, r = c.r, r3 = c.r3, tt = c.tt;
    const T alp1 = s.alp1, alp2 = s.alp2, sd = s.sd, cd = s.cd;
    const T xy = xi * c.y11, qx = q * c.x11, qy = q * c.y11;
    // strike slip
    du[0][0] = tt / 2 + alp2 * xi * qy;
    du[0][1] = alp2 * q / r;
    du[0][2] = alp1 * c.ale - alp2 * q * qy;
    // dip slip
    du[1][0] = alp2 * q / r;
    du[1][1] = tt / 2 + alp2 * et * qx;
    du[1][2] = alp1 * c.alx - alp2 * q * qx;
    // tensile
    du[2][0] = -alp1 * c.ale - alp2 * q * qy;
    du[2][1] = -alp1 * c.alx - alp2 * q * qx;
    du[2][2] = tt / 2 - alp2 * (et * qx + xi * qy);
    if constexpr (Grad) {
        const T xi2 = c.xi2, q2 = c.q2, y32 = c.y32, x11 = c.x11, d = c.d, y = c.y;
        du[0][3] = -alp1 * qy - alp2 * xi2 * q * y32;
        du[0][4] = -alp2 * xi * q / r3;
        du[0][5] = alp1 * xy + alp2 * xi * q2 * y32;
        du[0][6] = alp1 * xy * sd + alp2 * xi * c.fy + d / 2 * x11;
        du[0][7] = alp2 * c.ey;
        du[0][8] = alp1 * (cd / r + qy * sd) - alp2 * q * c.fy;
        du[0][9] = alp1 * xy * cd + alp2 * xi * c.fz + y / 2 * x11;
        du[0][10] = alp2 * c.ez;
        du[0][11] = -alp1 * (sd / r - qy * cd) - alp2 * q * c.fz;

        du[1][3] = -alp2 * xi * q / r3;
        du[1][4] = -qy / 2 - alp2 * et * q / r3;
        du[1][5] = alp1 / r + alp2 * q2 / r3;
        du[1][6] = alp2 * c.ey;
        du[1][7] = alp1 * d * x11 + xy / 2 * sd + alp2 * et * c.gy;
        du[1][8] = alp1 * y * x11 - alp2 * q * c.gy;
        du[1][9] = alp2 * c.ez;
        du[1][10] = alp1 * y * x11 + xy / 2 * cd + alp2 * et * c.gz;
        du[1][11] = -alp1 * d * x11 - alp2 * q * c.gz;

        du[2][3] = -alp1 * xy + alp2 * xi * q2 * y32;
        du[2][4] = -alp1 / r + alp2 * q2 / r3;
        du[2][5] = -alp1 * qy - alp2 * q * q2 * y32;
        du[2][6] = -alp1 * (cd / r + qy * sd) - alp2 * q * c.fy;
        du[2][7] = -alp1 * y * x11 - alp2 * q * c.gy;
        du[2][8] = alp1 * (d * x11 + xy * sd) + alp2 * q * c.hy;
        du[2][9] = alp1 * (sd / r - qy * cd) - alp2 * q * c.fz;
        du[2][10] = alp1 * d * x11 - alp2 * q * c.gz;
        du[2][11] = alp1 * (y * x11 + xy * cd) + alp2 * q * c.hz;
    }
}

// the part B of the displacement and its derivatives, per unit slip (ub)
template <bool Grad, typename T>
CUOKADA_HD inline void ub(const Corner<T> & c, const Source<T> & s, T du[3][NOUT]) {
    const T xi = c.xi, et = c.et, q = c.q, r = c.r, r3 = c.r3, tt = c.tt;
    const T xi2 = c.xi2, q2 = c.q2, y = c.y, d = c.d;
    const T sd = s.sd, cd = s.cd, sdsd = s.sdsd, cdcd = s.cdcd, sdcd = s.sdcd, alp3 = s.alp3;
    // r + d without the cancellation where d < 0, as r^2 - d^2 = xi^2 + y^2
    const T rd = d < 0 ? (xi2 + y * y) / (r - d) : r + d;
    const T d11 = 1 / (r * rd);
    const T aj2 = xi * y / rd * d11;
    const T aj5 = -(d + y * y / rd) * d11;
    T ai3, ai4, ak1, ak3, aj3, aj6;
    if (cd != 0) {
        if (xi == 0) {
            ai4 = 0;
        } else {
            T x = sqrt(xi2 + q2);
            ai4 = 1 / cdcd * (xi / rd * sdcd + 2 * atan((et * (x + q * cd) + x * (r + x) * sd) / (xi * (r + x) * cd)));
        }
        ai3 = (y * cd / rd - c.ale + sd * log(rd)) / cdcd;
        ak1 = xi * (d11 - c.y11 * sd) / cd;
        ak3 = (q * c.y11 - y * d11) / cd;
        aj3 = (ak1 - aj2 * sd) / cd;
        aj6 = (ak3 - aj5 * sd) / cd;
    } else {
        T rd2 = rd * rd;
        ai3 = (et / rd + y * q / rd2 - c.ale) / 2;
        ai4 = xi * y / rd2 / 2;
        ak1 = xi * q / rd * d11;
        ak3 = sd / rd * (xi2 * d11 - 1);
        aj3 = -xi / rd2 * (q2 * d11 - T(0.5));
        aj6 = -y / rd2 * (xi2 * d11 - T(0.5));
    }
    const T xy = xi * c.y11;
    const T ai1 = -xi / rd * cd - ai4 * sd;
    const T ai2 = log(rd) + ai3 * sd;
    const T qx = q * c.x11, qy = q * c.y11;
    // strike slip
    du[0][0] = -xi * qy - tt - alp3 * ai1 * sd;
    du[0][1] = -q / r + alp3 * y / rd * sd;
    du[0][2] = q * qy - alp3 * ai2 * sd;
    // dip slip
    du[1][0] = -q / r + alp3 * ai3 * sdcd;
    du[1][1] = -et * qx - tt - alp3 * xi / rd * sdcd;
    du[1][2] = q * qx + alp3 * ai4 * sdcd;
    // tensile
    du[2][0] = q * qy - alp3 * ai3 * sdsd;
    du[2][1] = q * qx + alp3 * xi / rd * sdsd;
    du[2][2] = et * qx + xi * qy - tt - alp3 * ai4 * sdsd;
    if constexpr (Grad) {
        const T y32 = c.y32, x11 = c.x11;
        const T ak2 = 1 / r + ak3 * sd;
        const T ak4 = xy * cd - ak1 * sd;
        const T aj1 = aj5 * cd - aj6 * sd;
        const T aj4 = -xy - aj2 * cd + aj3 * sd;

        du[0][3] = xi2 * q * y32 - alp3 * aj1 * sd;
        du[0][4] = xi * q / r3 - alp3 * aj2 * sd;
        du[0][5] = -xi * q2 * y32 - alp3 * aj3 * sd;
        du[0][6] = -xi * c.fy - d * x11 + alp3 * (xy + aj4) * sd;
        du[0][7] = -c.ey + alp3 * (1 / r + aj5) * sd;
        du[0][8] = q * c.fy - alp3 * (qy - aj6) * sd;
        du[0][9] = -xi * c.fz - y * x11 + alp3 * ak1 * sd;
        du[0][10] = -c.ez + alp3 * y * d11 * sd;
        du[0][11] = q * c.fz + alp3 * ak2 * sd;

        du[1][3] = xi * q / r3 + alp3 * aj4 * sdcd;
        du[1][4] = et * q / r3 + qy + alp3 * aj5 * sdcd;
        du[1][5] = -q2 / r3 + alp3 * aj6 * sdcd;
        du[1][6] = -c.ey + alp3 * aj1 * sdcd;
        du[1][7] = -et * c.gy - xy * sd + alp3 * aj2 * sdcd;
        du[1][8] = q * c.gy + alp3 * aj3 * sdcd;
        du[1][9] = -c.ez - alp3 * ak3 * sdcd;
        du[1][10] = -et * c.gz - xy * cd - alp3 * xi * d11 * sdcd;
        du[1][11] = q * c.gz - alp3 * ak4 * sdcd;

        du[2][3] = -xi * q2 * y32 - alp3 * aj4 * sdsd;
        du[2][4] = -q2 / r3 - alp3 * aj5 * sdsd;
        du[2][5] = q * q2 * y32 - alp3 * aj6 * sdsd;
        du[2][6] = q * c.fy - alp3 * aj1 * sdsd;
        du[2][7] = q * c.gy - alp3 * aj2 * sdsd;
        du[2][8] = -q * c.hy - alp3 * aj3 * sdsd;
        du[2][9] = q * c.fz + alp3 * ak3 * sdsd;
        du[2][10] = q * c.gz + alp3 * xi * d11 * sdsd;
        du[2][11] = -q * c.hz + alp3 * ak4 * sdsd;
    }
}

// the part C of the displacement and its derivatives, per unit slip (uc)
template <bool Grad, typename T>
CUOKADA_HD inline void uc(const Corner<T> & c, const Source<T> & s, T z, T du[3][NOUT]) {
    const T xi = c.xi, et = c.et, q = c.q, r = c.r, r2 = c.r2, r3 = c.r3, r5 = c.r5;
    const T xi2 = c.xi2, et2 = c.et2, q2 = c.q2, y = c.y, d = c.d;
    const T x11 = c.x11, y11 = c.y11, x32 = c.x32, y32 = c.y32;
    const T sd = s.sd, cd = s.cd, sdsd = s.sdsd, cdcd = s.cdcd, sdcd = s.sdcd;
    const T alp4 = s.alp4, alp5 = s.alp5;
    const T cc = d + z;
    const T x53 = (8 * r2 + 9 * r * xi + 3 * xi2) * x11 * x11 * x11 / r2;
    const T y53 = (8 * r2 + 9 * r * et + 3 * et2) * y11 * y11 * y11 / r2;
    const T h = q * cd - z;
    const T z32 = sd / r3 - h * y32;
    const T z53 = 3 * sd / r5 - h * y53;
    const T y0 = y11 - xi2 * y32;
    const T z0 = z32 - xi2 * z53;
    const T ppy = cd / r3 + q * y32 * sd;
    const T ppz = sd / r3 - q * y32 * cd;
    const T qq = z * y32 + z32 + z0;
    const T qqy = 3 * cc * d / r5 - qq * sd;
    const T qqz = 3 * cc * y / r5 - qq * cd + q * y32;
    const T xy = xi * y11;
    const T qy = q * y11;
    const T qr = 3 * q / r5;
    const T cdr = (cc + d) / r3;
    const T yy0 = y / r3 - y0 * cd;
    // strike slip
    du[0][0] = alp4 * xy * cd - alp5 * xi * q * z32;
    du[0][1] = alp4 * (cd / r + 2 * qy * sd) - alp5 * cc * q / r3;
    du[0][2] = alp4 * qy * cd - alp5 * (cc * et / r3 - z * y11 + xi2 * z32);
    // dip slip
    du[1][0] = alp4 * cd / r - qy * sd - alp5 * cc * q / r3;
    du[1][1] = alp4 * y * x11 - alp5 * cc * et * q * x32;
    du[1][2] = -d * x11 - xy * sd - alp5 * cc * (x11 - q2 * x32);
    // tensile
    du[2][0] = -alp4 * (sd / r + qy * cd) - alp5 * (z * y11 - q2 * z32);
    du[2][1] = alp4 * 2 * xy * sd + d * x11 - alp5 * cc * (x11 - q2 * x32);
    du[2][2] = alp4 * (y * x11 + xy * cd) + alp5 * q * (cc * et * x32 + xi * z32);
    if constexpr (Grad) {
        du[0][3] = alp4 * y0 * cd - alp5 * q * z0;
        du[0][4] = -alp4 * xi * (cd / r3 + 2 * q * y32 * sd) + alp5 * cc * xi * qr;
        du[0][5] = -alp4 * xi * q * y32 * cd + alp5 * xi * (3 * cc * et / r5 - qq);
        du[0][6] = -alp4 * xi * ppy * cd - alp5 * xi * qqy;
        du[0][7] = alp4 * 2 * (d / r3 - y0 * sd) * sd - y / r3 * cd - alp5 * (cdr * sd - et / r3 - cc * y * qr);
        du[0][8] = -alp4 * q / r3 + yy0 * sd + alp5 * (cdr * cd + cc * d * qr - (y0 * cd + q * z0) * sd);
        du[0][9] = alp4 * xi * ppz * cd - alp5 * xi * qqz;
        du[0][10] = alp4 * 2 * (y / r3 - y0 * cd) * sd + d / r3 * cd - alp5 * (cdr * cd + cc * d * qr);
        du[0][11] = yy0 * cd - alp5 * (cdr * sd - cc * y * qr - y0 * sdsd + q * z0 * cd);

        du[1][3] = -alp4 * xi / r3 * cd + alp5 * cc * xi * qr + xi * q * y32 * sd;
        du[1][4] = -alp4 * y / r3 + alp5 * cc * et * qr;
        du[1][5] = d / r3 - y0 * sd + alp5 * cc / r3 * (1 - 3 * q2 / r2);
        du[1][6] = -alp4 * et / r3 + y0 * sdsd - alp5 * (cdr * sd - cc * y * qr);
        du[1][7] = alp4 * (x11 - y * y * x32) - alp5 * cc * ((d + 2 * q * cd) * x32 - y * et * q * x53);
        du[1][8] = xi * ppy * sd + y * d * x32 + alp5 * cc * ((y + 2 * q * sd) * x32 - y * q2 * x53);
        du[1][9] = -q / r3 + y0 * sdcd - alp5 * (cdr * cd + cc * d * qr);
        du[1][10] = alp4 * y * d * x32 - alp5 * cc * ((y - 2 * q * sd) * x32 + d * et * q * x53);
        du[1][11] = -xi * ppz * sd + x11 - d * d * x32 - alp5 * cc * ((d - 2 * q * cd) * x32 - d * q2 * x53);

        du[2][3] = alp4 * xi / r3 * sd + xi * q * y32 * cd + alp5 * xi * (3 * cc * et / r5 - 2 * z32 - z0);
        du[2][4] = alp4 * 2 * y0 * sd - d / r3 + alp5 * cc / r3 * (1 - 3 * q2 / r2);
        du[2][5] = -alp4 * yy0 - alp5 * (cc * et * qr - q * z0);
        du[2][6] = alp4 * (q / r3 + y0 * sdcd) + alp5 * (z / r3 * cd + cc * d * qr - q * z0 * sd);
        du[2][7] = -alp4 * 2 * xi * ppy * sd - y * d * x32 + alp5 * cc * ((y + 2 * q * sd) * x32 - y * q2 * x53);
        du[2][8] = -alp4 * (xi * ppy * cd - x11 + y * y * x32)
                   + alp5 * (cc * ((d + 2 * q * cd) * x32 - y * et * q * x53) + xi * qqy);
        du[2][9] = -et / r3 + y0 * cdcd - alp5 * (z / r3 * sd - cc * y * qr - y0 * sdsd + q * z0 * cd);
        du[2][10] = alp4 * 2 * xi * ppz * sd - x11 + d * d * x32 - alp5 * cc * ((d - 2 * q * cd) * x32 - d * q2 * x53);
        du[2][11] = alp4 * (xi * ppz * cd + y * d * x32)
                    + alp5 * (cc * ((y - 2 * q * sd) * x32 + d * et * q * x53) + xi * qqz);
    }
}

// the flags for the log singularities on the extensions of the fault edges
template <typename T>
CUOKADA_HD inline void singular_flags(const T xi[2], const T et[2], T q, int kxi[2], int ket[2]) {
    const T eps = T(1e-6);
    const T r12 = sqrt(xi[0] * xi[0] + et[1] * et[1] + q * q);
    const T r21 = sqrt(xi[1] * xi[1] + et[0] * et[0] + q * q);
    const T r22 = sqrt(xi[1] * xi[1] + et[1] * et[1] + q * q);
    kxi[0] = (xi[0] < 0 && r21 + xi[1] < eps) ? 1 : 0;
    kxi[1] = (xi[0] < 0 && r22 + xi[1] < eps) ? 1 : 0;
    ket[0] = (et[0] < 0 && r12 + et[1] < eps) ? 1 : 0;
    ket[1] = (et[0] < 0 && r22 + et[1] < eps) ? 1 : 0;
}

// on a fault edge
template <typename T>
CUOKADA_HD inline bool on_edge(const T xi[2], const T et[2], T q) {
    return q == 0 && ((xi[0] * xi[1] <= 0 && et[0] * et[1] == 0) || (et[0] * et[1] <= 0 && xi[0] * xi[1] == 0));
}

// the displacement (and its derivatives if {Grad}) at (x, y, z), z <= 0, of a rectangle along
// strike from al1 to al2 and along dip from aw1 to aw2, at {depth}, per unit strike, dip and
// tensile slip; returns 0, 1 for z > 0, 2 on a fault edge (with zeros), as dc3d
template <bool Grad, typename T>
CUOKADA_HD inline int dc3d(T alpha, T x, T y, T z, T depth, T dip, T al1, T al2, T aw1, T aw2, T u[3][NOUT]) {
    constexpr int nout = Grad ? NOUT : 3;
    for (int s = 0; s < 3; ++s)
        for (int i = 0; i < NOUT; ++i)
            u[s][i] = 0;
    if (z > 0) {
        return 1;
    }
    const T eps = T(1e-6);
    const T pi2 = T(6.283185307179586);
    const Source<T> src(alpha, dip);
    const T sd = src.sd, cd = src.cd;
    T xi[2] = {x - al1, x - al2};
    for (int j = 0; j < 2; ++j)
        if (fabs(xi[j]) < eps) xi[j] = 0;

    T du[3][NOUT], dua[3][NOUT], dub[3][NOUT], duc[3][NOUT];
    int kxi[2], ket[2];
    bool ill = false;

    // the real source
    {
        const T d = depth + z;
        const T p = y * cd + d * sd;
        T q = y * sd - d * cd;
        T et[2] = {p - aw1, p - aw2};
        if (fabs(q) < eps) q = 0;
        for (int k = 0; k < 2; ++k)
            if (fabs(et[k]) < eps) et[k] = 0;
        if (on_edge(xi, et, q)) {
            return 2;
        }
        singular_flags(xi, et, q, kxi, ket);
        for (int k = 0; k < 2; ++k) {
            for (int j = 0; j < 2; ++j) {
                const Corner<T> c(xi[j], et[k], q, src, kxi[k], ket[j]);
                ill = ill || c.ill;
                ua<Grad>(c, src, dua);
                for (int s = 0; s < 3; ++s) {
                    for (int i = 0; i < nout; i += 3) {
                        du[s][i] = -dua[s][i];
                        du[s][i + 1] = -dua[s][i + 1] * cd + dua[s][i + 2] * sd;
                        du[s][i + 2] = -dua[s][i + 1] * sd - dua[s][i + 2] * cd;
                    }
                    if constexpr (Grad) {
                        du[s][9] = -du[s][9];
                        du[s][10] = -du[s][10];
                        du[s][11] = -du[s][11];
                    }
                    const T sign = (j + k == 1) ? T(-1) : T(1);
                    for (int i = 0; i < nout; ++i)
                        u[s][i] += sign * du[s][i];
                }
            }
        }
    }

    // the image source
    {
        const T d = depth - z;
        const T p = y * cd + d * sd;
        T q = y * sd - d * cd;
        T et[2] = {p - aw1, p - aw2};
        if (fabs(q) < eps) q = 0;
        for (int k = 0; k < 2; ++k)
            if (fabs(et[k]) < eps) et[k] = 0;
        if (on_edge(xi, et, q)) {
            for (int s = 0; s < 3; ++s)
                for (int i = 0; i < NOUT; ++i)
                    u[s][i] = 0;
            return 2;
        }
        singular_flags(xi, et, q, kxi, ket);
        // part C only matters below the surface, or for the derivatives
        const bool needc = Grad || z != 0;
        for (int k = 0; k < 2; ++k) {
            for (int j = 0; j < 2; ++j) {
                const Corner<T> c(xi[j], et[k], q, src, kxi[k], ket[j]);
                ill = ill || c.ill;
                ua<Grad>(c, src, dua);
                ub<Grad>(c, src, dub);
                if (needc) {
                    uc<Grad>(c, src, z, duc);
                } else {
                    for (int s = 0; s < 3; ++s)
                        for (int i = 0; i < 3; ++i)
                            duc[s][i] = 0;
                }
                for (int s = 0; s < 3; ++s) {
                    for (int i = 0; i < nout; i += 3) {
                        du[s][i] = dua[s][i] + dub[s][i] + z * duc[s][i];
                        du[s][i + 1] = (dua[s][i + 1] + dub[s][i + 1] + z * duc[s][i + 1]) * cd
                                       - (dua[s][i + 2] + dub[s][i + 2] + z * duc[s][i + 2]) * sd;
                        du[s][i + 2] = (dua[s][i + 1] + dub[s][i + 1] - z * duc[s][i + 1]) * sd
                                       + (dua[s][i + 2] + dub[s][i + 2] - z * duc[s][i + 2]) * cd;
                    }
                    if constexpr (Grad) {
                        du[s][9] += duc[s][0];
                        du[s][10] += duc[s][1] * cd - duc[s][2] * sd;
                        du[s][11] += -duc[s][1] * sd - duc[s][2] * cd;
                    }
                    const T sign = (j + k == 1) ? T(-1) : T(1);
                    for (int i = 0; i < nout; ++i)
                        u[s][i] += sign * du[s][i];
                }
            }
        }
    }

    for (int s = 0; s < 3; ++s)
        for (int i = 0; i < nout; ++i)
            u[s][i] /= pi2;
    return (sizeof(T) < sizeof(double) && ill) ? ILL : 0;
}

// a patch: the center (x, y) and its depth, length along strike, width along dip, strike and
// dip in degrees; x east, y north, z up, the strike clockwise from north, dip to its right
constexpr int RECT = 7;

// the displacement (DISP) or strain (STRAIN) at {obs} per unit strike, dip and tensile slip,
// out[comp * 3 + slip]; strain as (xx, yy, zz, xy, xz, yz), in the precision of T
template <int F, typename T>
CUOKADA_HD inline int rect_in(const T * obs, const T * patch, T nu, T * out) {
    const T deg = T(0.017453292519943295);
    const T alpha = 1 / (2 * (1 - nu));
    const T cs = cos(patch[5] * deg), ss = sin(patch[5] * deg);
    // the frame of the patch: x along strike, y to the left of it
    const T ex[2] = {ss, cs};
    const T ey[2] = {-cs, ss};
    const T dx = obs[0] - patch[0], dy = obs[1] - patch[1];
    const T x = dx * ex[0] + dy * ex[1];
    const T y = dx * ey[0] + dy * ey[1];
    const T hl = patch[3] / 2, hw = patch[4] / 2;
    T u[3][NOUT];
    int status;
    if constexpr (F == DISP) {
        status = dc3d<false>(alpha, x, y, obs[2], patch[2], patch[6], -hl, hl, -hw, hw, u);
        for (int s = 0; s < 3; ++s) {
            out[0 * 3 + s] = u[s][0] * ex[0] + u[s][1] * ey[0];
            out[1 * 3 + s] = u[s][0] * ex[1] + u[s][1] * ey[1];
            out[2 * 3 + s] = u[s][2];
        }
    } else {
        status = dc3d<true>(alpha, x, y, obs[2], patch[2], patch[6], -hl, hl, -hw, hw, u);
        // the rotation from the patch frame: columns are its axes
        const T R[3][3] = {{ex[0], ey[0], 0}, {ex[1], ey[1], 0}, {0, 0, 1}};
        for (int s = 0; s < 3; ++s) {
            // g[i][j] = du_i/dx_j in the patch frame
            T g[3][3];
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                    g[i][j] = u[s][3 + 3 * j + i];
            // R g R^T
            T rg[3][3], G[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    rg[i][j] = R[i][0] * g[0][j] + R[i][1] * g[1][j] + R[i][2] * g[2][j];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    G[i][j] = rg[i][0] * R[j][0] + rg[i][1] * R[j][1] + rg[i][2] * R[j][2];
            out[0 * 3 + s] = G[0][0];
            out[1 * 3 + s] = G[1][1];
            out[2 * 3 + s] = G[2][2];
            out[3 * 3 + s] = (G[0][1] + G[1][0]) / 2;
            out[4 * 3 + s] = (G[0][2] + G[2][0]) / 2;
            out[5 * 3 + s] = (G[1][2] + G[2][1]) / 2;
        }
    }
    if (status == 1) {
        for (int i = 0; i < F * 3; ++i)
            out[i] = nan<T>();
    }
    return status;
}

// below this cos(dip), but above the vertical cutoff of dc3d, single precision loses digits to
// the 1/cos(dip)^2 of part B, so those patches run in double
constexpr double STEEP = 0.1;

// as rect_in; in single precision, near vertical patches and stations near the extension of an
// edge run in double
template <int F, typename T>
CUOKADA_HD inline int rect(const T * obs, const T * patch, T nu, T * out) {
    if constexpr (sizeof(T) < sizeof(double)) {
        const T cd = fabs(cos(patch[6] * T(0.017453292519943295)));
        const bool steep = cd < T(STEEP) && cd >= T(1e-6);
        if (!steep) {
            const int status = rect_in<F, T>(obs, patch, nu, out);
            if (status != ILL) {
                return status;
            }
        }
        double o[3], p[RECT], r[F * 3];
        for (int i = 0; i < 3; ++i)
            o[i] = obs[i];
        for (int i = 0; i < RECT; ++i)
            p[i] = patch[i];
        const int status = rect_in<F, double>(o, p, double(nu), r);
        for (int i = 0; i < F * 3; ++i)
            out[i] = T(r[i]);
        return status;
    } else {
        return rect_in<F, T>(obs, patch, nu, out);
    }
}

} // namespace cuokada::okada

// end of file
