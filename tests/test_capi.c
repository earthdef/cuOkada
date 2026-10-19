/* -*- c -*-
 *
 * the c interface: a rectangle against its two triangles, and the modes against each other
 *
 * Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <cuokada/cuokada.h>

#define NOBS 4
#define DEG 0.017453292519943295

static int failures = 0;

static void check(const char *what, double a, double b, double tol) {
    if (!(fabs(a - b) <= tol * (1 + fabs(b)))) {
        printf("FAIL %s: %.12g vs %.12g\n", what, a, b);
        failures++;
    }
}

int main(void) {
    /* a buried, dipping patch and a few stations, one of them at depth */
    double rect[7] = {1.0, -2.0, 5.0, 10.0, 6.0, 30.0, 40.0};
    double obs[NOBS * 3] = {3.0, 4.0, 0.0, -8.0, 2.0, 0.0, 12.0, -7.0, 0.0, 2.0, 1.0, -1.5};
    double slip[3] = {1.0, -0.5, 0.25};

    /* the corners of the patch, and its triangles */
    double st = rect[5] * DEG, dp = rect[6] * DEG;
    double s[3] = {sin(st), cos(st), 0.0};
    double d[3] = {cos(st) * cos(dp), -sin(st) * cos(dp), -sin(dp)};
    double c[3] = {rect[0], rect[1], -rect[2]};
    double corner[4][3];
    double sl[4] = {-1, 1, 1, -1}, sw[4] = {-1, -1, 1, 1};
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < 3; ++i)
            corner[k][i] = c[i] + sl[k] * rect[3] / 2 * s[i] + sw[k] * rect[4] / 2 * d[i];
    /* counterclockwise seen from above: the normal points up, so the slip signs are okada's */
    int order[6] = {0, 2, 1, 0, 3, 2};
    double tris[2 * 9];
    for (int t = 0; t < 6; ++t)
        for (int i = 0; i < 3; ++i)
            tris[t * 3 + i] = corner[order[t]][i];

    /* the rectangle matrix and the triangle matrix, summed over the two triangles */
    double gr[NOBS * 3 * 1 * 3], gt[NOBS * 3 * 2 * 3];
    cuokada_rect_disp_matrix_d(obs, NOBS, rect, 1, 0.25, gr);
    cuokada_tri_disp_matrix_d(obs, NOBS, tris, 2, 0.25, 1, gt);
    for (int i = 0; i < NOBS; ++i)
        for (int k = 0; k < 3; ++k)
            for (int q = 0; q < 3; ++q) {
                double r = gr[(i * 3 + k) * 3 + q];
                double t = gt[((i * 3 + k) * 2 + 0) * 3 + q] + gt[((i * 3 + k) * 2 + 1) * 3 + q];
                check("rect vs tri", r, t, 1e-9);
            }

    /* the strain too */
    double er[NOBS * 6 * 3], et[NOBS * 6 * 2 * 3];
    cuokada_rect_strain_matrix_d(obs, NOBS, rect, 1, 0.25, er);
    cuokada_tri_strain_matrix_d(obs, NOBS, tris, 2, 0.25, 1, et);
    for (int i = 0; i < NOBS; ++i)
        for (int k = 0; k < 6; ++k)
            for (int q = 0; q < 3; ++q) {
                double r = er[(i * 6 + k) * 3 + q];
                double t = et[((i * 6 + k) * 2 + 0) * 3 + q] + et[((i * 6 + k) * 2 + 1) * 3 + q];
                check("rect vs tri strain", r, t, 1e-8);
            }

    /* sum and pairs against the matrix */
    double us[NOBS * 3], up[NOBS * 3], rects[NOBS * 7], slips[NOBS * 3];
    for (int i = 0; i < NOBS; ++i) {
        for (int k = 0; k < 7; ++k) rects[i * 7 + k] = rect[k];
        for (int k = 0; k < 3; ++k) slips[i * 3 + k] = slip[k];
    }
    cuokada_rect_disp_sum_d(obs, NOBS, rect, slip, 1, 0.25, us);
    cuokada_rect_disp_pairs_d(obs, rects, slips, NOBS, 0.25, up);
    for (int i = 0; i < NOBS; ++i)
        for (int k = 0; k < 3; ++k) {
            double g = 0;
            for (int q = 0; q < 3; ++q) g += gr[(i * 3 + k) * 3 + q] * slip[q];
            check("sum vs matrix", us[i * 3 + k], g, 1e-12);
            check("pairs vs matrix", up[i * 3 + k], g, 1e-12);
        }

    /* the gpu, through the memory functions of the interface */
    int ngpu = 0;
    cuokada_cuda_device_count(&ngpu);
    if (cuokada_has_cuda() && ngpu > 0) {
        void *d_obs = 0, *d_rect = 0, *d_g = 0;
        double gg[NOBS * 3 * 3];
        int kind = -1, status = 0;
        status += cuokada_cuda_malloc(&d_obs, sizeof(obs));
        status += cuokada_cuda_malloc(&d_rect, sizeof(rect));
        status += cuokada_cuda_malloc(&d_g, sizeof(gg));
        status += cuokada_cuda_memcpy(d_obs, obs, sizeof(obs), 0);
        status += cuokada_cuda_memcpy(d_rect, rect, sizeof(rect), 0);
        status += cuokada_cuda_rect_disp_matrix_d(d_obs, NOBS, d_rect, 1, 0.25, d_g, 0);
        status += cuokada_cuda_memcpy(gg, d_g, sizeof(gg), 0);
        status += cuokada_cuda_synchronize(0);
        status += cuokada_cuda_memory_kind(d_g, &kind);
        if (status != 0 || kind != 2) {
            printf("FAIL gpu memory: status %d, kind %d\n", status, kind);
            failures++;
        }
        for (int i = 0; i < NOBS * 3 * 3; ++i)
            check("gpu vs cpu", gg[i], gr[i], 1e-10);
        cuokada_cuda_free(d_obs);
        cuokada_cuda_free(d_rect);
        cuokada_cuda_free(d_g);
    }

    /* a station above the surface */
    double above[3] = {0.0, 0.0, 1.0}, ua[9];
    cuokada_rect_disp_matrix_d(above, 1, rect, 1, 0.25, ua);
    if (!isnan(ua[0])) {
        printf("FAIL above the surface: %g\n", ua[0]);
        failures++;
    }

    printf("%s: %d failures, cuda %d\n", failures ? "FAIL" : "PASS", failures, cuokada_has_cuda());
    return failures != 0;
}

/* end of file */
