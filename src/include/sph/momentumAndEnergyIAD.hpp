#pragma once

#include <vector>

#include <cmath>
#include "math.hpp"
#include "kernels.hpp"

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
void computeIAD(const std::vector<int> &l, Dataset &d)
{
    const int64_t n = l.size();
    const int64_t ngmax = d.ngmax;
    const int *clist = l.data();
    const int *neighbors = d.neighbors.data();
    const int *neighborsCount = d.neighborsCount.data();

    const T *h = d.h.data();
    const T *m = d.m.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();
    const T *ro = d.ro.data();

    T *c11 = d.c11.data();
    T *c12 = d.c12.data();
    T *c13 = d.c13.data();
    T *c22 = d.c22.data();
    T *c23 = d.c23.data();
    T *c33 = d.c33.data();

    const BBox<T> bbox = d.bbox;

    const T sincIndex = d.sincIndex;
    const T K = d.K;

    // std::vector<T> checkImem(n, 0), checkDeltaX(n, 0), checkDeltaY(n, 0), checkDeltaZ(n, 0);

#pragma omp parallel for schedule(guided)
    for (int pi = 0; pi < n; ++pi)
    {
        const int i = clist[pi];
        const int nn = neighborsCount[pi];

        T tau11 = 0.0, tau12 = 0.0, tau13 = 0.0, tau22 = 0.0, tau23 = 0.0, tau33 = 0.0;

        for (int pj = 0; pj < nn; ++pj)
        {
            const int j = neighbors[pi * ngmax + pj];

            // later can be stored into an array per particle
            const T dist = distancePBC(bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]); // store the distance from each neighbor
            // calculate the v as ratio between the distance and the smoothing length
            const T vloc = dist / h[i];
            const T W = wharmonic(vloc, h[i], sincIndex, K);

            T r_ijx = (x[i] - x[j]);
            T r_ijy = (y[i] - y[j]);
            T r_ijz = (z[i] - z[j]);

            applyPBC(bbox, 2.0 * h[i], r_ijx, r_ijy, r_ijz); // with this line energy grows even faster

            tau11 += r_ijx * r_ijx * m[j] / ro[j] * W;
            tau12 += r_ijx * r_ijy * m[j] / ro[j] * W;
            tau13 += r_ijx * r_ijz * m[j] / ro[j] * W;
            tau22 += r_ijy * r_ijy * m[j] / ro[j] * W;
            tau23 += r_ijy * r_ijz * m[j] / ro[j] * W;
            tau33 += r_ijz * r_ijz * m[j] / ro[j] * W;

/*            checkImem[i] += m[j] / ro[j] * W;
            checkDeltaX[i] += m[j] / ro[j] * (r_ijx) * W;
            checkDeltaY[i] += m[j] / ro[j] * (r_ijy) * W;
            checkDeltaZ[i] += m[j] / ro[j] * (r_ijz) * W;*/
        }

        const T det =
            tau11 * tau22 * tau33 + 2.0 * tau12 * tau23 * tau13 - tau11 * tau23 * tau23 - tau22 * tau13 * tau13 - tau33 * tau12 * tau12;

        c11[i] = (tau22 * tau33 - tau23 * tau23) / det;
        c12[i] = (tau13 * tau23 - tau33 * tau12) / det;
        c13[i] = (tau12 * tau23 - tau22 * tau13) / det;
        c22[i] = (tau11 * tau33 - tau13 * tau13) / det;
        c23[i] = (tau13 * tau12 - tau11 * tau23) / det;
        c33[i] = (tau11 * tau22 - tau12 * tau12) / det;
    }
}

template <typename T, class Dataset>
void computeMomentumAndEnergyIAD(const std::vector<int> &l, Dataset &d)
{
    const T gradh_i = 1.0;
    const T gradh_j = 1.0;

    const int64_t n = l.size();
    const int64_t ngmax = d.ngmax;
    const int *clist = l.data();
    const int *neighbors = d.neighbors.data();
    const int *neighborsCount = d.neighborsCount.data();

    const T *h = d.h.data();
    const T *m = d.m.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();
    const T *vx = d.vx.data();
    const T *vy = d.vy.data();
    const T *vz = d.vz.data();
    const T *ro = d.ro.data();
    const T *c = d.c.data();
    const T *p = d.p.data();

    const T *c11 = d.c11.data();
    const T *c12 = d.c12.data();
    const T *c13 = d.c13.data();
    const T *c22 = d.c22.data();
    const T *c23 = d.c23.data();
    const T *c33 = d.c33.data();

    T *du = d.du.data();
    T *grad_P_x = d.grad_P_x.data();
    T *grad_P_y = d.grad_P_y.data();
    T *grad_P_z = d.grad_P_z.data();

    const BBox<T> bbox = d.bbox;

    const T sincIndex = d.sincIndex;
    const T K = d.K;

#pragma omp parallel for schedule(guided)
    for (int pi = 0; pi < n; ++pi)
    {
        const int i = clist[pi];
        const int nn = neighborsCount[pi];

        T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0, energyAV = 0.0;
        for (int pj = 0; pj < nn; ++pj)
        {
            const int j = neighbors[pi * ngmax + pj];

            T r_ijx = (x[i] - x[j]);
            T r_ijy = (y[i] - y[j]);
            T r_ijz = (z[i] - z[j]);

            T r_jix = (x[j] - x[i]);
            T r_jiy = (y[j] - y[i]);
            T r_jiz = (z[j] - z[i]);

            applyPBC(bbox, 2.0 * h[i], r_ijx, r_ijy, r_ijz);
            applyPBC(bbox, 2.0 * h[i], r_jix, r_jiy, r_jiz);

            const T dist = std::sqrt(r_ijx * r_ijx + r_ijy * r_ijy + r_ijz * r_ijz);
            // const T dist = distancePBC(bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]); // store the distance from each neighbor

            const T v_ijx = (vx[i] - vx[j]);
            const T v_ijy = (vy[i] - vy[j]);
            const T v_ijz = (vz[i] - vz[j]);

            const T v1 = dist / h[i];
            const T v2 = dist / h[j];

            const T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

            const T W1 = wharmonic(v1, h[i], sincIndex, K);
            const T W2 = wharmonic(v2, h[j], sincIndex, K);

            const T kern11_i = c11[i] * r_jix;
            const T kern12_i = c12[i] * r_jiy;
            const T kern13_i = c13[i] * r_jiz;
            const T kern21_i = c12[i] * r_jix;
            const T kern22_i = c22[i] * r_jiy;
            const T kern23_i = c23[i] * r_jiz;
            const T kern31_i = c13[i] * r_jix;
            const T kern32_i = c23[i] * r_jiy;
            const T kern33_i = c33[i] * r_jiz;

            const T kern11_j = c11[j] * r_jix;
            const T kern12_j = c12[j] * r_jiy;
            const T kern13_j = c13[j] * r_jiz;
            const T kern21_j = c12[j] * r_jix;
            const T kern22_j = c22[j] * r_jiy;
            const T kern23_j = c23[j] * r_jiz;
            const T kern31_j = c13[j] * r_jix;
            const T kern32_j = c23[j] * r_jiy;
            const T kern33_j = c33[j] * r_jiz;

            const T termA1_i = (kern11_i + kern12_i + kern13_i) * W1;
            const T termA2_i = (kern21_i + kern22_i + kern23_i) * W1;
            const T termA3_i = (kern31_i + kern32_i + kern33_i) * W1;

            const T termA1_j = (kern11_j + kern12_j + kern13_j) * W2;
            const T termA2_j = (kern21_j + kern22_j + kern23_j) * W2;
            const T termA3_j = (kern31_j + kern32_j + kern33_j) * W2;

            const T pro_i = p[i] / (gradh_i * ro[i] * ro[i]);
            const T pro_j = p[j] / (gradh_j * ro[j] * ro[j]);

            const T r_square = dist * dist;
            const T viscosity_ij = artificial_viscosity(ro[i], ro[j], h[i], h[j], c[i], c[j], rv, r_square);

            const T grad_Px_AV = 0.5 * (m[i] / ro[i] * viscosity_ij * termA1_i + m[j] / ro[j] * viscosity_ij * termA1_j);
            const T grad_Py_AV = 0.5 * (m[i] / ro[i] * viscosity_ij * termA2_i + m[j] / ro[j] * viscosity_ij * termA2_j);
            const T grad_Pz_AV = 0.5 * (m[i] / ro[i] * viscosity_ij * termA3_i + m[j] / ro[j] * viscosity_ij * termA3_j);

            momentum_x += m[j] * (pro_i * termA1_i + pro_j * termA1_j) + grad_Px_AV;
            momentum_y += m[j] * (pro_i * termA2_i + pro_j * termA2_j) + grad_Py_AV;
            momentum_z += m[j] * (pro_i * termA3_i + pro_j * termA3_j) + grad_Pz_AV;

            energy += m[j] * 2.0 * pro_i * (v_ijx * termA1_i + v_ijy * termA2_i + v_ijz * termA3_i);
            energyAV += grad_Px_AV * v_ijx + grad_Py_AV * v_ijy + grad_Pz_AV * v_ijz;
        }

        du[i] = 0.5 * (energy + energyAV);
        grad_P_x[i] = momentum_x;
        grad_P_y[i] = momentum_y;
        grad_P_z[i] = momentum_z;
    }
};
} // namespace sph
} // namespace sphexa
