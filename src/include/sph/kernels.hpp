#pragma once

#include "math.hpp"
#include "BBox.hpp"

namespace sphexa
{

#define PI 3.14159265358979323846

#ifdef USE_STD_MATH_IN_KERNELS
#define math_namespace std
#else
#define math_namespace sphexa::math
#endif

#ifdef __NVCC__
#define CUDA_PREFIX __device__
#else
#define CUDA_PREFIX
#endif


template <typename T>
inline T compute_3d_k(T n)
{
    // b0, b1, b2 and b3 are defined in "SPHYNX: an accurate density-based SPH method for astrophysical applications",
    // DOI: 10.1051/0004-6361/201630208
    T b0 = 2.7012593e-2;
    T b1 = 2.0410827e-2;
    T b2 = 3.7451957e-3;
    T b3 = 4.7013839e-2;

    return b0 + b1 * std::sqrt(n) + b2 * n + b3 * std::sqrt(n * n * n);
}

template <typename T>
inline T wharmonic(T v, T h, T sincIndex, T K)
{
    T Pv = (PI / 2.0) * v;
    T sincv = math_namespace::sin(Pv) / Pv;
    return K / (h * h * h) * math_namespace::pow(sincv, (int)sincIndex);
}

template <typename T>
CUDA_PREFIX inline T wharmonic_derivative(T v, T h, T sincIndex, T K)
{

    T Pv = (PI / 2.0) * v;
    T cotv = math_namespace::cos(Pv) / math_namespace::sin(Pv);
    ; // 1.0 / tan(P * v);
    T sincv = math_namespace::sin(Pv) / (Pv);
    T sincnv = math_namespace::pow(sincv, (int)sincIndex);
    T ret = sincIndex * (Pv * cotv - 1.0) * sincnv * (K / (h * h * h * h * h * v * v)); 
    // printf("wharmonic_derivative called with v=%f, cotv=%f, sincIndex=%f, ret=%f\n", v, cotv, sincIndex, ret);
    return ret;
}

template <typename T>
CUDA_PREFIX inline T artificial_viscosity(T ro_i, T ro_j, T h_i, T h_j, T c_i, T c_j, T rv, T r_square)
{
    T alpha = 1.0;
    T beta = 2.0;
    T epsilon = 0.01;

    T ro_ij = (ro_i + ro_j) / 2.0;
    T c_ij = (c_i + c_j) / 2.0;
    T h_ij = (h_i + h_j) / 2.0;

    // calculate viscosity_ij according to Monaghan & Gringold 1983
    T viscosity_ij = 0.0;
    if (rv < 0.0)
    {
        // calculate muij
        T mu_ij = (h_ij * rv) / (r_square + epsilon * h_ij * h_ij);
        viscosity_ij = (-alpha * c_ij * mu_ij + beta * mu_ij * mu_ij) / ro_ij;
    }

    return viscosity_ij;
}

template <typename T>
CUDA_PREFIX inline void applyPBC(const BBox<T> &bbox, const T r, T &xx, T &yy, T &zz)
{
    if (bbox.PBCx && xx > r)
        xx -= (bbox.xmax - bbox.xmin);
    else if (bbox.PBCx && xx < -r)
        xx += (bbox.xmax - bbox.xmin);

    if (bbox.PBCy && yy > r)
        yy -= (bbox.ymax - bbox.ymin);
    else if (bbox.PBCy && yy < -r)
        yy += (bbox.ymax - bbox.ymin);

    if (bbox.PBCz && zz > r)
        zz -= (bbox.zmax - bbox.zmin);
    else if (bbox.PBCz && zz < -r)
        zz += (bbox.zmax - bbox.zmin);

    // xx += bbox.PBCx * ((xx < -r) - (xx > r)) * (bbox.xmax-bbox.xmin);
    // yy += bbox.PBCy * ((yy < -r) - (yy > r)) * (bbox.ymax-bbox.ymin);
    // zz += bbox.PBCz * ((zz < -r) - (zz > r)) * (bbox.zmax-bbox.zmin);
}

template <typename T>
inline T distancePBC(const BBox<T> &bbox, const T hi, const T x1, const T y1, const T z1, const T x2, const T y2, const T z2)
{
    T xx = x1 - x2;
    T yy = y1 - y2;
    T zz = z1 - z2;

    applyPBC<T>(bbox, 2.0 * hi, xx, yy, zz);

    return std::sqrt(xx * xx + yy * yy + zz * zz);
}

} // namespace sphexa
