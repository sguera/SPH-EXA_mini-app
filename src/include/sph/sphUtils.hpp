#pragma once


#include <cmath>

#include "cudaFunctionAnnotation.hpp"
#include "BBox.hpp"

namespace sphexa
{
template <typename T>
CUDA_DEVICE_HOST_FUN inline T compute_3d_k(T n)
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
CUDA_DEVICE_HOST_FUN inline void applyPBC(const BBox<T> &bbox, const T r, T &xx, T &yy, T &zz)
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
CUDA_DEVICE_HOST_FUN inline T distancePBC(const BBox<T> &bbox, const T hi, const T x1, const T y1, const T z1, const T x2, const T y2, const T z2)
{
    T xx = x1 - x2;
    T yy = y1 - y2;
    T zz = z1 - z2;

    applyPBC<T>(bbox, 2.0 * hi, xx, yy, zz);

    return std::sqrt(xx * xx + yy * yy + zz * zz);
}

} // namespace sphexa
