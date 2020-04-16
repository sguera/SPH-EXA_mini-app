#pragma once

#include <math.h>
#include <vector>

#ifdef USE_MPI
#include "mpi.h"
#endif

#include "cudaFunctionAnnotation.hpp"

namespace sphexa
{

template <typename T>
class BBox
{
public:
    BBox(T xmin = -1, T xmax = 1, T ymin = -1, T ymax = 1, T zmin = -1, T zmax = 1, bool PBCx = false, bool PBCy = false, bool PBCz = false)
        : xmin(xmin)
        , xmax(xmax)
        , ymin(ymin)
        , ymax(ymax)
        , zmin(zmin)
        , zmax(zmax)
        , PBCx(PBCx)
        , PBCy(PBCy)
        , PBCz(PBCz)
    {
    }

    void setBox(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax, bool PBCx, bool PBCy, bool PBCz)
    {
        this->xmin = xmin;
        this->xmax = xmax;
        this->ymin = ymin;
        this->ymax = ymax;
        this->zmin = zmin;
        this->zmax = zmax;
        this->PBCx = PBCx;
        this->PBCy = PBCy;
        this->PBCz = PBCz;
    }


    inline void compute(const std::vector<int> &clist, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z)
    {
        if (!PBCx) xmin = INFINITY;
        if (!PBCx) xmax = -INFINITY;
        if (!PBCy) ymin = INFINITY;
        if (!PBCy) ymax = -INFINITY;
        if (!PBCz) zmin = INFINITY;
        if (!PBCz) zmax = -INFINITY;

        for (int i = 0; i < (int)clist.size(); i++)
        {
            T xx = x[clist[i]];
            T yy = y[clist[i]];
            T zz = z[clist[i]];

            if (!PBCx && xx < xmin) xmin = xx;
            if (!PBCx && xx > xmax) xmax = xx;
            if (!PBCy && yy < ymin) ymin = yy;
            if (!PBCy && yy > ymax) ymax = yy;
            if (!PBCz && zz < zmin) zmin = zz;
            if (!PBCz && zz > zmax) zmax = zz;
        }
    }

    inline void compute(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z)
    {
        if (!PBCx) xmin = INFINITY;
        if (!PBCx) xmax = -INFINITY;
        if (!PBCy) ymin = INFINITY;
        if (!PBCy) ymax = -INFINITY;
        if (!PBCz) zmin = INFINITY;
        if (!PBCz) zmax = -INFINITY;

        for (int i = 0; i < (int)x.size(); i++)
        {
            T xx = x[i];
            T yy = y[i];
            T zz = z[i];

            if (!PBCx && xx < xmin) xmin = xx;
            if (!PBCx && xx > xmax) xmax = xx;
            if (!PBCy && yy < ymin) ymin = yy;
            if (!PBCy && yy > ymax) ymax = yy;
            if (!PBCz && zz < zmin) zmin = zz;
            if (!PBCz && zz > zmax) zmax = zz;
        }
    }

#ifdef USE_MPI
    inline void computeGlobal(const std::vector<int> &clist, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z)
    {
        compute(clist, x, y, z);

        MPI_Allreduce(MPI_IN_PLACE, &xmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &ymin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &zmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &xmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &ymax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &zmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    inline void computeGlobal(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z)
    {
        compute(x, y, z);

        MPI_Allreduce(MPI_IN_PLACE, &xmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &ymin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &zmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &xmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &ymax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &zmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
#else
    inline void computeGlobal(const std::vector<int> &clist, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z)
    {
        compute(clist, x, y, z);
    }

    inline void computeGlobal(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z)
    {
        compute(x, y, z);
    }
#endif

    T xmin, xmax, ymin, ymax, zmin, zmax;
    bool PBCx, PBCy, PBCz;
};

template <typename T>
CUDA_DEVICE_HOST_FUN inline void applyPBC(const BBox<T> &bbox, const T r, T &xx, T &yy, T &zz)
{
    // todo: add warning after update of h if h is bigger than the half-width of the bbox of any dimension...
    //       but this is actually not a problem for only the bbox, but a limitation/assumption of the domain/simulation
    //       architecture. We only allow wrap-around once (i.e. a particle cannot be a neighbor of another particle
    //       multiple times (e.g once "directly" and once "through a PBC" then again through applying the PBC a
    //       second time...)
    //
    // todo: refactor such that parameter r is removed. It actually has no influence on whether applying the PBC
    //       has an effect...
    //
    // todo: optimize! This function is called for basically every kernel call and the bbox bounds don't change
    //       in an iteration until positions are updated -> probably makes sense to store the widths in the bbox
    //       after the update, no?


    const T xWidth = bbox.xmax - bbox.xmin;
    if (bbox.PBCx && xx > xWidth / 2.0)
        xx -= xWidth;
    else if (bbox.PBCx && xx < - xWidth / 2.0)
        xx += xWidth;

    const T yWidth = bbox.ymax - bbox.ymin;
    if (bbox.PBCy && yy > yWidth / 2.0)
        yy -= yWidth;
    else if (bbox.PBCy && yy < - yWidth / 2.0)
        yy += yWidth;

    const T zWidth = bbox.zmax - bbox.zmin;
    if (bbox.PBCz && zz > zWidth / 2.0)
        zz -= zWidth;
    else if (bbox.PBCz && zz < - zWidth / 2.0)
        zz += zWidth;

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
