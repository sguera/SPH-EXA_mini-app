#pragma once

#include <math.h>
#include <vector>
// #include "config.hpp"

#ifdef USE_MPI
#include "mpi.h"
#endif

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
#endif

    T xmin, xmax, ymin, ymax, zmin, zmax;
    bool PBCx, PBCy, PBCz;
};

} // namespace sphexa
