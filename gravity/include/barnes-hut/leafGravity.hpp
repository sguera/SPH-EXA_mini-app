#pragma once

#include "sphexa.hpp"

#include "cstone/primitives/stl.hpp"
#include "cstone/tree/octree_internal.hpp"

namespace gravity
{

template <class T>
void gatherGravValues(GravityData<T> *gv)
{
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &(*gv).mTot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, &(*gv).xcm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &(*gv).ycm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &(*gv).zcm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, &(*gv).qxxa, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &(*gv).qxya, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &(*gv).qxza, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &(*gv).qyya, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &(*gv).qyza, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &(*gv).qzza, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}

/**
 * @brief Computes gravity forces for a given tree node represented by the start and end morton codes
 *
 * @tparam I T float or double.
 * @tparam T I 32- or 64-bit unsigned.
 * \param[in] firstCode   lower Morton code
 * \param[in] secondCode  upper Morton code
 * @param list
 * @param x List of x coordinates.
 * @param y List of y coordinates.
 * @param z List of z coordinates.
 * @param m Vector of masses.
 * @param codes
 * @param box
 * @param withGravitySync
 */
template <class I, class T>
CUDA_HOST_DEVICE_FUN GravityData<T> computeNodeGravity(const T *x, const T *y, const T *z, const T *m, int nParticles, T xmin, T xmax,
                                                       T ymin, T ymax, T zmin, T zmax, bool withGravitySync = false)
{
    GravityData<T> gv;

    gv.xce = (xmax + xmin) / 2.0;
    gv.yce = (ymax + ymin) / 2.0;
    gv.zce = (zmax + zmin) / 2.0;

    gv.dx = xmax - xmin;

    for (size_t i = 0; i < nParticles; ++i)
    {
        T xx = x[i];
        T yy = y[i];
        T zz = z[i];

        T m_i = m[i];

        gv.xcm += xx * m_i;
        gv.ycm += yy * m_i;
        gv.zcm += zz * m_i;

        gv.mTot += m_i;

        T rx = xx - gv.xce;
        T ry = yy - gv.yce;
        T rz = zz - gv.zce;

        gv.qxxa += rx * rx * m_i;
        gv.qxya += rx * ry * m_i;
        gv.qxza += rx * rz * m_i;
        gv.qyya += ry * ry * m_i;
        gv.qyza += ry * rz * m_i;
        gv.qzza += rz * rz * m_i;

        gv.particleIdx = i;
    }

    if (withGravitySync) gatherGravValues(&gv);

    if (nParticles > 1)
    {
        gv.xcm /= gv.mTot;
        gv.ycm /= gv.mTot;
        gv.zcm /= gv.mTot;

        T rx = gv.xce - gv.xcm;
        T ry = gv.yce - gv.ycm;
        T rz = gv.zce - gv.zcm;
        gv.qxx = gv.qxxa - rx * rx * gv.mTot;
        gv.qxy = gv.qxya - rx * ry * gv.mTot;
        gv.qxz = gv.qxza - rx * rz * gv.mTot;
        gv.qyy = gv.qyya - ry * ry * gv.mTot;
        gv.qyz = gv.qyza - ry * rz * gv.mTot;
        gv.qzz = gv.qzza - rz * rz * gv.mTot;

        gv.trq = gv.qxx + gv.qyy + gv.qzz;
    }
    else if (nParticles == 1)
    {
        size_t idx = gv.particleIdx;

        gv.xcm = x[idx];
        gv.ycm = y[idx];
        gv.zcm = z[idx];

        gv.xce = x[idx];
        gv.yce = y[idx];
        gv.zce = z[idx];

        gv.qxx = 0;
        gv.qxy = 0;
        gv.qxz = 0;
        gv.qyy = 0;
        gv.qyz = 0;
        gv.qzz = 0;

        gv.trq = 0;
        gv.dx = 0; // used to indicate that node is a leaf
    }
    gv.pcount = nParticles;
    return gv;
}

/**
 * @brief
 *
 * @tparam I
 * @tparam T
 * @param tree
 * @param x
 * @param y
 * @param z
 * @param m
 * @param codes
 * @param box
 * @param gravityTreeData
 * @param withGravitySync
 */
template <class I, class T>
void calculateLeafGravityData(const std::vector<I> &tree, const std::vector<T> &x,
                              const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m, const std::vector<I> &codes,
                              const cstone::Box<T> &box, GravityTree<T> &gravityTreeData, bool withGravitySync = false)
{
    int i = 0;
    for (auto it = tree.begin(); it + 1 != tree.end(); ++it)
    {
        I firstCode = *it;
        I secondCode = *(it + 1);

        // TODO: Search only from codes+lastParticle to the end since we know particles can not be in multiple nodes
        int startIndex = stl::lower_bound(codes.data(), codes.data() + codes.size(), firstCode) - codes.data();
        int endIndex = stl::upper_bound(codes.data(), codes.data() + codes.size(), secondCode) - codes.data();
        int nParticles = endIndex - startIndex;
        // NOTE: we should use node counts to get the last one, otherwise, we might find 0 particles where there should be 1 in case bucketsize is 1
        //int endIndex = startIndex + csCounts[i];
        //int nParticles = csCounts[i];
        // NOTE: using morton codes to compute geometrical center. It might not be accurate.
        I lastCode = codes[nParticles - 1];
        T xmin = decodeXCoordinate(firstCode, box);
        T xmax = decodeXCoordinate(lastCode, box);
        T ymin = decodeYCoordinate(firstCode, box);
        T ymax = decodeYCoordinate(lastCode, box);
        T zmin = decodeZCoordinate(firstCode, box);
        T zmax = decodeZCoordinate(lastCode, box);

        gravityTreeData[i++] =
            computeNodeGravity<I, T>(x.data() + startIndex, y.data() + startIndex, z.data() + startIndex, m.data() + startIndex, nParticles,
                                     xmin, xmax, ymin, ymax, zmin, zmax, withGravitySync);
    }
}

} // namespace gravity
