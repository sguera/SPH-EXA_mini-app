#pragma once

#include "sphexa.hpp"

#include "cstone/bsearch.hpp"
#include "cstone/octree_internal.hpp"

namespace gravity
{

template <class I>
using InternalNode = std::pair<I, I>;

template <class T>
struct GravityData
{
    T mTot = 0.0;
    T xce, yce, zce;
    T xcm = 0.0, ycm = 0.0, zcm = 0.0;

    T qxx = 0.0, qxy = 0.0, qxz = 0.0;
    T qyy = 0.0, qyz = 0.0;
    T qzz = 0.0;

    T qxxa = 0.0, qxya = 0.0, qxza = 0.0;
    T qyya = 0.0, qyza = 0.0;
    T qzza = 0.0;

    T trq = 0.0;
    int pcount = 0;

    // std::vector<int> particleIdxList;
    // std::vector<int> globalParticleIdxList;

    T dx;                // side of a cell;
    int particleIdx = 0; // filled only if node is a leaf

    void print() { printf("mTot = %.15f, qxx = %.15f, trq = %.15f, xcm = %.15f\n", mTot, qxx, trq, xcm); }
};

/*
template <class I, class T>
using TreeData = std::map<InternalNode<I>, GravityData<T>>;
*/
template <class T>
using GravityTree = std::vector<GravityData<T>>;

template <class T>
void gatherGravValues(GravityData<T> *gv, bool global, int assignee)
{
#ifdef USE_MPI
    if (global && assignee == -1)
    {
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
    }
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

    bool global = false;
    int assignee = -1;
    if (withGravitySync) gatherGravValues(&gv, global, assignee);

    if (nParticles > 1 || global)
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
void calculateLeafGravityData(const std::vector<I> &tree, const std::vector<unsigned> &csCounts, const std::vector<T> &x,
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
        // int endIndex = stl::upper_bound(codes.data(), codes.data() + codes.size(), secondCode) - codes.data();
        // int nParticles = endIndex - startIndex;
        int endIndex = startIndex + csCounts[i];
        int nParticles = csCounts[i];
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

template <class I, class T>
void aggregateNodeGravity(const std::vector<I> &tree, cstone::Octree<I, cstone::LocalTree> &localTree, cstone::TreeNodeIndex i,
                          GravityTree<T> &gravityLeafData, GravityTree<T> &gravityInternalData, const std::vector<T> &x,
                          const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m, const std::vector<I> &codes,
                          const cstone::Box<T> &box)
{
    cstone::OctreeNode<I> node = localTree.internalTree()[i];

    GravityData<T> gv;

    cstone::pair<T> xrange = localTree.x(i, box);
    cstone::pair<T> yrange = localTree.y(i, box);
    cstone::pair<T> zrange = localTree.z(i, box);
    gv.xce = (xrange[1] + xrange[0]) / 2.0;
    gv.yce = (yrange[1] + yrange[0]) / 2.0;
    gv.zce = (zrange[1] + zrange[0]) / 2.0;
    gv.dx = xrange[1] - xrange[0];

    for (int j = 0; j < 8; ++j)
    {
        cstone::TreeNodeIndex child = node.child[j];
        GravityData<T> current;
        if (node.childType[j] == cstone::OctreeNode<I>::ChildType::internal) { current = gravityInternalData[child]; }
        else
        {
            current = gravityLeafData[child];
        }
        gv.mTot += current.mTot;
        gv.xcm += current.xcm * current.mTot;
        gv.ycm += current.ycm * current.mTot;
        gv.zcm += current.zcm * current.mTot;
    }
    gv.xcm /= gv.mTot;
    gv.ycm /= gv.mTot;
    gv.zcm /= gv.mTot;

    size_t n = codes.size();

    for (int j = 0; j < 8; ++j)
    {
        cstone::TreeNodeIndex child = node.child[j];
        GravityData<T> partialGravity;
        if (node.childType[j] == cstone::OctreeNode<I>::ChildType::internal) { partialGravity = gravityInternalData[child]; }
        else
        {
            partialGravity = gravityLeafData[child];
        }

        T rx = partialGravity.xcm - gv.xcm;
        T ry = partialGravity.ycm - gv.ycm;
        T rz = partialGravity.zcm - gv.zcm;

        gv.qxxa += partialGravity.qxx + rx * rx * partialGravity.mTot;
        gv.qxya += partialGravity.qxy + rx * ry * partialGravity.mTot;
        gv.qxza += partialGravity.qxz + rx * rz * partialGravity.mTot;
        gv.qyya += partialGravity.qyy + ry * ry * partialGravity.mTot;
        gv.qyza += partialGravity.qyz + ry * rz * partialGravity.mTot;
        gv.qzza += partialGravity.qzz + rz * rz * partialGravity.mTot;

        gv.pcount += partialGravity.pcount;
    }

    if (gv.pcount == 1) gv.dx = 0;

    gv.qxx = gv.qxxa;
    gv.qxy = gv.qxya;
    gv.qxz = gv.qxza;
    gv.qyy = gv.qyya;
    gv.qyz = gv.qyza;
    gv.qzz = gv.qzza;

    gv.trq = gv.qxx + gv.qyy + gv.qzz;
    gravityInternalData[i] = gv;
}

template <class I, class T>
void recursiveBuildGravityTree(const std::vector<I> &tree, cstone::Octree<I, cstone::LocalTree> &localTree, cstone::TreeNodeIndex i,
                               GravityTree<T> &gravityLeafData, GravityTree<T> &gravityInternalData, const std::vector<T> &x,
                               const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m, const std::vector<I> &codes,
                               const cstone::Box<T> &box)
{
    cstone::OctreeNode<I> node = localTree.internalTree()[i];

    for (int j = 0; j < 8; ++j)
    {
        cstone::TreeNodeIndex child = node.child[j];
        if (node.childType[j] == cstone::OctreeNode<I>::ChildType::internal)
        {
            recursiveBuildGravityTree(tree, localTree, child, gravityLeafData, gravityInternalData, x, y, z, m, codes, box);
        }
    }
    aggregateNodeGravity(tree, localTree, i, gravityLeafData, gravityInternalData, x, y, z, m, codes, box);
}

/**
 * @brief
 *
 * @tparam I
 * @tparam T
 * @param tree
 * @param localTree
 * @param globalTree
 * @param x
 * @param y
 * @param z
 * @param m
 * @param codes
 * @param box
 * @param withGravitySync
 */
template <class I, class T>
std::tuple<GravityTree<T>, GravityTree<T>>
buildGravityTree(const std::vector<I> &tree, cstone::Octree<I, cstone::GlobalTree> &globalTree,
                 cstone::Octree<I, cstone::LocalTree> &localTree, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                 const std::vector<T> &m, const std::vector<I> &codes, const cstone::Box<T> &box, bool withGravitySync = false)
{
    std::vector<cstone::OctreeNode<I>> internalOctree = localTree.internalTree();
    const std::vector<I> cstree = localTree.tree();
    const std::vector<unsigned> csCounts = localTree.nodeCounts();

    GravityTree<T> gravityLeafData(cstone::nNodes(cstree));
    calculateLeafGravityData(cstree, csCounts, x, y, z, m, codes, box, gravityLeafData, false);

    GravityTree<T> gravityInternalData(internalOctree.size());
    recursiveBuildGravityTree(cstree, localTree, 0, gravityLeafData, gravityInternalData, x, y, z, m, codes, box);

    return std::make_tuple(std::move(gravityLeafData), std::move(gravityInternalData));
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
 */
template <class I, class T>
void showParticles(const std::vector<I> &tree, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                   const std::vector<T> &m, const std::vector<I> &codes, const cstone::Box<T> &box)
{
    // size_t n = cstone::nNodes(tree);
    size_t n = codes.size();
    size_t totalParticles = 0;
    size_t i = 0;
    for (auto it = tree.cbegin(); it + 1 != tree.cend(); ++it)
    {
        I firstCode = *it;
        I secondCode = *(it + 1);
        int startIndex = stl::lower_bound(codes.data(), codes.data() + n, firstCode) - codes.data();
        int endIndex = stl::upper_bound(codes.data(), codes.data() + n, secondCode) - codes.data();
        int nParticles = endIndex - startIndex;
        totalParticles += nParticles;

        T xmin = decodeXCoordinate(firstCode, box);
        T xmax = decodeXCoordinate(secondCode, box);
        T ymin = decodeYCoordinate(firstCode, box);
        T ymax = decodeYCoordinate(secondCode, box);
        T zmin = decodeZCoordinate(firstCode, box);
        T zmax = decodeZCoordinate(secondCode, box);

        int level = cstone::treeLevel(secondCode - firstCode);

        printf("%o, %o, %d, %d, %f %f %f %f %f %f\n", firstCode, secondCode, level, nParticles, xmin, xmax, ymin, ymax, zmin, zmax);

        for (int i = 0; i < nParticles; ++i)
        {
            // x.data()[startIndex + i];
        }
    }

    printf("found %ld particles\n", totalParticles);
}

} // namespace gravity
