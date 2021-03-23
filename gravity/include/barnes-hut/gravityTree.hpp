#pragma once

#include "sphexa.hpp"

#include "cstone/primitives/stl.hpp"
#include "cstone/tree/octree_internal.hpp"

namespace gravity
{

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

    // GravityData& operator=(GravityData other) = delete;
};

template <class T>
using GravityTree = std::vector<GravityData<T>>;

template <class I, class T>
class GravityOctree : public cstone::Octree<I>
{
public:
    void update(const I *firstLeaf, const I *lastLeaf) { cstone::Octree<I>::update(firstLeaf, lastLeaf); }

    void buildGravityTree(const std::vector<I> &tree, const std::vector<unsigned> &nodeCounts, const std::vector<T> &x,
                          const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m, const std::vector<I> &codes,
                          const cstone::Box<T> &box, const cstone::SpaceCurveAssignment<I>& sfcAssignment)
    {
        leafData_.resize(cstone::nNodes(tree));
        calculateLeafGravityData(tree, nodeCounts, x, y, z, m, codes, box, leafData_, sfcAssignment);

        internalData_.resize(this->nTreeNodes() - cstone::nNodes(tree));
        recursiveBuildGravityTree(tree, *this, 0, leafData_, internalData_, x, y, z, m, codes, box);

        printf("TOTAL PARTICLE COUNT: %d pcout\n", internalData_[0].pcount);
    }

    const GravityTree<T> &leafData() const { return leafData_; }
    const GravityTree<T> &internalData() const { return internalData_; }

private:
    GravityTree<T> leafData_;
    GravityTree<T> internalData_;
};

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

    MPI_Allreduce(MPI_IN_PLACE, &(*gv).pcount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
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
GravityData<T> computeNodeGravity(const T *x, const T *y, const T *z, const T *m, const I *codes, int nParticles, T xmin, T xmax, T ymin, T ymax, T zmin,
                                  T zmax, int rank, const cstone::SpaceCurveAssignment<I>& assignment)
{
    GravityData<T> gv;

    gv.xce = (xmax + xmin) / 2.0;
    gv.yce = (ymax + ymin) / 2.0;
    gv.zce = (zmax + zmin) / 2.0;

    gv.dx = abs(xmax - xmin);

    int localParticles = 0;

    for (size_t i = 0; i < nParticles; ++i)
    {
        bool halo = false;
        for (int range = 0 ; range < assignment.nRanges(rank); ++ range)
        {
            if(codes[i] < assignment.rangeStart(rank, range) || codes[i] >= assignment.rangeEnd(rank, range))
            {
                halo = true;
            }
        }
        if (halo) continue;
        else localParticles ++;

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

    gv.pcount = localParticles;

    gatherGravValues(&gv);

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
    return gv;
}

template <class I>
void localParticleList(std::vector<unsigned>& plist, const I* codes, int size, int start, int np, const cstone::SpaceCurveAssignment<I>& sfcAssignment)
{
    int end = start + np;
    for(int i = start ; i < end ; ++ i)
    {
        I code = codes[i];
    }

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
void calculateLeafGravityData(const std::vector<I> &tree, const std::vector<unsigned> &nodeCounts, const std::vector<T> &x,
                              const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m, const std::vector<I> &codes,
                              const cstone::Box<T> &box, GravityTree<T> &gravityTreeData, const cstone::SpaceCurveAssignment<I>& sfcAssignment)
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
        // NOTE: we should use node counts to get the last one, otherwise, we might find 0 particles where there should be 1 in case
        // bucketsize is 1
        // int endIndex = startIndex + nodeCounts[i];
        assert(nParticles == nodeCounts[i]);
        // NOTE: using morton codes to compute geometrical center. It might not be accurate.
        I lastCode = codes[startIndex + nParticles - 1];

        I endCode = secondCode - 1;
        // we use morton codes from the global tree to compute the same box coordinates accross all ranks without reduction
        T xmin = decodeXCoordinate(firstCode, box);
        T xmax = decodeXCoordinate(endCode, box);
        T ymin = decodeYCoordinate(firstCode, box);
        T ymax = decodeYCoordinate(endCode, box);
        T zmin = decodeZCoordinate(firstCode, box);
        T zmax = decodeZCoordinate(endCode, box);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        gravityTreeData[i] = computeNodeGravity<I, T>(x.data() + startIndex, y.data() + startIndex, z.data() + startIndex,
                                                      m.data() + startIndex, codes.data() + startIndex, nParticles, xmin, xmax, ymin, ymax, zmin, zmax, rank, sfcAssignment);
        i++;
    }
}

/*
    TODO: check that pcounts in leaf nodes sums up to the number of particles
    unsigned int pcounts = 0;
    for (auto x : gravityOctree.leafData())
    {
        pcounts += x.pcount;
    }
    printf("total pcounts = %d\n", pcounts);
*/

/**
 * @brief aggregate using parallel axis theorem
 *
 * @tparam I
 * @tparam T
 * @param tree
 * @param octree
 * @param i
 * @param gravityLeafData
 * @param gravityInternalData
 * @param x
 * @param y
 * @param z
 * @param m
 * @param codes
 * @param box
 */
template <class I, class T>
void aggregateNodeGravity(const std::vector<I> &tree, const GravityOctree<I, T> &octree, cstone::TreeNodeIndex i,
                          GravityTree<T> &gravityLeafData, GravityTree<T> &gravityInternalData, const std::vector<T> &x,
                          const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m, const std::vector<I> &codes,
                          const cstone::Box<T> &box)
{
    // cstone::OctreeNode<I> node = localTree.internalTree()[i];

    GravityData<T> gv;

    pair<T> xrange = octree.x(i, box);
    pair<T> yrange = octree.y(i, box);
    pair<T> zrange = octree.z(i, box);
    gv.xce = (xrange[1] + xrange[0]) / 2.0;
    gv.yce = (yrange[1] + yrange[0]) / 2.0;
    gv.zce = (zrange[1] + zrange[0]) / 2.0;
    gv.dx = abs(xrange[1] - xrange[0]);

    for (int j = 0; j < 8; ++j)
    {
        // cstone::TreeNodeIndex child = node.child[j];
        cstone::TreeNodeIndex child = octree.childDirect(i, j);
        GravityData<T> current;
        // if (node.childType[j] == cstone::OctreeNode<I>::ChildType::internal) { current = gravityInternalData[child]; }
        if (!octree.isLeafChild(i, j)) { current = gravityInternalData[child]; }
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
        // cstone::TreeNodeIndex child = node.child[j];
        cstone::TreeNodeIndex child = octree.childDirect(i, j);
        GravityData<T> partialGravity;
        // if (node.childType[j] == cstone::OctreeNode<I>::ChildType::internal) { partialGravity = gravityInternalData[child]; }
        if (!octree.isLeafChild(i, j)) { partialGravity = gravityInternalData[child]; }
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
void recursiveBuildGravityTree(const std::vector<I> &tree, const GravityOctree<I, T> &octree, cstone::TreeNodeIndex i,
                               GravityTree<T> &gravityLeafData, GravityTree<T> &gravityInternalData, const std::vector<T> &x,
                               const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m, const std::vector<I> &codes,
                               const cstone::Box<T> &box)
{
    // cstone::OctreeNode<I> node = octree.internalTree()[i];

    for (int j = 0; j < 8; ++j)
    {
        // cstone::TreeNodeIndex child = node.child[j];
        cstone::TreeNodeIndex child = octree.childDirect(i, j);
        // if (node.childType[j] == cstone::OctreeNode<I>::ChildType::internal)
        if (!octree.isLeafChild(i, j))
        {
            recursiveBuildGravityTree(tree, octree, child, gravityLeafData, gravityInternalData, x, y, z, m, codes, box);
        }
    }
    aggregateNodeGravity(tree, octree, i, gravityLeafData, gravityInternalData, x, y, z, m, codes, box);
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
buildGravityTree(const std::vector<I> &tree, const std::vector<unsigned> &nodeCounts, const GravityOctree<I, T> &octree,
                 const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m,
                 const std::vector<I> &codes, const cstone::Box<T> &box, bool withGravitySync = false)
{
    GravityTree<T> gravityLeafData(cstone::nNodes(tree));
    calculateLeafGravityData(tree, nodeCounts, x, y, z, m, codes, box, gravityLeafData, false);

    GravityTree<T> gravityInternalData(octree.nTreeNodes() - cstone::nNodes(tree));
    recursiveBuildGravityTree(tree, octree, 0, gravityLeafData, gravityInternalData, x, y, z, m, codes, box);

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
