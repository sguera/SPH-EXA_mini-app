#pragma once

#include "sphexa.hpp"
#include "gravityTree.hpp"
#include <sys/time.h>

namespace gravity
{

constexpr static double gravityTolerance = 0.5;

template <class I, class T>
void gravityTreeWalkParticle(const std::vector<I> &tree, cstone::TreeNodeIndex nodeIdx,
                             const GravityOctree<I, T> &gravityOctree, const int i, const I *codes, const T *xi, const T *yi, const T *zi,
                             const T *hi, const T *hj, const T *mj, T *fx, T *fy, T *fz, T *ugrav)
{
    //const std::vector<cstone::OctreeNode<I>> &internalTree = localTree.internalTree();
    //const cstone::OctreeNode<I> &tnode = internalTree[nodeIdx];

    for (int c = 0; c < 8; ++c) // go deeper to the childs
    {
        // TODO: Check if we need to store it for remote
        cstone::TreeNodeIndex childIdx = gravityOctree.childDirect(nodeIdx, c);
        // TODO: use the gravity octree interface to get either a leaf or an internal data
        const GravityData<T> &gnode =
            (gravityOctree.isLeafChild(nodeIdx, c)) ? gravityOctree.leafData()[childIdx] : gravityOctree.internalData()[childIdx];

        if (gnode.pcount == 0) continue;

        const T d1 = std::abs(xi[i] - gnode.xce);
        const T d2 = std::abs(yi[i] - gnode.yce);
        const T d3 = std::abs(zi[i] - gnode.zce);
        const T dc = 4.0 * hi[i] + gnode.dx / 2.0;

        if (d1 <= dc && d2 <= dc && d3 <= dc) // intersecting
        {
            if (gnode.dx == 0) // node is a leaf
            {
                // If tree node assignee is -1 it means that this tree node is shared across a few computing nodes.
                // uncomment the if below if you want to skip calculating gravity contribution of this nodes
                // if (gnode.assignee == -1) return;

                const auto j = gnode.particleIdx;

                // if (i != j) // skip calculating gravity contribution of myself
                if (!(xi[i] == gnode.xce && yi[i] == gnode.yce && zi[i] == gnode.zce))
                {
                    const T dd2 = d1 * d1 + d2 * d2 + d3 * d3;
                    const T dd5 = std::sqrt(dd2);

                    T g0;

                    if (dd5 > 2.0 * hi[i] && dd5 > 2.0 * hj[j]) { g0 = 1.0 / dd5 / dd2; }
                    else
                    {
                        const T hij = hi[i] + hj[j];
                        const T vgr = dd5 / hij;
                        const T mefec = std::min(1.0, vgr * vgr * vgr);
                        g0 = mefec / dd5 / dd2;
                    }
                    const T r1 = xi[i] - gnode.xcm;
                    const T r2 = yi[i] - gnode.ycm;
                    const T r3 = zi[i] - gnode.zcm;

                    fx[i] -= g0 * r1 * mj[j];
                    fy[i] -= g0 * r2 * mj[j];
                    fz[i] -= g0 * r3 * mj[j];
                    ugrav[i] += g0 * dd2 * mj[j];
                }
            }
            else
            {
                if (!gravityOctree.isLeafChild(nodeIdx, c))
                {
                    gravityTreeWalkParticle(tree, childIdx, gravityOctree, i, codes, xi, yi, zi, hi, hj, mj, fx, fy, fz,
                                            ugrav);
                }
                else
                {
                    // TODO: leaf node contains a cluster, iterate through particles
                    // printf("[WARNING] Going deeper into a leaf node. This should not be the case!\n");
                }
            }
        }
        else // not intersecting
        {
            const T r1 = xi[i] - gnode.xcm;
            const T r2 = yi[i] - gnode.ycm;
            const T r3 = zi[i] - gnode.zcm;
            const T dd2 = r1 * r1 + r2 * r2 + r3 * r3;

            if (gnode.dx * gnode.dx <= gravityTolerance * dd2)
            {
                // If tree node assignee is -1 it means that this tree node is shared across a few computing nodes.
                // uncomment the if below if you want to skip calculating gravity contribution of this nodes
                // if (gnode.assignee == -1) return;

                const T dd5 = sqrt(dd2);
                const T d32 = 1.0 / dd5 / dd2;

                T g0;

                if (gnode.dx == 0) // node is a leaf
                {
                    const int j = gnode.particleIdx;
                    const T v1 = dd5 / hi[i];
                    const T v2 = dd5 / hj[j];

                    if (v1 > 2.0 && v2 > 2.0) { g0 = gnode.mTot * d32; }
                    else
                    {
                        const T hij = hi[i] + hj[j];
                        const T vgr = dd5 / hij;
                        const T mefec = std::min(1.0, vgr * vgr * vgr);
                        g0 = mefec * d32 * gnode.mTot;
                    }

                    fx[i] -= g0 * r1;
                    fy[i] -= g0 * r2;
                    fz[i] -= g0 * r3;
                    ugrav[i] += g0 * dd2;
                }
                else // node is not leaf
                {
                    g0 = gnode.mTot * d32; // Base Value
                    fx[i] -= g0 * r1;
                    fy[i] -= g0 * r2;
                    fz[i] -= g0 * r3;
                    ugrav[i] += g0 * dd2; // eof Base value

                    const T r5 = dd2 * dd2 * dd5;
                    const T r7 = r5 * dd2;

                    const T qr1 = r1 * gnode.qxx + r2 * gnode.qxy + r3 * gnode.qxz;
                    const T qr2 = r1 * gnode.qxy + r2 * gnode.qyy + r3 * gnode.qyz;
                    const T qr3 = r1 * gnode.qxz + r2 * gnode.qyz + r3 * gnode.qzz;

                    const T rqr = r1 * qr1 + r2 * qr2 + r3 * qr3;

                    const T c1 = (-7.5 / r7) * rqr;
                    const T c2 = 3.0 / r5;
                    const T c3 = 0.5 * gnode.trq;

                    fx[i] += c1 * r1 + c2 * (qr1 + c3 * r1);
                    fy[i] += c1 * r2 + c2 * (qr2 + c3 * r2);
                    fz[i] += c1 * r3 + c2 * (qr3 + c3 * r3);
                    ugrav[i] -= (1.5 / r5) * rqr + c3 * d32;
                }
            }
            else // go deeper
            {
                if (!gravityOctree.isLeafChild(nodeIdx, c))
                {
                    gravityTreeWalkParticle(tree, childIdx, gravityOctree, i, codes, xi, yi, zi, hi, hj, mj, fx, fy, fz,
                                            ugrav);
                }
                else
                {
                    // printf("[WARNING] Going deeper into a leaf node. This should not be the case!\n");
                }
            }
        }
    }
}

template <class I, class T, class Dataset>
void gravityTreeWalkTask(const sphexa::Task &t, Dataset &d, const std::vector<I> &tree,
                         const GravityOctree<I, T> &gravityOctree, const cstone::Box<T> &box,
                         bool withGravitySync = false)
{
    const size_t n = t.clist.size();
    const int *clist = t.clist.data();

    const I *co = d.codes.data();
    const T *xi = d.x.data();
    const T *yi = d.y.data();
    const T *zi = d.z.data();
    const T *hi = d.h.data();

    const T *hj = d.h.data();
    const T *mj = d.m.data();

    T *fx = d.fx.data();
    T *fy = d.fy.data();
    T *fz = d.fz.data();
    T *ugrav = d.ugrav.data();

#pragma omp parallel for schedule(guided)
    for (size_t pi = 0; pi < n; ++pi)
    {
        const int i = clist[pi];
        fx[i] = fy[i] = fz[i] = ugrav[i] = 0.0;

        // if (i < 1)
        {
            gravityTreeWalkParticle(tree, 0, gravityOctree, i, co, xi, yi, zi, hi, hj, mj, fx, fy, fz, ugrav);
            //printf("%d\n", count);
            //assert(count == 65536);

            // printf("i=%d fx[i]=%.15f, fy[i]=%.15f, fz[i]=%.15f, ugrav[i]=%f\n", i, fx[i], fy[i], fz[i], ugrav[i]);
        }
    }
}

template <class I, class T, class Dataset>
void gravityTreeWalk(std::vector<sphexa::Task> &taskList, const std::vector<I> &tree, Dataset &d, const GravityOctree<I, T> &gravityOctree,
                     const cstone::Box<T> &box)
{
    for (const auto &task : taskList)
    {
        gravityTreeWalkTask(task, d, tree, gravityOctree, box);
    }
}

} // namespace gravity
