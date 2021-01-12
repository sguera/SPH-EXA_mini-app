#pragma once

#include <cmath>
#include <map>
#include <mutex>
#include <unordered_set>

#include "../Octree.hpp"

namespace sphexa
{
namespace sph
{

constexpr static double gravityTolerance = 0.5;
std::mutex mtx;

using RankToParticles = std::map<int, std::unordered_set<int>>;

template <typename T>
void treeWalkRef(const Octree<T> &node, const int i, const T *xi, const T *yi, const T *zi, const T *hi, const T *hj, const T *mj, T *fx,
                 T *fy, T *fz, T *ugrav, RankToParticles &particlesForRemoteGravCalculations)
{
    const auto gnode = dynamic_cast<const GravityOctree<T> &>(node);

    // Skip empty treenodes that are not part of global tree
    // Carefull! empty global tree nodes cannot be skipped. They are used to fill RankToParticles map used later in remote grav calculations
    if (gnode.particleIdxList.empty() && !gnode.global) return;

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
#ifndef NDEBUG
                if (std::isnan(fx[i])) printf("i=%d fx[i]=%.15f, g0=%f\n", i, fx[i], g0);
#endif
            }
        }
        else
        {
            for (const auto &child : gnode.cells) // go deeper to the childs
            {
                if (child->global && child->assignee != child->comm_rank && child->assignee != -1)
                {
                    std::lock_guard<std::mutex> l(mtx);
                    particlesForRemoteGravCalculations[child->assignee].insert(i);
                    continue;
                }

                treeWalkRef(*child, i, xi, yi, zi, hi, hj, mj, fx, fy, fz, ugrav, particlesForRemoteGravCalculations);
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

#ifndef NDEBUG
                if (std::isnan(fx[i])) printf("NE i=%d fx[i]=%.15f g0=%f\n", i, fx[i], g0);
#endif
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

#ifndef NDEBUG
                if (std::isnan(c1))
                {
                    printf("r7=%e dd2=%e dd5=%e r1=%e r2=%e r3=%e\n", r7, dd2, dd5, r1, r2, r3);
                    exit(0);
                }

                if (std::isnan(fx[i]))
                {
                    printf("NI, NL i=%d fx[i]=%f c1=%f c2=%f c3=%f gnode.trq=%f\n", i, fx[i], c1, c2, c3, gnode.trq);
                    exit(0);
                }
#endif
            }
        }
        else // go deeper
        {
            for (const auto &child : gnode.cells) // go deeper to the childs
            {
                if (child->global && child->assignee != child->comm_rank && child->assignee != -1)
                {
                    std::lock_guard<std::mutex> l(mtx);
                    particlesForRemoteGravCalculations[child->assignee].insert(i);
                    continue;
                }

                treeWalkRef(*child, i, xi, yi, zi, hi, hj, mj, fx, fy, fz, ugrav, particlesForRemoteGravCalculations);
            }
        }
    }
}

template <typename T, typename Dataset>
void gravityTreeWalkImpl(const Task &t, const GravityOctree<T> &tree, Dataset &d, RankToParticles &particlesForRemoteGravCalculations)
{
    const size_t n = t.clist.size();
    const int *clist = t.clist.data();
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

        treeWalkRef(tree, i, xi, yi, zi, hi, hj, mj, fx, fy, fz, ugrav, particlesForRemoteGravCalculations);
    }
}

template <typename T, class Dataset>
RankToParticles gravityTreeWalk(const std::vector<Task> &taskList, const GravityOctree<T> &tree, Dataset &d)
{
    RankToParticles particlesForRemoteGravCalculations;

    for (const auto &task : taskList)
    {
        gravityTreeWalkImpl<T>(task, tree, d, particlesForRemoteGravCalculations);
    }

    return particlesForRemoteGravCalculations;
}

} // namespace sph
} // namespace sphexa
