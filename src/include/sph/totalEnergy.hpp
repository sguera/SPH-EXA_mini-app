#pragma once

#include <vector>
#include <iostream>
#include <tuple>

#ifdef USE_MPI
#include "mpi.h"
#endif

namespace sphexa
{
namespace sph
{
template <typename T, class Dataset>
void computeTotalEnergy(const std::vector<int> &l, Dataset &d)
{
    const int n = l.size();
    const int *clist = l.data();

    const T *u = d.u.data();
    const T *vx = d.vx.data();
    const T *vy = d.vy.data();
    const T *vz = d.vz.data();
    const T *m = d.m.data();
    T &etot = d.etot;
    T &ecin = d.ecin;
    T &eint = d.eint;

    T ecintmp = 0.0, einttmp = 0.0;
    //#pragma omp parallel for reduction(+ : ecintmp, einttmp)
    //for (int pi = 0; pi < n; pi++)
    //{
    //    int i = clist[pi];

    //    T vmod2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
    //    ecintmp += 0.5 * m[i] * vmod2;
    //    einttmp += u[i] * m[i];
    //}

    auto policy = hpx::parallel::execution::par;

    //auto sum_op1 =
    //    [&d](T tmp, int i)
    //    {
    //        T vmod2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
    //        tmp += 0.5 * d.m.at(i) * vmod2;
    //        return tmp;
    //    };
    //auto sum_op2 =
    //    [=](T tmp, int i)
    //    {
    //        tmp += u[i] * m[i];
    //        return tmp;
    //    };

    //ecintmp = hpx::parallel::reduce(policy, clist, clist+n, ecintmp, sum_op1);
    //einttmp = hpx::parallel::reduce(policy, clist, clist+n, einttmp, sum_op2);

    std::vector<double> ecinV(n), eintV(n);
    hpx::parallel::for_loop(policy,
        0, n,
        [&ecinV, &eintV, clist, vx, vy, vz, m, u](int pi)
    {
        int i = clist[pi];
        T vmod2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
        ecinV[i] = 0.5 * m[i] * vmod2;
        eintV[i] = u[i] * m[i]; 
    });

    ecintmp = hpx::parallel::reduce(policy, std::begin(ecinV), std::end(ecinV), ecintmp);
    einttmp = hpx::parallel::reduce(policy, std::begin(eintV), std::end(eintV), einttmp);

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &ecintmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &einttmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    ecin = ecintmp;
    eint = einttmp;
    etot = ecin + eint;
}
} // namespace sph
} // namespace sphexa
