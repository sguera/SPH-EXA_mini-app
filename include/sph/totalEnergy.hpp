#pragma once

#include <vector>
#include <iostream>

#ifdef USE_MPI
#include "mpi.h"
#endif

namespace sphexa
{
namespace sph
{
template <typename T, class Dataset>
void computeTotalEnergyImpl(const Task &t, Dataset &d, T &ecin, T &eint)
{
    const size_t n = t.clist.size();
    const int *clist = t.clist.data();

    const T *u = d.u.data();
    const T *vx = d.vx.data();
    const T *vy = d.vy.data();
    const T *vz = d.vz.data();
    const T *m = d.m.data();

    T ecintmp = 0.0, einttmp = 0.0;
#pragma omp parallel for reduction(+ : ecintmp, einttmp)
    for (size_t pi = 0; pi < n; pi++)
    {
        int i = clist[pi];

        T vmod2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];

#ifdef NDEBUG
        if (std::isnan(u[i]) || std::isnan(m[i]) || std::isnan(vmod2))
            printf("ERROR::TotalEnergy(%d) u[i]=%f m[i]=%f, vmod2=%f\n", i, u[i], m[i], vmod2);
#endif

        ecintmp += 0.5 * m[i] * vmod2;
        einttmp += u[i] * m[i];
    }

    ecin = ecintmp;
    eint = einttmp;
}

template <typename T, class Dataset>
void computeTotalEnergy(const std::vector<Task> &taskList, Dataset &d)
{
    T ecintmp = 0, einttmp = 0;
    for (const auto &task : taskList)
    {
        T tecin = 0, teint = 0;
        computeTotalEnergyImpl<T>(task, d, tecin, teint);
        ecintmp += tecin;
        einttmp += teint;
    }

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &ecintmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &einttmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

#ifdef NDEBUG
    if (std::isnan(ecintmp) || std::isnan(einttmp)) printf("ERROR::TotalEnergy tecin=%f teint=%f\n", ecintmp, einttmp);
#endif

    d.ecin = ecintmp;
    d.eint = einttmp;
    d.etot = ecintmp + einttmp;
}

template <typename T, class Dataset>
void computeTotalEnergyWithGravityImpl(const Task &t, Dataset &d, T &ecin, T &eint, T &egrav)
{
    const size_t n = t.clist.size();
    const int *clist = t.clist.data();

    const T *u = d.u.data();
    const T *vx = d.vx.data();
    const T *vy = d.vy.data();
    const T *vz = d.vz.data();
    const T *m = d.m.data();
    const T *ugrav = d.ugrav.data();

    const T g = d.g;

    T ecintmp = 0.0, einttmp = 0.0, egravtmp = 0.0;
#pragma omp parallel for reduction(+ : ecintmp, einttmp) reduction(- : egravtmp)
    for (size_t pi = 0; pi < n; pi++)
    {
        int i = clist[pi];

        T vmod2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];

#ifdef NDEBUG
        if (std::isnan(u[i]) || std::isnan(m[i]) || std::isnan(vmod2))
        {
            printf("ERROR::TotalEnergy(%d) u[i]=%f m[i]=%f, vmod2=%f\n", i, u[i], m[i], vmod2);
            exit(EXIT_FAILURE);
        }

#endif

        ecintmp += 0.5 * m[i] * vmod2;
        einttmp += u[i] * m[i];
        egravtmp -= m[i] * g * 0.5 * ugrav[i];
    }

    ecin = ecintmp;
    eint = einttmp;
    egrav = egravtmp;
}

template <typename T, class Dataset>
void computeTotalEnergyWithGravity(const std::vector<Task> &taskList, Dataset &d)
{
    T ecintmp = 0, einttmp = 0, egravtmp = 0;
    for (const auto &task : taskList)
    {
        T tecin = 0, teint = 0, tegrav = 0;
        computeTotalEnergyWithGravityImpl<T>(task, d, tecin, teint, tegrav);
        ecintmp += tecin;
        einttmp += teint;
        egravtmp += tegrav;
    }

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &ecintmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &einttmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &egravtmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

#ifdef NDEBUG
    if (std::isnan(ecintmp) || std::isnan(einttmp) || std::isnan(egravtmp))
        printf("ERROR::TotalEnergy tecin=%f teint=%f, tegrav=%f\n", ecintmp, einttmp, egravtmp);
#endif

    d.ecin = ecintmp;
    d.eint = einttmp;
    d.egrav = egravtmp;
    d.etot = ecintmp + einttmp + egravtmp;
}

} // namespace sph
} // namespace sphexa
