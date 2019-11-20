#pragma once

#include <cstdio>
#include <vector>
#include "BBox.hpp"
#include "sph/kernels.hpp"

namespace sphexa
{
template <typename T>
struct ParticlesData
{
    inline void resize(const size_t size)
    {
        for (unsigned int i = 0; i < data.size(); ++i)
            data[i]->resize(size);
    }

    int iteration;                               // Current iteration
    int n, side, count;                          // Number of particles
    std::vector<T> x, y, z, x_m1, y_m1, z_m1;    // Positions
    std::vector<T> vx, vy, vz;                   // Velocities
    std::vector<T> ro, ro_0;                     // Density
    std::vector<T> u;                            // Internal Energy
    std::vector<T> p, p_0;                       // Pressure
    std::vector<T> h;                            // Smoothing Length
    std::vector<T> m;                            // Mass
    std::vector<T> c;                            // Speed of sound
    std::vector<T> grad_P_x, grad_P_y, grad_P_z; // gradient of the pressure
    std::vector<T> du, du_m1;                    // variation of the energy
    std::vector<T> dt, dt_m1;
    std::vector<T> c11, c12, c13, c22, c23, c33; // IAD components

    T ttot, etot, ecin, eint;
    T minDt;

    BBox<T> bbox;

    std::vector<std::vector<T> *> data{&x,    &y,     &z,  &x_m1,  &y_m1, &z_m1, &vx,  &vy,       &vz,       &ro,
                                       &ro_0, &u,     &p,  &p_0,   &h,    &m,    &c,   &grad_P_x, &grad_P_y, &grad_P_z,
                                       &du,   &du_m1, &dt, &dt_m1, &c11,  &c12,  &c13, &c22,      &c23,      &c33};
#ifdef USE_MPI
    MPI_Comm comm;
    int pnamelen = 0;
    char pname[MPI_MAX_PROCESSOR_NAME];
#endif

    int rank = 0;
    int nrank = 1;

    constexpr static T sincIndex = 6.0;
    constexpr static T Kcour = 0.2;
    constexpr static T maxDtIncrease = 1.1;
    const static T K;
    static T dx;

    // settings
    constexpr static ushort noOfGpuLoopSplits = 4; // No. of loop splits running in GPU to fit into the GPU memory

    //CUDA
    T *d_x, *d_y, *d_z, *d_m, *d_h;
    T *d_ro;
    BBox<T> *d_bbox;

};

template <typename T>
T ParticlesData<T>::dx = 0.01;

template <typename T>
const T ParticlesData<T>::K = sphexa::compute_3d_k(sincIndex);

} // namespace sphexa
