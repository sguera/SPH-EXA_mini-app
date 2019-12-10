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

    size_t iteration;                            // Current iteration
    size_t n, side, count;                       // Number of particles
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
    std::vector<T> maxvsignal;

    T ttot, etot, ecin, eint;
    T minDt;

    BBox<T> bbox;

    std::vector<std::vector<T> *> data{&x,  &y,     &z,   &x_m1, &y_m1, &z_m1, &vx,       &vy,       &vz,        &ro, &ro_0,
                                       &u,  &p,     &p_0, &h,    &m,    &c,    &grad_P_x, &grad_P_y, &grad_P_z,  &du, &du_m1,
                                       &dt, &dt_m1, &c11, &c12,  &c13,  &c22,  &c23,      &c33,      &maxvsignal};
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
};

template <typename T>
const T ParticlesData<T>::K = sphexa::compute_3d_k(sincIndex);

} // namespace sphexa
