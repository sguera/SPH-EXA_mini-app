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

template <typename T>
struct ParticlesDataEvrard
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

    std::vector<T> fx, fy, fz, ugrav; // Gravity
    std::vector<T> cv;                // Specific heat
    std::vector<T> temp;              // Temperature
    std::vector<T> mue;               // Mean molecular weigh of electrons
    std::vector<T> mui;               // Mean molecular weight of ions

    T ttot = 0.0, etot, ecin, eint, egrav = 0.0;
    T minDt, minDmy = 1e-4, minTmpDt;

    sphexa::BBox<T> bbox;

    std::vector<std::vector<T> *> data{&x,   &y,   &z,   &x_m1, &y_m1,       &z_m1,     &vx,       &vy, &vz,    &ro, &ro_0,  &u,   &p,
                                       &p_0, &h,   &m,   &c,    &grad_P_x,   &grad_P_y, &grad_P_z, &du, &du_m1, &dt, &dt_m1, &c11, &c12,
                                       &c13, &c22, &c23, &c33,  &maxvsignal, &fx,       &fy,       &fz, &ugrav, &cv, &temp,  &mue, &mui};
#ifdef USE_MPI
    MPI_Comm comm;
    int pnamelen = 0;
    char pname[MPI_MAX_PROCESSOR_NAME];
#endif

    int rank = 0;
    int nrank = 1;

    constexpr static T g = 1.0; // for Evrard Collapse Gravity.
    // constexpr static T g = 6.6726e-8; // the REAL value of g. g is 1.0 for Evrard mainly

    constexpr static T sincIndex = 5.0;
    constexpr static T Kcour = 0.2;
    constexpr static T maxDtIncrease = 1.1;
    constexpr static size_t ngmin = 5, ng0 = 100, ngmax = 150;
    const static T K;
};

template <typename T>
const T ParticlesDataEvrard<T>::K = sphexa::compute_3d_k(sincIndex);

} // namespace sphexa
