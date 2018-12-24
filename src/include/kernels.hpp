#pragma once

#include <cmath>

namespace sphexa
{

template<typename T>
inline T update_smoothing_length(const int ng0, const int ngi, const T hi)
{
    const T c0 = 7.0;
    const T exp = 1.0/3.0;

    T ka = pow((1.0 + c0 * ng0 / ngi), exp);

    return hi * 0.5 * ka;
}

template<typename T>
inline T compute_3d_k(T n)
{
    //b0, b1, b2 and b3 are defined in "SPHYNX: an accurate density-based SPH method for astrophysical applications", DOI: 10.1051/0004-6361/201630208
    T b0 = 2.7012593e-2;
    T b1 = 2.0410827e-2;
    T b2 = 3.7451957e-3;
    T b3 = 4.7013839e-2;

    return b0 + b1 * sqrt(n) + b2 * n + b3 * sqrt(n*n*n);
}

template<typename T>
inline T wharmonic(T v, T h, T K)
{
    T value = (PI/2.0) * v;
    return K/(h*h*h) * pow((sin(value)/value), 5);
}

template<typename T>
inline T wharmonic_derivative(T v, T h, T K)
{
    T value = (PI/2.0) * v;
    // r_ih = v * h
    // extra h at the bottom comes from the chain rule of the partial derivative
    T kernel = wharmonic(v, h, K);

    return 5.0 * (PI/2.0) * kernel / (h * h) / v * ((1.0 / tan(value)) - (1.0 / value));
}

template<typename T>
inline T artificial_viscosity(T ro_i, T ro_j, T h_i, T h_j, T c_i, T c_j, T rv, T r_square)
{
    T alpha = 1.0;
    T beta = 2.0;
    T epsilon = 0.01;

    T ro_ij = (ro_i + ro_j) / 2.0;
    T c_ij = (c_i + c_j) / 2.0;
    T h_ij = (h_i + h_j) / 2.0;


    //calculate viscosity_ij according to Monaghan & Gringold 1983
    T viscosity_ij = 0.0;
    if (rv < 0.0)
    {
        //calculate muij
        T mu_ij = (h_ij * rv) / (r_square + epsilon * h_ij * h_ij);
        viscosity_ij = (-alpha * c_ij * mu_ij + beta * mu_ij * mu_ij) / ro_ij;
    }

    return viscosity_ij;
}

}

