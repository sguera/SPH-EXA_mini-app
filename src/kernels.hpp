#pragma once

#include <cmath>

namespace sphexa
{

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
inline void eos(const T R, const T gamma, const T ro, const T u, const T mui, T &pressure, T &temperature, T &soundspeed, T &cv)
{
    cv = (gamma - 1) * R / mui;
    temperature = u / cv;
    T tmp = u * (gamma - 1);
    pressure = ro * tmp;
    soundspeed = sqrt(tmp);
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
    if (rv < 0.0){
        //calculate muij
        T mu_ij = (h_ij * rv) / (r_square + epsilon * h_ij * h_ij);
        viscosity_ij = (-alpha * c_ij * mu_ij + beta * mu_ij * mu_ij) / ro_ij;
    }

    return viscosity_ij;
}

}

