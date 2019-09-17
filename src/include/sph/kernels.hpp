#pragma once

#include "math.hpp"

#ifdef USE_STD_MATH_IN_KERNELS
#define math_namespace std
#else
#define math_namespace ::sphexa::math
#endif

namespace sphexa
{

template <typename T>
CUDA_DEVICE_HOST_FUN inline T wharmonic_std(T v)
{
    if (v == 0.0) return 1.0;

    const T Pv = (PI / 2.0) * v;

    return std::sin(Pv) / Pv;
}

template <typename T>
CUDA_DEVICE_HOST_FUN inline T wharmonic_derivative_std(T v)
{
    if (v == 0.0) return 0.0;

    const T Pv = (PI / 2.0) * v;
    const T sincv = std::sin(Pv) / (Pv);

    return sincv * (PI / 2.0) * ((std::cos(Pv) / std::sin(Pv)) - 1.0 / Pv);
}

template <typename T>
CUDA_DEVICE_FUN inline T wharmonic_lt(const T v)
{
    namespace lt = sphexa::lookup_tables;

    const size_t idx = (v * lt::wharmonicLookupTableSize / 2.0);

    return (idx >= lt::wharmonicLookupTableSize) ? 0.0 : lt::wharmonicLookupTable[idx];
}

template <typename T>
CUDA_DEVICE_FUN inline T wharmonic_lt_with_derivative(const T v)
{
    namespace lt = sphexa::lookup_tables;

    const size_t halfTableSize = lt::wharmonicLookupTableSize / 2.0;
    const size_t idx = v * halfTableSize;

    return (idx >= lt::wharmonicLookupTableSize)
               ? 0.0
               : lt::wharmonicLookupTable[idx] + lt::wharmonicDerivativeLookupTable[idx] * (v - (T)idx / halfTableSize);
}

template <typename T>
CUDA_DEVICE_FUN inline T wharmonic_derivative_lt(const T v)
{
    namespace lt = sphexa::lookup_tables;

    const size_t idx = (v * lt::wharmonicLookupTableSize / 2.0);

    return (idx >= lt::wharmonicLookupTableSize) ? -0.5 : lt::wharmonicDerivativeLookupTable[idx];
}

template <typename T>
CUDA_DEVICE_HOST_FUN inline T wharmonic_derivative_deprecated(T v, T h, T sincIndex, T K)
{
    T Pv = (PI / 2.0) * v;
    T cotv = math_namespace::cos(Pv) / math_namespace::sin(Pv);
    ; // 1.0 / tan(P * v);
    T sincv = math_namespace::sin(Pv) / (Pv);
    T sincnv = math_namespace::pow(sincv, (int)sincIndex);
    T ret = sincIndex * (Pv * cotv - 1.0) * sincnv * (K / (h * h * h * h * h * v * v));
    // printf("wharmonic_derivative called with v=%f, cotv=%f, sincIndex=%f, ret=%f\n", v, cotv, sincIndex, ret);
    return ret;
}

#ifdef USE_STD_MATH_IN_KERNELS
constexpr auto wharmonic = wharmonic_std<double>;
constexpr auto wharmonic_derivative = wharmonic_derivative_std<double>;
#else
// constexpr auto wharmonic = wharmonic_lt<double>;
constexpr auto wharmonic = wharmonic_lt_with_derivative<double>;
constexpr auto wharmonic_derivative = wharmonic_derivative_lt<double>;
#endif

template <typename T>
CUDA_DEVICE_FUN inline T artificial_viscosity(const T ro_i, const T ro_j, const T h_i, const T h_j, const T c_i, const T c_j, const T rv,
                                          const T r_square)
{
    const T alpha = 1.0;
    const T beta = 2.0;
    const T epsilon = 0.01;

    const T ro_ij = (ro_i + ro_j) / 2.0;
    const T c_ij = (c_i + c_j) / 2.0;
    const T h_ij = (h_i + h_j) / 2.0;

    // calculate viscosity_ij according to Monaghan & Gringold 1983
    T viscosity_ij = 0.0;
    if (rv < 0.0)
    {
        // calculate muij
        const T mu_ij = (h_ij * rv) / (r_square + epsilon * h_ij * h_ij);
        viscosity_ij = (-alpha * c_ij * mu_ij + beta * mu_ij * mu_ij) / ro_ij;
    }

    return viscosity_ij;
}

} // namespace sphexa
