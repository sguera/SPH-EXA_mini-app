#pragma once

#include <cmath>
#include <cstdio>
#include <array>

#include "cudaFunctionAnnotation.hpp"

#define PI 3.14159265358979323846

#include "sph/lookupTables.hpp"

namespace sphexa
{
namespace math
{
/* Small powers, such as the ones used inside the SPH kernel
 * are transformed into straight multiplications. */
template <typename T>
CUDA_DEVICE_HOST_FUN inline T pow(T a, int b)
{
    if (b == 0)
        return 1;
    else if (b == 1)
        return a;
    else if (b == 2)
        return a * a;
    else if (b == 3)
        return a * a * a;
    else if (b == 4)
        return a * a * a * a;
    else if (b == 5)
        return a * a * a * a * a;
    else if (b == 6)
        return a * a * a * a * a * a;
    else if (b == 7)
        return a * a * a * a * a * a * a;
    else
        return std::pow(a, b);
}

template <typename T>
CUDA_DEVICE_FUN inline T cos(T n)
{
    namespace lt = sphexa::lookup_tables;

    const T f = n * lt::sinCosLTMaxCircleAngle / PI;
    const int i = static_cast<int>(f);

    return i < 0 ? lt::fast_cossin_table[((-i) + lt::sinCosLTQuarterMaxCircleAngle) & lt::sinCosLTMaskMaxCircleAngle]
                 : lt::fast_cossin_table[(i + lt::sinCosLTQuarterMaxCircleAngle) & lt::sinCosLTMaskMaxCircleAngle];
}

template <typename T>
CUDA_DEVICE_FUN inline T sin(T n)
{
    namespace lt = sphexa::lookup_tables;

    const T f = n * lt::sinCosLTHalfMaxCircleAngle / PI;
    const int i = static_cast<int>(f);

    return i < 0 ? lt::fast_cossin_table[(-((-i) & lt::sinCosLTMaskMaxCircleAngle)) + lt::sinCosLTMaxCircleAngle]
                 : lt::fast_cossin_table[i & lt::sinCosLTMaskMaxCircleAngle];
}

} // namespace math
} // namespace sphexa
