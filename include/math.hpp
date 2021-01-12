#pragma once

#ifdef USE_ACC
#include <accelmath.h>
#else
#include <cmath>
#endif

#include <cstdio>
#include <array>

#include "cudaFunctionAnnotation.hpp"

#define PI 3.14159265358979323846

#ifdef USE_STD_MATH_IN_KERNELS
#define math_namespace std
#else
#define math_namespace ::sphexa::math
#endif

namespace sphexa
{
namespace math
{
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
} // namespace math
} // namespace sphexa
