#pragma once

#include <cmath>
#include <cstdio>
#include <array>

#ifdef __NVCC__
#define CUDA_PREFIX __device__
#else
#define CUDA_PREFIX
#endif

#define PI 3.14159265358979323846

namespace sphexa
{
namespace math
{
/* Small powers, such as the ones used inside the SPH kernel
 * are transformed into straight multiplications. */
template <typename T>
CUDA_PREFIX inline T pow(T a, int b)
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

/* Fast lookup table implementation for sin and cos */
constexpr int MAX_CIRCLE_ANGLE = 131072;
constexpr int HALF_MAX_CIRCLE_ANGLE = MAX_CIRCLE_ANGLE / 2;
constexpr int QUARTER_MAX_CIRCLE_ANGLE = MAX_CIRCLE_ANGLE / 4;
constexpr int MASK_MAX_CIRCLE_ANGLE = MAX_CIRCLE_ANGLE - 1;

template <typename T, std::size_t N>
std::array<T, N> createCosSinLookupTable()
{
    std::array<T, N> lt;

    const auto halfSize = N / 2;
    for (size_t i = 0; i < N; ++i)
        lt[i] = (T)std::sin((T)i * PI / halfSize);

    return lt;
}

#if defined(__NVCC__)
// needs to be copied to gpu before any CUDA calculations
__device__ extern double fast_cossin_table[MAX_CIRCLE_ANGLE];
#else
static auto fast_cossin_table = createCosSinLookupTable<double, MAX_CIRCLE_ANGLE>();
#endif

template <typename T>
CUDA_PREFIX inline T cos(T n)
{
    const T f = n * HALF_MAX_CIRCLE_ANGLE / PI;
    const int i = static_cast<int>(f);

    return i < 0 ? fast_cossin_table[((-i) + QUARTER_MAX_CIRCLE_ANGLE) & MASK_MAX_CIRCLE_ANGLE]
                 : fast_cossin_table[(i + QUARTER_MAX_CIRCLE_ANGLE) & MASK_MAX_CIRCLE_ANGLE];
}

template <typename T>
CUDA_PREFIX inline T sin(T n)
{
    const T f = n * HALF_MAX_CIRCLE_ANGLE / PI;
    const int i = static_cast<int>(f);

    return i < 0 ? fast_cossin_table[(-((-i) & MASK_MAX_CIRCLE_ANGLE)) + MAX_CIRCLE_ANGLE] : fast_cossin_table[i & MASK_MAX_CIRCLE_ANGLE];
}

template <typename T>
struct GpuCosSinLookupTableInitializer
{
    GpuCosSinLookupTableInitializer()
    {
#if defined(USE_OMP_TARGET)
#pragma omp target enter data map(to : fast_cossin_table [0:MAX_CIRCLE_ANGLE])
#elif defined(__NVCC__)
        cudaMemcpyToSymbol(fast_cossin_table, createCosSinLookupTable<double, MAX_CIRCLE_ANGLE>().data(), MAX_CIRCLE_ANGLE * sizeof(double));
#endif
    }

    ~GpuCosSinLookupTableInitializer()
    {
#if defined(USE_OMP_TARGET)
#pragma omp target exit data map(delete : fast_cossin_table [0:MAX_CIRCLE_ANGLE])
#elif defined(__NVCC__)
        cudaFree(fast_cossin_table);
#endif
    }
};
extern math::GpuCosSinLookupTableInitializer<double> ltinit;

} // namespace math
} // namespace sphexa
