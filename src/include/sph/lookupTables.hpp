#pragma once

#include "../math.hpp"
#include "sphUtils.hpp"
#include "../cudaFunctionAnnotation.hpp"

#ifdef __NVCC__
#include "cuda/utils.cuh"
#endif

namespace sphexa
{

template <typename T>
CUDA_DEVICE_HOST_FUN inline T wharmonic_std(T v);

template <typename T>
CUDA_DEVICE_HOST_FUN inline T wharmonic_derivative_std(T v);

namespace lookup_tables
{
template <typename T, std::size_t N>
std::array<T, N> createWharmonicLookupTable()
{
    std::array<T, N> lt;

    const T halfsSize = N / 2.0;
    for (size_t i = 0; i < N; ++i)
    {
        T normalizedVal = i / halfsSize;
        lt[i] = wharmonic_std(normalizedVal);
    }
    return lt;
}

template <typename T, std::size_t N>
std::array<T, N> createWharmonicDerivativeLookupTable()
{
    std::array<T, N> lt;

    const T halfsSize = N / 2.0;
    for (size_t i = 0; i < N; ++i)
    {
        T normalizedVal = i / halfsSize;
        lt[i] = wharmonic_derivative_std(normalizedVal);
    }

    return lt;
}

template <typename T, std::size_t N>
std::array<T, N> createCosSinLookupTable()
{
    std::array<T, N> lt;

    const auto halfSize = N / 2;
    for (size_t i = 0; i < N; ++i)
        lt[i] = (T)std::sin((T)i * PI / halfSize);

    return lt;
}

constexpr int sinCosLTMaxCircleAngle = 131072;
constexpr int sinCosLTHalfMaxCircleAngle = sinCosLTMaxCircleAngle / 2;
constexpr int sinCosLTQuarterMaxCircleAngle = sinCosLTMaxCircleAngle / 4;
constexpr int sinCosLTMaskMaxCircleAngle = sinCosLTMaxCircleAngle - 1;
constexpr int sinCosLTSize = sinCosLTMaxCircleAngle;

constexpr size_t wharmonicLookupTableSize = 20000;

#if defined(__NVCC__)
__device__ extern double wharmonicLookupTable[wharmonicLookupTableSize];
__device__ extern double wharmonicDerivativeLookupTable[wharmonicLookupTableSize];
__device__ extern double fast_cossin_table[sinCosLTSize];
#else
static auto wharmonicLookupTable = createWharmonicLookupTable<double, wharmonicLookupTableSize>();
static auto wharmonicDerivativeLookupTable = createWharmonicDerivativeLookupTable<double, wharmonicLookupTableSize>();
static auto fast_cossin_table = createCosSinLookupTable<double, sinCosLTSize>();
#endif

template <typename T>
struct GpuLookupTableInitializer
{
    GpuLookupTableInitializer()
    {
#if defined(USE_OMP_TARGET)
#pragma omp target enter data map(to : fast_cossin_table [0:sinCosLTSize])
#pragma omp target enter data map(to : wharmonicLookupTable [0:wharmonicLookupTableSize])
#pragma omp target enter data map(to : wharmonicDerivativeLookupTable [0:wharmonicLookupTableSize])
#elif defined(__NVCC__)
        namespace cuda = sphexa::sph::cuda;

        cuda::CHECK_CUDA_ERR(
            cudaMemcpyToSymbol(fast_cossin_table, createCosSinLookupTable<T, sinCosLTSize>().data(), sinCosLTSize * sizeof(T)));
        cuda::CHECK_CUDA_ERR(cudaMemcpyToSymbol(wharmonicLookupTable, createWharmonicLookupTable<T, wharmonicLookupTableSize>().data(),
                                                wharmonicLookupTableSize * sizeof(T)));
        cuda::CHECK_CUDA_ERR(cudaMemcpyToSymbol(wharmonicDerivativeLookupTable,
                                                createWharmonicDerivativeLookupTable<T, wharmonicLookupTableSize>().data(),
                                                wharmonicLookupTableSize * sizeof(T)));
#endif
    }

    ~GpuLookupTableInitializer()
    {
#if defined(USE_OMP_TARGET)
#pragma omp target exit data map(delete : fast_cossin_table [0:sinCosLTSize])
#pragma omp target exit data map(delete : wharmonicLookupTable [0:wharmonicLookupTableSize])
#pragma omp target exit data map(delete : wharmonicDerivativeLookupTable [0:wharmonicLookupTableSize])
#elif defined(__NVCC__)
        cudaFree(fast_cossin_table);
        cudaFree(wharmonicLookupTable);
        cudaFree(wharmonicDerivativeLookupTable);
#endif
    }
};

extern GpuLookupTableInitializer<double> ltinit;

} // namespace lookup_tables
} // namespace sphexa
