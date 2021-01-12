#pragma once

#include "../math.hpp"
#include "../cudaFunctionAnnotation.hpp"
#include "kernels.hpp"

#ifdef __NVCC__
#include "cuda/cudaUtils.cuh"
#endif

namespace sphexa
{

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

constexpr size_t wharmonicLookupTableSize = 20000;

#if defined(__NVCC__)
extern __device__ double wharmonicLookupTable[wharmonicLookupTableSize];
extern __device__ double wharmonicDerivativeLookupTable[wharmonicLookupTableSize];
#else
static auto wh = createWharmonicLookupTable<double, wharmonicLookupTableSize>();
static auto whd = createWharmonicDerivativeLookupTable<double, wharmonicLookupTableSize>();
static const double *wharmonicLookupTable = wh.data();
static const double *wharmonicDerivativeLookupTable = whd.data();
#endif

#ifndef USE_STD_MATH_IN_KERNELS
template <typename T>
struct GpuLookupTableInitializer
{
    GpuLookupTableInitializer()
    {
#if defined(USE_OMP_TARGET)
#pragma omp target enter data map(to : wharmonicLookupTable [0:wharmonicLookupTableSize])
        ; // Empty instruction is minimum required for pgi
#pragma omp target enter data map(to : wharmonicDerivativeLookupTable [0:wharmonicLookupTableSize])
        ; // Empty instruction is minimum required for pgi
#elif defined(__NVCC__)
        namespace cuda = sphexa::sph::cuda;

        // cuda::CHECK_CUDA_ERR(cuda::utils::cudaMalloc(wharmonicLookupTableSize * sizeof(T), wharmonicLookupTable,
        // wharmonicDerivativeLookupTable));
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
#pragma omp target exit data map(delete : wharmonicLookupTable [0:wharmonicLookupTableSize])
        ; // Empty instruction is minimum required for pgi
#pragma omp target exit data map(delete : wharmonicDerivativeLookupTable [0:wharmonicLookupTableSize])
        ; // Empty instruction is minimum required for pgi
          // #elif defined(__NVCC__)
          //         namespace cuda = sphexa::sph::cuda;
          //         cuda::CHECK_CUDA_ERR(cuda::utils::cudaFree(wharmonicLookupTable, wharmonicDerivativeLookupTable));
#endif
    }
};

extern GpuLookupTableInitializer<double> ltinit;

#ifndef USE_CUDA

GpuLookupTableInitializer<double> ltinit;
#endif

#endif

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

} // namespace lookup_tables

#ifdef USE_STD_MATH_IN_KERNELS
constexpr auto wharmonic = wharmonic_std<double>;
constexpr auto wharmonic_derivative = wharmonic_derivative_std<double>;
#else
constexpr auto wharmonic = lookup_tables::wharmonic_lt_with_derivative<double>;
constexpr auto wharmonic_derivative = lookup_tables::wharmonic_derivative_lt<double>;
#endif

} // namespace sphexa
