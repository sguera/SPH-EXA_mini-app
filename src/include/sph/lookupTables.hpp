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
template <typename T>
struct TestArr
{
  TestArr()
  {
#if defined (USE_ACC)
    #pragma acc enter data copyin(this)
    #pragma acc enter data copyin (arr[0:size])
#endif
  }
    T operator()(const int v) const
    {
      return arr[v];
    }
    const static size_t size = 10000;
    const T arr[size] = {0.5};
};

template <typename T>
struct Wharmonic
{
    Wharmonic()
        {
#if defined (USE_ACC)
    #pragma acc enter data copyin(this)
#endif
        }
    T operator()(const T v) const
    {
        // With PGI v18.5-0 lines below won't compile. Workaround: use unsigned or ints here
        const size_t halfTableSize = size / 2.0;
        const size_t idx = v * halfTableSize;

       return (idx >= size)
                     ? 0.0
         //   : ltt[idx];
         : arr[idx];// + wharmonicDerivativeLookupTable[idx] * (v - (T)idx / halfTableSize);
    }

    static const int size = 20000;
        //const T arr[size] = {0.5};
    const T* arr = createWharmonicLookupTable<T, size>().data();
    // const std::array<T, wharmonicLookupTableSize> wharmonicDerivativeLookupTable =
    //     createWharmonicDerivativeLookupTable<double, wharmonicLookupTableSize>();
};

constexpr size_t wharmonicLookupTableSize = 20000;

#if defined(__NVCC__)
__device__ extern double wharmonicLookupTable[wharmonicLookupTableSize];
__device__ extern double wharmonicDerivativeLookupTable[wharmonicLookupTableSize];
#else
// static auto wharmonicLookupTable = createWharmonicLookupTable<double, wharmonicLookupTableSize>();
// static auto wharmonicDerivativeLookupTable = createWharmonicDerivativeLookupTable<double, wharmonicLookupTableSize>();
#endif

template <typename T>
struct GpuLookupTableInitializer
{
    GpuLookupTableInitializer()
    {
#if defined(USE_OMP_TARGET)
#pragma omp target enter data map(to : wharmonicLookupTable [0:wharmonicLookupTableSize])
#pragma omp target enter data map(to : wharmonicDerivativeLookupTable [0:wharmonicLookupTableSize])
#elif defined(__NVCC__)
        namespace cuda = sphexa::sph::cuda;

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
#pragma omp target exit data map(delete : wharmonicDerivativeLookupTable [0:wharmonicLookupTableSize])
#elif defined(__NVCC__)
        cudaFree(wharmonicLookupTable);
        cudaFree(wharmonicDerivativeLookupTable);
#endif
    }
};

extern GpuLookupTableInitializer<double> ltinit;

} // namespace lookup_tables
} // namespace sphexa
