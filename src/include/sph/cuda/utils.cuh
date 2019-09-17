#pragma once

#include <cuda.h>
#include <type_traits>

namespace sphexa
{
namespace sph
{
namespace cuda
{
#define CHECK_CUDA_ERR(errcode) utils::checkErr((errcode), __FILE__, __LINE__, #errcode);

namespace utils
{

inline void checkErr(cudaError_t err, const char *filename, int lineno, const char *funcName)
{
    if (err != cudaSuccess)
    {
        const char *errName = cudaGetErrorName(err);
        const char *errStr = cudaGetErrorString(err);
        fprintf(stderr, "CUDA Error at %s:%d. Function %s returned err %d: %s - %s\n", filename, lineno, funcName, err, errName, errStr);
    }
}

inline cudaError_t cudaFree() { return cudaSuccess; }

template <typename Ptr, typename... Ptrs>
inline cudaError_t cudaFree(Ptr first, Ptrs... ptrs)
{
    static_assert(std::is_pointer<Ptr>::value, "Parameter passed to cudaFree must be a pointer type");

    const auto ret = ::cudaFree(first);
    if (ret == cudaSuccess) return cudaFree(ptrs...);

    return ret;
}

inline cudaError_t cudaMalloc(size_t bytes) { return cudaSuccess; }

template <typename Ptr, typename... Ptrs>
inline cudaError_t cudaMalloc(size_t bytes, Ptr &devptr, Ptrs &&... ptrs)
{
    static_assert(std::is_pointer<Ptr>::value, "Parameter passed to cudaMalloc must be a pointer type");

    const auto ret = ::cudaMalloc((void **)&devptr, bytes);
    if (ret == cudaSuccess) return cudaMalloc(bytes, ptrs...);

    return ret;
}

} // namespace utils
} // namespace cuda
} // namespace sph
} // namespace sphexa
