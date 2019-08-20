#include <cuda.h>

namespace sphexa
{
namespace sph
{
namespace cuda
{
#define CHECK_CUDA_ERR(errcode) checkErr((errcode), __FILE__, __LINE__, #errcode);

inline void checkErr(cudaError_t err, const char *filename, int lineno, const char *funcName)
{
    if (err != cudaSuccess)
    {
        const char *errName = cudaGetErrorName(err);
        const char *errStr = cudaGetErrorString(err);
        fprintf(stderr, "CUDA Error at %s:%d. Function %s returned err %d: %s - %s\n", filename, lineno, funcName, err, errName, errStr);
    }
}
}
} // namespace sph
} // namespace sphexa
