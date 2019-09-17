#pragma once

#ifdef __NVCC__
#define CUDA_DEVICE_HOST_FUN __device__ __host__
#define CUDA_DEVICE_FUN __device__
#else
#define CUDA_DEVICE_HOST_FUN
#define CUDA_DEVICE_FUN
#endif
