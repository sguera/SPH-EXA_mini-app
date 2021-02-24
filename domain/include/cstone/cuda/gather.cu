/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! \file
 * \brief  Exposes gather functionality to reorder arrays by a map
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "errorcheck.cuh"
#include "gather.cuh"

namespace cstone
{

template<class T, class LocalIndex>
class DeviceMemory
{
    static constexpr int alignment = 4096/sizeof(T);
public:

    DeviceMemory() = default;

    ~DeviceMemory()
    {
        if (allocatedSize_ > 0)
        {
            checkCudaErrors(cudaFree(d_ordering_));
            checkCudaErrors(cudaFree(d_buffer_));
        }
    }

    void reallocate(std::size_t newSize)
    {
        if (newSize > allocatedSize_)
        {
            // allocate 5% extra to avoid reallocation on small increase
            newSize = double(newSize) * 1.05;
            // round up newSize to next 4K boundary
            newSize += newSize%alignment;

            if (allocatedSize_ > 0)
            {
                checkCudaErrors(cudaFree(d_ordering_));

                checkCudaErrors(cudaFree(d_buffer_));
            }

            checkCudaErrors(cudaMalloc((void**)&d_ordering_,  newSize * sizeof(LocalIndex)));
            checkCudaErrors(cudaMalloc((void**)&(d_buffer_), 2 * newSize * sizeof(T)));

            allocatedSize_ = newSize;
        }
    }

    LocalIndex* ordering() { return d_ordering_; }

    T* deviceBuffer(int i)
    {
        if (i > 1) throw std::runtime_error("buffer index out of bounds\n");
        return d_buffer_ + i * allocatedSize_;
    }

private:
    std::size_t allocatedSize_{0} ;

    //! \brief reorder map
    LocalIndex* d_ordering_;

    //! \brief device buffers
    T* d_buffer_;
};


template<class ValueType, class CodeType, class IndexType>
DeviceGather<ValueType, CodeType, IndexType>::DeviceGather()
    : deviceMemory_(std::make_unique<DeviceMemory<ValueType, IndexType>>())
{}

template<class ValueType, class CodeType, class IndexType>
void DeviceGather<ValueType, CodeType, IndexType>::setReorderMap(const IndexType* map_first,
                                                                 const IndexType* map_last)
{
    mapSize_      = map_last - map_first;
    deviceMemory_->reallocate(mapSize_);
    // upload new ordering to the device
    cudaMemcpy(deviceMemory_->ordering(), map_first, mapSize_ * sizeof(IndexType), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());
}

template<class ValueType, class CodeType, class IndexType>
void DeviceGather<ValueType, CodeType, IndexType>::getReorderMap(IndexType* map_first)
{
    cudaMemcpy(map_first, deviceMemory_->ordering(), mapSize_ * sizeof(IndexType), cudaMemcpyDeviceToHost);
}

template<class I>
__global__ void iotaKernel(I* buffer, size_t n, size_t offset)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        buffer[tid] = offset + tid;
    }
}

template<class ValueType, class CodeType, class IndexType>
void DeviceGather<ValueType, CodeType, IndexType>::setMapFromCodes(CodeType* codes_first, CodeType* codes_last)
{
    mapSize_      = codes_last - codes_first;
    deviceMemory_->reallocate(mapSize_);

    // the deviceBuffer is allocated as a single chunk of size 2 * mapSize_ * sizeof(T)
    // so we can reuse it for mapSize_ elements of I, as long as the static assert holds
    static_assert(sizeof(CodeType) <= 2 * sizeof(ValueType), "buffer size not big enough for codes device array\n");
    CodeType* d_codes = reinterpret_cast<CodeType*>(deviceMemory_->deviceBuffer(0));

    // send Morton codes to the device
    cudaMemcpy(d_codes, codes_first, mapSize_ * sizeof(CodeType), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());

    constexpr int nThreads = 256;
    int nBlocks = (mapSize_ + nThreads - 1) / nThreads;
    iotaKernel<<<nBlocks, nThreads>>>(deviceMemory_->ordering(), mapSize_, 0);
    checkCudaErrors(cudaGetLastError());

    // sort Morton codes on device as keys, track new ordering on the device
    thrust::sort_by_key(thrust::device,
                        thrust::device_pointer_cast(d_codes),
                        thrust::device_pointer_cast(d_codes+mapSize_),
                        thrust::device_pointer_cast(deviceMemory_->ordering()));
    checkCudaErrors(cudaGetLastError());

    // send sorted codes back to host
    cudaMemcpy(codes_first, d_codes, mapSize_ * sizeof(CodeType), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

template<class ValueType, class CodeType, class IndexType>
DeviceGather<ValueType, CodeType, IndexType>::~DeviceGather() = default;


template<class T, class I>
__global__ void reorder(I* map, T* source, T* destination, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        destination[tid] = source[map[tid]];
    }
}

template<class ValueType, class CodeType, class IndexType>
void DeviceGather<ValueType, CodeType, IndexType>::operator()(ValueType* values)
{
    constexpr int nThreads = 256;
    int nBlocks = (mapSize_ + nThreads - 1) / nThreads;

    // upload to device
    cudaMemcpy(deviceMemory_->deviceBuffer(0), values, mapSize_ * sizeof(ValueType), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());

    // reorder on device
    reorder<<<nBlocks, nThreads>>>(deviceMemory_->ordering(),
                                   deviceMemory_->deviceBuffer(0),
                                   deviceMemory_->deviceBuffer(1),
                                   mapSize_);
    checkCudaErrors(cudaGetLastError());

    // download to host
    cudaMemcpy(values, deviceMemory_->deviceBuffer(1), mapSize_ * sizeof(ValueType), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
}

template class DeviceGather<float,  unsigned, unsigned>;
template class DeviceGather<float,  uint64_t, unsigned>;
template class DeviceGather<double, unsigned, unsigned>;
template class DeviceGather<double, uint64_t, unsigned>;
template class DeviceGather<float,  unsigned, uint64_t>;
template class DeviceGather<float,  uint64_t, uint64_t>;
template class DeviceGather<double, unsigned, uint64_t>;
template class DeviceGather<double, uint64_t, uint64_t>;

} // namespace cstone