#pragma once

#include <vector>

#include "Task.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template <typename T, class Dataset>
extern void computeDensity(const std::vector<Task> &taskList, Dataset &d);

template <typename T, class Dataset>
extern void computeDensity(const Task &task, Dataset &d);

template <typename T, class Dataset>
extern void computeDensity(TaskQueue &tq, Dataset &d);

template <typename T, class Dataset>
extern void computeDensity(std::vector<Task>::iterator tbegin, std::vector<Task>::iterator tend, Dataset &d);

template <typename T, class Dataset>
extern void computeMomentumAndEnergy(const std::vector<Task> &taskList, Dataset &d);

template <typename T, class Dataset>
extern void computeIAD(const std::vector<Task> &taskList, Dataset &d);

template <typename T, class Dataset>
extern void computeMomentumAndEnergyIAD(const std::vector<Task> &taskList, Dataset &d);

template <typename T, class Dataset>
extern void copyInDensity(Dataset &d);

template <typename T, class Dataset>
extern void copyOutDensity(Dataset &d);


template <class T>
struct CudaAllocator
{
    typedef T value_type;
    CudaAllocator();
    // template <class U>
    // constexpr CudaAllocator(const CudaAllocator &);// noexcept {}[[nodiscard]] T *allocate(std::size_t n);
    [[nodiscard]] T* allocate(std::size_t n);
    void deallocate(T *p, std::size_t) noexcept;
};

template <class T, class U>
bool operator==(const CudaAllocator<T> &, const CudaAllocator<U> &)
{
    return true;
}
template <class T, class U>
bool operator!=(const CudaAllocator<T> &, const CudaAllocator<U> &)
{
    return false;
}

template <class T>
struct Mallocator {
  typedef T value_type;
  Mallocator() = default;
  template <class U> constexpr Mallocator(const Mallocator<U>&) noexcept {}
  [[nodiscard]] T* allocate(std::size_t n) {
    if(n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
    if(auto p = static_cast<T*>(std::malloc(n*sizeof(T)))) return p;
    throw std::bad_alloc();
  }
  void deallocate(T* p, std::size_t) noexcept { std::free(p); }
};
template <class T, class U>
bool operator==(const Mallocator<T>&, const Mallocator<U>&) { return true; }
template <class T, class U>
bool operator!=(const Mallocator<T>&, const Mallocator<U>&) { return false; }


} // namespace cuda
} // namespace sph
} // namespace sphexa
