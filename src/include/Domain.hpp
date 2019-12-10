#pragma once

#include <vector>
#include <cmath>
#include <numeric>

#include "Octree.hpp"
#include "Task.hpp"

namespace sphexa
{

// Helper functions
template <typename T>
inline T normalize(T d, T min, T max)
{
    return (d - min) / (max - min);
}

template <typename T>
void reorderSwap(const std::vector<int> &ordering, std::vector<T> &arrayList)
{
    std::vector<T> tmp(ordering.size());
    for (unsigned int i = 0; i < ordering.size(); i++)
        tmp[i] = arrayList[ordering[i]];
    tmp.swap(arrayList);
}

template <typename T>
void reorder(const std::vector<int> &ordering, std::vector<std::vector<T> *> &arrayList)
{
    for (unsigned int i = 0; i < arrayList.size(); i++)
        reorderSwap(ordering, *arrayList[i]);
}

template <class Dataset>
void reorder(const std::vector<int> &ordering, Dataset &d)
{
    reorder(ordering, d.data);
}

template <typename T>
void makeDataArray(std::vector<std::vector<T> *> &data, std::vector<T> *d)
{
    data.push_back(d);
}

template <typename T, typename... Args>
void makeDataArray(std::vector<std::vector<T> *> &data, std::vector<T> *first, Args... args)
{
    data.push_back(first);
    makeDataArray(data, args...);
}

template <typename T, class Dataset>
class Domain
{
public:
    Domain() = default;
    virtual ~Domain() = default;

    virtual void create(Dataset &d)
    {
        const std::vector<T> &x = d.x;
        const std::vector<T> &y = d.y;
        const std::vector<T> &z = d.z;

        const size_t n = d.count;

        clist.resize(n);
        for (size_t i = 0; i < n; i++)
            clist[i] = i;

        std::vector<int> ordering(n);

        d.bbox.computeGlobal(clist, d.x, d.y, d.z);

        // Each process creates a tree based on the gathered sample
        octree.cells.clear();
        octree = Octree<T>(d.bbox.xmin, d.bbox.xmax, d.bbox.ymin, d.bbox.ymax, d.bbox.zmin, d.bbox.zmax, 0, 1);
        octree.buildTree(clist, x, y, z, ordering);
        reorder(ordering, d);
    }

    virtual void update(Dataset &d)
    {
        d.bbox.computeGlobal(clist, d.x, d.y, d.z);
        octree.cells.clear();
        octree.xmin = d.bbox.xmin;
        octree.xmax = d.bbox.xmax;
        octree.ymin = d.bbox.ymin;
        octree.ymax = d.bbox.ymax;
        octree.zmin = d.bbox.zmin;
        octree.zmax = d.bbox.zmax;
        return;
    }

    void buildTree(Dataset &d)
    {
        // Finally remap everything
        std::vector<int> ordering(d.x.size());

        std::vector<int> list(d.x.size());
#pragma omp parallel for
        for (int i = 0; i < (int)d.x.size(); i++)
            list[i] = i;

        // We need this to expand halo
        octree.buildTree(list, d.x, d.y, d.z, ordering);
        reorder(ordering, d);

        octree.mapList(clist);
    }

    void createTasks(std::vector<Task> &taskList, const size_t nTasks)
    {
        const int partitionSize = clist.size() / nTasks;
        const int lastPartitionOffset = clist.size() - nTasks * partitionSize;

        taskList.resize(nTasks);

#pragma omp parallel for
        for (size_t i = 0; i < nTasks; ++i)
        {
            const int begin = i * partitionSize;
            const int end = (i + 1) * partitionSize + (i == nTasks - 1 ? lastPartitionOffset : 0);
            const size_t size = end - begin;

            taskList[i].resize(size);
            for (size_t j = 0; j < size; j++)
                taskList[i].clist[j] = clist[j + begin];
        }
    }

    // placeholder for the non-distributed domain implementation
    template <typename... Args>
    void synchronizeHalos(Args...)
    {
    }

    void synchronizeHalos(std::vector<std::vector<T> *> &) {}

    Octree<T> octree;
    std::vector<int> clist;
};

} // namespace sphexa
