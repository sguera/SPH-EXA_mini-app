#pragma once

#include <vector>

namespace sphexa
{
struct Task
{
    Task(size_t size = 0) : clist(size), neighbors(size * ngmax), neighborsCount(size) {}

    void resize(size_t size)
    {
    	clist.resize(size);
    	neighbors.resize(size * ngmax);
    	neighborsCount.resize(size);
    }

    std::vector<int> clist;
    std::vector<int> neighbors;
    std::vector<int> neighborsCount;

    constexpr static size_t ngmax = 300;
    constexpr static size_t ng0 = 250;
};
} // namespace sphexa
