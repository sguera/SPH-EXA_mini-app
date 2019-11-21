#pragma once

#include <vector>

#include "kernels.hpp"

namespace sphexa
{
namespace sph
{
template <typename T, class Dataset>
void computeEquationOfStateImpl(const Task &t, Dataset &d)
{
    const size_t n = t.clist.size();
    const int *clist = t.clist.data();

    const T *ro = d.ro.data();

    T *p = d.p.data();
    T *c = d.c.data();
    T *u = d.u.data();

    const T gamma1 = 5.0/3.0 - 1.0;

#pragma omp parallel for
    for (size_t pi = 0; pi < n; pi++)
    {
        const int i = clist[pi];

        p[i] = u[i] * ro[i] * gamma1;
        c[i] = sqrt(gamma1 * u[i]);
    }
}

template <typename T, class Dataset>
void computeEquationOfState(const std::vector<Task> &taskList, Dataset &d)
{
    for (const auto &task : taskList)
    {
        computeEquationOfStateImpl<T>(task, d);
    }
}

} // namespace sph
} // namespace sphexa
