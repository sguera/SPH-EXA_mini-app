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
    const T *ro_0 = d.ro_0.data();
    const T *p_0 = d.p_0.data();
    T *p = d.p.data();
    T *c = d.c.data();
    T *u = d.u.data();

    const T heatCapacityRatio = 7.0;
    const T speedOfSound0 = 3500.0;
    const T density0 = 1.0;

    // (ro_0 / 7.0) * c^2
    // const T chi = (1000.0 / 7.0) * (35.0 * 35.0);
    const T chi = (density0 / heatCapacityRatio) * (speedOfSound0 * speedOfSound0);

#pragma omp parallel for
    for (size_t pi = 0; pi < n; pi++)
    {
        const int i = clist[pi];

        p[i] = chi * (pow(ro[i] / ro_0[i], heatCapacityRatio) - 1.0) + p_0[i];
        c[i] = speedOfSound0; // * sqrt(pow(ro[i]/ro_0[i], heatCapacityRatio-1.));
        u[i] = 1.0;           //
        // u[i] = 1e-10;
        // 1e7 per unit of mass (1e-3 or 1g)
    }
}

template <typename T, class Dataset>
void computeEquationOfState(const std::vector<Task> &taskList, Dataset &d)
{
#ifdef USE_HPX
    auto policy = hpx::parallel::execution::par;
    hpx::parallel::for_loop(policy, 0, taskList.size(),
        [&taskList, &d](size_t i)
        {
            computeEquationOfStateImpl<T>(taskList[i], d);
        }
    );
#else
    for (const auto &task : taskList)
    {
        computeEquationOfStateImpl<T>(task, d);
    }
#endif
}

} // namespace sph
} // namespace sphexa
