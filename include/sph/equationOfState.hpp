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
    for (const auto &task : taskList)
    {
        computeEquationOfStateImpl<T>(task, d);
    }
}

template <typename T, typename Dataset>
void computeEquationOfStateEvrardImpl(const Task &t, Dataset &d)
{
    const size_t n = t.clist.size();
    const int *clist = t.clist.data();

    const T R = 8.317e7, gamma = (5.0 / 3.0);

    const T *ro = d.ro.data();
    const T *mui = d.mui.data();
    const T *u = d.u.data();

    T *temp = d.temp.data();
    T *p = d.p.data();
    T *c = d.c.data();
    T *cv = d.cv.data();

#pragma omp parallel for
    for (size_t pi = 0; pi < n; ++pi)
    {
        const int i = clist[pi];

        cv[i] = 1.5 * R / mui[i];
        temp[i] = u[i] / cv[i];
        T tmp = u[i] * (gamma - 1);
        p[i] = ro[i] * tmp;
        c[i] = sqrt(tmp);

        if (std::isnan(c[i]) || std::isnan(cv[i]))
            printf("ERROR:equation_of_state c %f cv %f temp %f u %f p %f\n", c[i], cv[i], temp[i], u[i], p[i]);
    }
}
template <typename T, class Dataset>
void computeEquationOfStateEvrard(const std::vector<Task> &taskList, Dataset &d)
{
    for (const auto &task : taskList)
    {
        computeEquationOfStateEvrardImpl<T>(task, d);
    }
}
} // namespace sph
} // namespace sphexa
