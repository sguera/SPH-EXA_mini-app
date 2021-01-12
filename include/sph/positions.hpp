#pragma once

#include <vector>
#include <cmath>
#include <tuple>

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
struct computeAccelerationWithGravity
{
    std::tuple<T, T, T> operator()(const int idx, Dataset &d)
    {
        const T G = d.g;
        const T ax = -(d.grad_P_x[idx] - G * d.fx[idx]);
        const T ay = -(d.grad_P_y[idx] - G * d.fy[idx]);
        const T az = -(d.grad_P_z[idx] - G * d.fz[idx]);
        return std::make_tuple(ax, ay, az);
    }
};

template <typename T, class Dataset>
struct computeAcceleration
{
    std::tuple<T, T, T> operator()(const int idx, Dataset &d)
    {
        return std::make_tuple(-d.grad_P_x[idx], -d.grad_P_y[idx], -d.grad_P_z[idx]);
    }
};

template <typename T, class FunctAccel, class Dataset>
void computePositionsImpl(const Task &t, Dataset &d)
{
    FunctAccel accelFunct;

    const size_t n = t.clist.size();
    const int *clist = t.clist.data();

    const T *dt = d.dt.data();
    const T *du = d.du.data();

    T *x = d.x.data();
    T *y = d.y.data();
    T *z = d.z.data();
    T *vx = d.vx.data();
    T *vy = d.vy.data();
    T *vz = d.vz.data();
    T *x_m1 = d.x_m1.data();
    T *y_m1 = d.y_m1.data();
    T *z_m1 = d.z_m1.data();
    T *u = d.u.data();
    T *du_m1 = d.du_m1.data();
    T *dt_m1 = d.dt_m1.data();

    const BBox<T> &bbox = d.bbox;

#pragma omp parallel for
    for (size_t pi = 0; pi < n; pi++)
    {
        const int i = clist[pi];
        T ax, ay, az;
        std::tie(ax, ay, az) = accelFunct(i, d);

#ifndef NDEBUG
        if (std::isnan(ax) || std::isnan(ay) || std::isnan(az))
            printf("ERROR::UpdateQuantities(%d) acceleration: (%f %f %f)\n", i, ax, ay, az);
#endif

        // Update positions according to Press (2nd order)
        T deltaA = dt[i] + 0.5 * dt_m1[i];
        T deltaB = 0.5 * (dt[i] + dt_m1[i]);

        T valx = (x[i] - x_m1[i]) / dt_m1[i];
        T valy = (y[i] - y_m1[i]) / dt_m1[i];
        T valz = (z[i] - z_m1[i]) / dt_m1[i];

        vx[i] = valx + ax * deltaA;
        vy[i] = valy + ay * deltaA;
        vz[i] = valz + az * deltaA;

        x_m1[i] = x[i];
        y_m1[i] = y[i];
        z_m1[i] = z[i];

        // x[i] = x + dt[i] * valx + ax * dt[i] * deltaB;
        x[i] += dt[i] * valx + (vx[i] - valx) * dt[i] * deltaB / deltaA;
        y[i] += dt[i] * valy + (vy[i] - valy) * dt[i] * deltaB / deltaA;
        z[i] += dt[i] * valz + (vz[i] - valz) * dt[i] * deltaB / deltaA;

        const T xw = (bbox.xmax - bbox.xmin);
        const T yw = (bbox.ymax - bbox.ymin);
        const T zw = (bbox.zmax - bbox.zmin);

        if (bbox.PBCx && x[i] < bbox.xmin)
        {
            x[i] += xw;
            x_m1[i] += xw;
        }
        else if (bbox.PBCx && x[i] > bbox.xmax)
        {
            x[i] -= xw;
            x_m1[i] -= xw;
        }
        if (bbox.PBCy && y[i] < bbox.ymin)
        {
            y[i] += yw;
            y_m1[i] += yw;
        }
        else if (bbox.PBCy && y[i] > bbox.ymax)
        {
            y[i] -= yw;
            y_m1[i] -= yw;
        }
        if (bbox.PBCz && z[i] < bbox.zmin)
        {
            z[i] += zw;
            z_m1[i] += zw;
        }
        else if (bbox.PBCz && z[i] > bbox.zmax)
        {
            z[i] -= zw;
            z_m1[i] -= zw;
        }

        // Update the energy according to Adams-Bashforth (2nd order)
        deltaA = 0.5 * dt[i] * dt[i] / dt_m1[i];
        deltaB = dt[i] + deltaA;

        u[i] += du[i] * deltaB - du_m1[i] * deltaA;

#ifndef NDEBUG
        if (std::isnan(u[i]))
            printf("ERROR::UpdateQuantities(%d) internal energy: u %f du %f dB %f du_m1 %f dA %f\n", i, u[i], du[i], deltaB, du_m1[i],
                   deltaA);
#endif

        du_m1[i] = du[i];
        dt_m1[i] = dt[i];
    }
}

template <typename T, class FunctAccel, class Dataset>
void computePositions(const std::vector<Task> &taskList, Dataset &d)
{
    for (const auto &task : taskList)
    {
        computePositionsImpl<T, FunctAccel>(task, d);
    }
}

} // namespace sph
} // namespace sphexa
