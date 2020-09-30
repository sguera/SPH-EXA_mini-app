#pragma once

#include <cmath>
#include <vector>

#include "BBox.hpp"
#include "sph/kernels.hpp"
#include "ParticlesData.hpp"

namespace sphexa
{
template <typename T>
class SedovDataGenerator
{
public:
    static ParticlesData<T> generate(const size_t side)
    {
        ParticlesData<T> pd;

#ifdef USE_MPI
        pd.comm = MPI_COMM_WORLD;
        MPI_Comm_size(pd.comm, &pd.nrank);
        MPI_Comm_rank(pd.comm, &pd.rank);
        MPI_Get_processor_name(pd.pname, &pd.pnamelen);
#endif

        pd.n = side * side * side;
        pd.side = side;
        pd.count = side * side * side;

        load(pd);
        init(pd);

        return pd;
    }

    // void load(const std::string &filename)
    static void load(ParticlesData<T> &pd)
    {
        size_t split = pd.n / pd.nrank;
        size_t remaining = pd.n - pd.nrank * split;

        pd.count = split;
        if (pd.rank == 0) pd.count += remaining;

        pd.resize(pd.count);

        if(pd.rank == 0)
            std::cout << "Approx: " << pd.count * (pd.data.size() * 64.0) / (8.0 * 1000.0 * 1000.0 * 1000.0) << "GB allocated on rank 0." << std::endl; 

        size_t offset = pd.rank * split;
        if (pd.rank > 0) offset += remaining;

#pragma omp parallel for
        for (size_t i = 0; i < pd.side; ++i)
        {
            double lz = -0.5 + 1.0 / (2.0 * pd.side) + i * 1.0 / pd.side;
            for (size_t j = 0; j < pd.side; ++j)
            {
                double lx = -0.5 + 1.0 / (2.0 * pd.side) + j * 1.0 / pd.side;
                for (size_t k = 0; k < pd.side; ++k)
                {
                    size_t lindex = i * pd.side * pd.side + j * pd.side + k;

                    if (lindex >= offset && lindex < offset + pd.count)
                    {
                        double ly = -0.5 + 1.0 / (2.0 * pd.side) + k * 1.0 / pd.side;

                        pd.z[lindex - offset] = lz;
                        pd.y[lindex - offset] = ly;
                        pd.x[lindex - offset] = lx;
                        pd.vx[lindex - offset] = 0.0;
                        pd.vy[lindex - offset] = 0.0;
                        pd.vz[lindex - offset] = 0.0;
                    }
                }
            }
        }
    }

    static void init(ParticlesData<T> &pd)
    {
        const T firstTimeStep = 1e-6;
        const T dx = 1.0 / pd.side;

        const double myPI = std::acos(-1.0);

        double energytot=1.0;
        double width=0.10;
        double ener0=energytot / std::pow(myPI,(3.0/2.0)) / 1.0 / std::pow(width,3);

#pragma omp parallel for
        for (size_t i = 0; i < pd.count; i++)
        {
            //pd.u[i]=ener0*exp(-(radius(i)**2/width**2))+1.d-08
            //pd.p[i]=u(i)*1.d0*(gamma-1.d0)

            //printf("%f %f %f\n", pd.x[i], pd.y[i], pd.z[i]);
            // CGS
            //pd.x[i] = pd.x[i];
            //pd.y[i] = pd.y[i];
            //pd.z[i] = pd.z[i];
            //pd.vx[i] = pd.vx[i] * 100.0;
            //pd.vy[i] = pd.vy[i] * 100.0;
            //pd.vz[i] = pd.vz[i] * 100.0;

            double radius = sqrt(pd.x[i] * pd.x[i] +  pd.y[i] * pd.y[i]+ pd.z[i] * pd.z[i]);

            pd.u[i] = ener0 * exp(-(std::pow(radius,2) / std::pow(width,2))) + 1.e-08;

            pd.p[i] = pd.u[i]*1.0*((5.0/3.0)-1.0);

            pd.m[i] = 1.0 / pd.n; // 1.0;//1000000.0/n;//1.0;//0.001;//0.001;//0.001;//1.0;
            //pd.c[i] = 3500.0;           // 35.0;//35.0;//35000
            pd.h[i] = 1.5 * dx;//0.28577500E-01 / 2.0; //2.0 * dx;         // 0.02;//0.02;
            pd.ro[i] = 1.0;             // 1.0e3;//.0;//1e3;//1e3;

            pd.mui[i] = 10.0;

            pd.du[i] = pd.du_m1[i] = 0.0;
            pd.dt[i] = pd.dt_m1[i] = firstTimeStep;
            pd.minDt = firstTimeStep;

            pd.grad_P_x[i] = pd.grad_P_y[i] = pd.grad_P_z[i] = 0.0;

            pd.x_m1[i] = pd.x[i] - pd.vx[i] * firstTimeStep;
            pd.y_m1[i] = pd.y[i] - pd.vy[i] * firstTimeStep;
            pd.z_m1[i] = pd.z[i] - pd.vz[i] * firstTimeStep;
        }

        pd.bbox.computeGlobal(pd.x, pd.y, pd.z);

        pd.bbox.zmax += dx / 2.0;
        pd.bbox.zmin -= dx / 2.0;
        pd.bbox.ymax += dx / 2.0;
        pd.bbox.ymin -= dx / 2.0;
        pd.bbox.xmax += dx / 2.0;
        pd.bbox.xmin -= dx / 2.0;
        pd.bbox.setBox(pd.bbox.xmin, pd.bbox.xmax, pd.bbox.ymin, pd.bbox.ymax, pd.bbox.zmin, pd.bbox.zmax, true, true, true);

        pd.etot = pd.ecin = pd.eint = 0.0;
        pd.ttot = 0.0;

        if (pd.rank == 0 && 2.0 * pd.h[0] > (pd.bbox.zmax - pd.bbox.zmin) / 2.0)
        {
            printf("ERROR::Sedov::init()::SmoothingLength (%.2f) too large (%.2f) (n too small?)\n", pd.h[0],
                   pd.bbox.zmax - pd.bbox.zmin);
#ifdef USE_MPI
            MPI_Finalize();
            exit(0);
#endif
        }
    }
};

} // namespace sphexa
