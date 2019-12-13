#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <mpi.h>

#include "BBox.hpp"
#include "sph/kernels.hpp"
#include "ParticlesData.hpp"

namespace sphexa
{
template <typename T>
class WindblobDataGenerator
{
public:
    static ParticlesData<T> generate(const std::string &filename)
    {
        ParticlesData<T> pd;

#ifdef USE_MPI
        pd.comm = MPI_COMM_WORLD;
        MPI_Comm_size(pd.comm, &pd.nrank);
        MPI_Comm_rank(pd.comm, &pd.rank);
        MPI_Get_processor_name(pd.pname, &pd.pnamelen);
#endif

        pd.n = 3157385;

        load(pd, filename);
        init(pd);

        return pd;
    }

#define CHECK_ERR(func) { \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: calling %s (%s)\n",__LINE__, #func, errorString); \
    } \
}

    // void load(const std::string &filename)
    static void load(ParticlesData<T> &pd, const std::string &filename)
    {
        size_t split = pd.n / pd.nrank;
        size_t remaining = pd.n - pd.nrank * split;

        pd.count = split;
        if(pd.rank == 0)
            pd.count += remaining;

        pd.resize(pd.count);

        MPI_File fh;
        MPI_Status status;

        MPI_Offset col = pd.n*sizeof(double);

        MPI_Offset offset = pd.rank*split*sizeof(double);
        if(pd.rank > 0)
            offset += remaining*sizeof(double);

        int err = MPI_File_open(MPI_COMM_SELF, filename.c_str(), MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
        CHECK_ERR(MPI_File_open to write);

        MPI_File_set_view(fh, offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.x[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 1*col+offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.y[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 2*col+offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.z[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 3*col+offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.vx[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 4*col+offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.vy[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 5*col+offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.vz[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 6*col+offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.ro[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 7*col+offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.p[0], pd.count, MPI_DOUBLE, &status);

        MPI_File_set_view(fh, 8*col+offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
        MPI_File_read(fh, &pd.h[0], pd.count, MPI_DOUBLE, &status);

        //printf("%%lf %lf %lf %lf %lf %lf %lf %lf\n", pd.x[i], pd.y[i], pd.z[i], pd.vx[i], pd.vy[i], pd.vz[i], pd.p[i], pd.h[i]);

        MPI_File_close(&fh);
    }

    static void init(ParticlesData<T> &pd)
    {
        const T firstTimeStep = 1e-9;
        const T gamma = 5.0/3.0;

        #pragma omp parallel for
        for (size_t i = 0; i < pd.count; i++)
        {
            pd.x[i] = pd.x[i];
            pd.y[i] = pd.y[i];
            pd.z[i] = pd.z[i];
            pd.vx[i] = pd.vx[i];
            pd.vy[i] = pd.vy[i];
            pd.vz[i] = pd.vz[i];
            pd.p[i] = pd.p[i];

            pd.m[i] = 1.99915e-08;
            pd.du[i] = pd.du_m1[i] = 0.0;
            pd.dt[i] = pd.dt_m1[i] = firstTimeStep;
            pd.minDt = firstTimeStep;

            pd.h[i] = pd.h[i];
            pd.u[i] = pd.p[i]/gamma/pd.ro[i];
            pd.c[i] = sqrt((2.0/3.0)*pd.u[i]);

            pd.grad_P_x[i] = pd.grad_P_y[i] = pd.grad_P_z[i] = 0.0;

            pd.x_m1[i] = pd.x[i] - pd.vx[i] * firstTimeStep;
            pd.y_m1[i] = pd.y[i] - pd.vy[i] * firstTimeStep;
            pd.z_m1[i] = pd.z[i] - pd.vz[i] * firstTimeStep;
        }

        pd.bbox.setBox(0.0, 0.25, 0.0, 0.25, 0.0, 1.0, true, true, true);
        pd.etot = pd.ecin = pd.eint = 0.0;
        pd.ttot = 0.0;

        pd.iteration = 0;

        if (pd.rank == 0 && 2.0 * pd.h[0] > (pd.bbox.zmax - pd.bbox.zmin) / 2.0)
        {
            printf("ERROR::SqPatch::init()::SmoothingLength (%.2f) too large (%.2f) (n too small?)\n", pd.h[0],
                   pd.bbox.zmax - pd.bbox.zmin);
#ifdef USE_MPI
            MPI_Finalize();
            exit(0);
#endif
        }
    }
};
} // namespace sphexa
