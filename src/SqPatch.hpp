#pragma once

#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>

#include "BBox.hpp"
#include "sph/kernels.hpp"

template <typename T>
class SqPatch
{
public:
    SqPatch(int side)
        : n(side * side * side)
        , side(side)
        , count(side * side * side)
        , arrayList({&x, &y, &z,        &x_m1,     &y_m1,     &z_m1, &vx,    &vy, &vz,    &ro,  &ro_0, &u,   &p,   &p_0, &h,
                &m, &c, &grad_P_x, &grad_P_y, &grad_P_z, &du,   &du_m1, &dt, &dt_m1, &c11, &c12,  &c13, &c22, &c23, &c33}) //, ng0(ng0),
                                                                                                                           // ngmax(1.5*ng0)
    {
#ifdef USE_MPI
        comm = MPI_COMM_WORLD;
        MPI_Comm_size(comm, &nrank);
        MPI_Comm_rank(comm, &rank);
        MPI_Get_processor_name(pname, &pnamelen);
#endif

        load();
        init();
    }

    inline void resize(unsigned int size)
    {
        for (unsigned int i = 0; i < arrayList.size(); i++)
            arrayList[i]->resize(size);
        neighbors.resize(size * ngmax);
        neighborsCount.resize(size);
    }

    // void load(const std::string &filename)
    void load()
    {
        int sideX = side;
        int sideY = side;
        int sideZ = 1;//side;
        n = side * side;
        bbox.PBCz = false;

        count = n / nrank;
        int offset = n % nrank;

        workload.resize(nrank);
        displs.resize(nrank);

        workload[0] = count + offset;
        displs[0] = 0;

        for (int i = 1; i < nrank; i++)
        {
            workload[i] = count;
            displs[i] = displs[i - 1] + workload[i - 1];
        }

        count = workload[rank];

        resize(count);

        const double omega = 5.0;
        const double myPI = std::acos(-1.0);

#pragma omp parallel for
        for (int i = 0; i < sideZ; ++i)
        {
            double lz = -0.5 + 1.0 / (2.0 * sideZ) + i * 1.0 / sideZ;

            for (int j = 0; j < sideY; ++j)
            {
                // double ly = -0.5 + 1.0 / (2.0 * side) +  (double)j / (double)side;
                double lx = -0.5 + 1.0 / (2.0 * sideY) + j * 1.0 / sideY;

                for (int k = 0; k < sideX; ++k)
                {
                    int lindex = i * sideZ * sideZ + j * sideY + k;

                    if (lindex >= displs[rank] && lindex < displs[rank] + workload[rank])
                    {
                        double ly = -0.5 + 1.0 / (2.0 * sideX) + k * 1.0 / sideX;
                        // double lx = -0.5 + 1.0 / (2.0 * side) + (double)k / (double)side;

                        double lvx = omega * ly;
                        double lvy = -omega * lx;
                        double lvz = 0.;
                        double lp_0 = 0.;

                        for (int m = 1; m <= 39; m += 2)
                            for (int l = 1; l <= 39; l += 2)
                                lp_0 = lp_0 - 32.0 * (omega * omega) / (m * l * (myPI * myPI)) /
                                                  ((m * myPI) * (m * myPI) + (l * myPI) * (l * myPI)) * sin(m * myPI * (lx + 0.5)) *
                                                  sin(l * myPI * (ly + 0.5));

                        lp_0 *= 1000.0;

                        z[lindex - displs[rank]] = lz;
                        y[lindex - displs[rank]] = ly;
                        x[lindex - displs[rank]] = lx;
                        vx[lindex - displs[rank]] = lvx;
                        vy[lindex - displs[rank]] = lvy;
                        vz[lindex - displs[rank]] = lvz;
                        p_0[lindex - displs[rank]] = lp_0;
                    }
                }
            }
        }
    }

    void init()
    {
        dx = 100.0 / side;

        #pragma omp parallel for
        for (int i = 0; i < count; i++)
        {
            // CGS
            x[i] = x[i] * 100.0;
            y[i] = y[i] * 100.0;
            z[i] = 0.0;//z[i] * 100.0;
            vx[i] = vx[i] * 100.0;
            vy[i] = vy[i] * 100.0;
            vz[i] = 0.0;//vz[i] * 100.0;
            p[i] = p_0[i] = p_0[i] * 10.0;

            m[i] = 1000000.0 / n; // 1.0;//1000000.0/n;//1.0;//0.001;//0.001;//0.001;//1.0;
            c[i] = 3500.0;        // 35.0;//35.0;//35000
            h[i] = 5.0 * dx;//2.5 * dx;      // 0.02;//0.02;
            ro[i] = 1.0;          // 1.0e3;//.0;//1e3;//1e3;
            ro_0[i] = 1.0;        // 1.0e3;//.0;//1e3;//1e3;

            du[i] = du_m1[i] = 0.0;
            dt[i] = dt_m1[i] = 1e-6;

            grad_P_x[i] = grad_P_y[i] = grad_P_z[i] = 0.0;

            x_m1[i] = x[i] - vx[i] * dt[0];
            y_m1[i] = y[i] - vy[i] * dt[0];
            z_m1[i] = 0.0;//z[i] - vz[i] * dt[0];
        }

#ifdef USE_MPI
        bbox.computeGlobal(x, y, z, comm);
#else
        bbox.compute(x, y, z);
#endif
        //bbox.PBCz = true;
        bbox.zmax += dx / 2.0;
        bbox.zmin -= dx / 2.0;

        etot = ecin = eint = 0.0;
        ttot = 0.0;
        minDt = 0.0;

        if (bbox.PBCz && rank == 0 && 2.0 * h[0] > (bbox.zmax - bbox.zmin) / 2.0)
        {
            printf("ERROR::SqPatch::init()::SmoothingLength (%.2f) too large (%.2f) (n too small?)\n", h[0], bbox.zmax - bbox.zmin);
#ifdef USE_MPI
            MPI_Finalize();
            exit(0);
#endif
        }
    }

    void writeData(const std::vector<int> &clist, std::string dumpfilename)
    {
        for(int turn=0; turn<nrank; turn++)
        {
            if(turn == rank)
            {
                std::ofstream dump;

                if(rank == 0)
                    dump.open(dumpfilename);
                else
                    dump.open(dumpfilename, std::ios_base::app);

                for (unsigned int pi = 0; pi < clist.size(); pi++)
                {
                    const int i = clist[pi];
                    const int nn = neighborsCount[pi];

                    dump << x[i] << ' ' << y[i] << ' ' << z[i] << ' ';
                    dump << vx[i] << ' ' << vy[i] << ' ' << vz[i] << ' ';
                    dump << h[i] << ' ' << ro[i] << ' ' << u[i] << ' ' << p[i] << ' ' << c[i] << ' ';
                    dump << grad_P_x[i] << ' ' << grad_P_y[i] << ' ' << grad_P_z[i] << ' ';
                    // T rad = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
                    // T vrad = (vx[i] * x[i] + vy[i] * y[i] + vz[i] * z[i]) / rad;
                    // dump << rad << ' ' << vrad << std::endl;
                    dump << rank << ' ' << nn << std::endl;
                }

                dump.close();

                MPI_Barrier(MPI_COMM_WORLD);
            }
            else
                MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    void writeConstants(const int iteration, const int nntot, std::ofstream &constants)
    {
        if (rank == 0)
        {
            constants << iteration << ' ' << ttot << ' ' << minDt << ' ' << etot << ' ' << ecin << ' ' << eint << ' ' << nntot << ' '
                      << std::endl;
            constants.flush();
        }
    }

    int iteration;                               // Current iteration
    int n, side, count;                          // Number of particles
    std::vector<T> x, y, z, x_m1, y_m1, z_m1;    // Positions
    std::vector<T> vx, vy, vz;                   // Velocities
    std::vector<T> ro, ro_0;                     // Density
    std::vector<T> u;                            // Internal Energy
    std::vector<T> p, p_0;                       // Pressure
    std::vector<T> h;                            // Smoothing Length
    std::vector<T> m;                            // Mass
    std::vector<T> c;                            // Speed of sound
    std::vector<T> grad_P_x, grad_P_y, grad_P_z; // gradient of the pressure
    std::vector<T> du, du_m1;                    // variation of the energy
    std::vector<T> dt, dt_m1;

    std::vector<T> c11, c12, c13, c22, c23, c33;

    T ttot, etot, ecin, eint;
    T minDt;

    sphexa::BBox<T> bbox;

    std::vector<int> neighbors; // List of neighbor indices per particle.
    std::vector<int> neighborsCount;
    std::vector<int> workload, displs;

#ifdef USE_MPI
    MPI_Comm comm;
    int pnamelen = 0;
    char pname[MPI_MAX_PROCESSOR_NAME];
#endif

    int rank = 0;
    int nrank = 1;

    std::vector<std::vector<T> *> arrayList;
    constexpr static T sincIndex = 6.0;
    constexpr static T Kcour = 0.2;
    constexpr static T maxDtIncrease = 1.1;
    constexpr static size_t ngmin = 5, ng0 = 500, ngmax = 650;
    const static T K;
    static T dx;

    // settings
    constexpr static ushort noOfGpuLoopSplits = 4; // No. of loop splits running in GPU to fit into the GPU memory
};

template <typename T>
T SqPatch<T>::dx = 0.01;

template <typename T>
const T SqPatch<T>::K = sphexa::compute_3d_k(sincIndex);
