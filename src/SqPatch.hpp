#pragma once

#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>

#include "BBox.hpp"

template<typename T>
class SqPatch
{
public:
    #ifdef USE_MPI
        SqPatch(int side, int ng0, MPI_Comm comm = MPI_COMM_WORLD) : 
            n(side*side*side), side(side), count(side*side*side), comm(comm),data({&x, &y, &z, &x_m1, &y_m1, &z_m1, &vx, &vy, &vz, 
                &ro, &ro_0, &u, &p, &p_0, &h, &m, &c, &grad_P_x, &grad_P_y, &grad_P_z, &du, &du_m1, &dt, &dt_m1}), ng0(ng0), ngmax(1.2*ng0)
        {
            MPI_Comm_size(comm, &nrank);
            MPI_Comm_rank(comm, &rank);
            MPI_Get_processor_name(pname, &pnamelen);
            loadMPI();
            init();
        }
    #else
         SqPatch(int side, int ng0) : 
            n(side*side*side), side(side), count(side*side*side), data({&x, &y, &z, &x_m1, &y_m1, &z_m1, &vx, &vy, &vz, 
                &ro, &ro_0, &u, &p, &p_0, &h, &m, &c, &grad_P_x, &grad_P_y, &grad_P_z, &du, &du_m1, &dt, &dt_m1}), ng0(ng0), ngmax(1.2*ng0)
        {
            std::cout << n << std::endl;
            resize(n);
            load();
            init();
        }
    #endif

    inline void resize(unsigned int size)
    {
        for(unsigned int i=0; i<data.size(); i++)
            data[i]->resize(size);
        neighbors.resize(size);
    }

    //void load(const std::string &filename)
    void load()
    {
        // input file stream
        // std::ifstream inputfile(filename, std::ios::binary);

        // if(inputfile.is_open())
        // {
        //     // read the contents of the file into the vectors
        //     inputfile.read(reinterpret_cast<char*>(x.data()), sizeof(T)*x.size());
        //     inputfile.read(reinterpret_cast<char*>(y.data()), sizeof(T)*y.size());
        //     inputfile.read(reinterpret_cast<char*>(z.data()), sizeof(T)*z.size());
        //     inputfile.read(reinterpret_cast<char*>(vx.data()), sizeof(T)*vx.size());
        //     inputfile.read(reinterpret_cast<char*>(vy.data()), sizeof(T)*vy.size());
        //     inputfile.read(reinterpret_cast<char*>(vz.data()), sizeof(T)*vz.size());
        //     inputfile.read(reinterpret_cast<char*>(p_0.data()), sizeof(T)*p_0.size());
        // }
        // else
        //     std::cout << "ERROR: " << "in opening file " << filename << std::endl;
        const double omega = 5.0;
        const double myPI = std::acos(-1.0);

        #pragma omp parallel for
        for (int i = 0; i < side; ++i)
        {
            double lz = -0.5 + 1.0 / (2.0 * side) + (double)i / (double)side;

            for (int j = 0; j < side; ++j)
            {
                double lx = -0.5 + 1.0 / (2 * side) + (double)j / (double)side;

                for (int k = 0; k < side; ++k)
                {
                    double ly = -0.5 + 1.0 / (2 * side) + (double)k / (double)side;
                    
                    double lvx = omega * ly;
                    double lvy = -omega * lx;
                    double lvz = 0.;
                    double lp_0 = 0.;

                    for (int m = 1; m < 39; m+=2)
                        for (int l = 1; l < 39; l+=2)
                            lp_0 = lp_0 - 32.0 * (omega * omega) / ((double)m * (double)l * (myPI * myPI)) / (((double)m * myPI) * ((double)m * myPI) + ((double)l * myPI) * ((double)l * myPI)) * sin((double)m * myPI * (lx + 0.5)) * sin((double)l * myPI * (ly + 0.5));

                    //lp_0 *= 1000.0;

                    //add to the vectors the current calculated values
                    int lindex = i*side*side + j*side + k;

                    z[lindex] = lz;
                    y[lindex] = ly;
                    x[lindex] = lx;
                    vx[lindex] = lvx;
                    vy[lindex] = lvy;
                    vz[lindex] = lvz;
                    p_0[lindex] = lp_0;
                }
            }
        }
    }

    #ifdef USE_MPI
    void loadMPI()
    {
        count = n / nrank;
        int offset = n % count;
        
        workload.resize(nrank);
        std::vector<int> displs(nrank);

        workload[0] = count+offset;
        displs[0] = 0;

        for(int i=1; i<nrank; i++)
        {
            workload[i] = count;
            displs[i] = displs[i-1] + workload[i-1];
        }

        if(rank == 0)
        {
            count += offset;

            resize(n);
            load();

            MPI_Scatterv(&x[0], &workload[0], &displs[0], MPI_DOUBLE, MPI_IN_PLACE, count, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(&y[0], &workload[0], &displs[0], MPI_DOUBLE, MPI_IN_PLACE, count, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(&z[0], &workload[0], &displs[0], MPI_DOUBLE, MPI_IN_PLACE, count, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(&vx[0], &workload[0], &displs[0], MPI_DOUBLE, MPI_IN_PLACE, count, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(&vy[0], &workload[0], &displs[0], MPI_DOUBLE, MPI_IN_PLACE, count, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(&vz[0], &workload[0], &displs[0], MPI_DOUBLE, MPI_IN_PLACE, count, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(&p_0[0], &workload[0], &displs[0], MPI_DOUBLE, MPI_IN_PLACE, count, MPI_DOUBLE, 0, comm);

            resize(count);
        }
        else
        {
            resize(count);

            MPI_Scatterv(NULL, &workload[0], &displs[0], MPI_DOUBLE, &x[0], count, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(NULL, &workload[0], &displs[0], MPI_DOUBLE, &y[0], count, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(NULL, &workload[0], &displs[0], MPI_DOUBLE, &z[0], count, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(NULL, &workload[0], &displs[0], MPI_DOUBLE, &vx[0], count, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(NULL, &workload[0], &displs[0], MPI_DOUBLE, &vy[0], count, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(NULL, &workload[0], &displs[0], MPI_DOUBLE, &vz[0], count, MPI_DOUBLE, 0, comm);
            MPI_Scatterv(NULL, &workload[0], &displs[0], MPI_DOUBLE, &p_0[0], count, MPI_DOUBLE, 0, comm);
        }
    }
    #endif

    void init()
    {
        for(int i=0; i<count; i++)
        {
            // CGS
            x[i] = x[i] * 100.0;
            y[i] = y[i] * 100.0;
            z[i] = z[i] * 100.0;
            vx[i] = vx[i] * 100.0;
            vy[i] = vy[i] * 100.0;
            vz[i] = vz[i] * 100.0;
            p_0[i] = p_0[i] * 10.0;

            m[i] = 1.0;//0.001;//0.001;//1.0;
            c[i] = 3500.0;//35.0;//35.0;//35000
            h[i] = 2.0;//0.02;//0.02;
            ro[i] = 1.0;//1e3;//1e3;
            ro_0[i] = 1.0;//1e3;//1e3;

            du[i] = du_m1[i] = 0.0;
            dt[i] = dt_m1[i] = 1e-7;

            grad_P_x[i] = grad_P_y[i] = grad_P_z[i] = 0.0;

            x_m1[i] = x[i] - vx[i] * dt[0];
            y_m1[i] = y[i] - vy[i] * dt[0];
            z_m1[i] = z[i] - vz[i] * dt[0];
        }

        bbox.PBCz = true;
        bbox.zmin = -50;//-0.5;//-50
        bbox.zmax = 50;//0.5;//50

        etot = ecin = eint = 0.0;
        ttot = 0.0;
        
        for(auto i : neighbors)
            i.reserve(ngmax);
    }

    void writeFile(const std::vector<int> &clist, std::ofstream &outputFile)
    {
        #ifdef USE_MPI
            std::vector<int> workload(nrank);

            int load = (int)clist.size();
            MPI_Allgather(&load, 1, MPI_INT, &workload[0], 1, MPI_INT, MPI_COMM_WORLD);

            std::vector<int> displs(nrank);

            displs[0] = 0;
            for(int i=1; i<nrank; i++)
                displs[i] = displs[i-1]+workload[i-1];

            if(rank == 0)
            {
                resize(n);

                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &x[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &y[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &z[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &vx[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &vy[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &vz[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &h[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &ro[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &u[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &p[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &c[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &grad_P_x[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &grad_P_y[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(MPI_IN_PLACE, (int)clist.size(), MPI_DOUBLE, &grad_P_z[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
            else
            {
                MPI_Gatherv(&x[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&y[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&z[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&vx[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&vy[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&vz[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&h[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&ro[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&u[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&p[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&c[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&grad_P_x[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&grad_P_y[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(&grad_P_z[0], (int)clist.size(), MPI_DOUBLE, NULL, &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        #endif

        if(rank == 0)
        {
            for(int i=0; i<n; i++)
            {
                outputFile << x[i] << ' ' << y[i] << ' ' << z[i] << ' ';
                outputFile << vx[i] << ' ' << vy[i] << ' ' << vz[i] << ' ';
                outputFile << h[i] << ' ' << ro[i] << ' ' << u[i] << ' ' << p[i] << ' ' << c[i] << ' ';
                outputFile << grad_P_x[i] << ' ' << grad_P_y[i] << ' ' << grad_P_z[i] << ' ';
                T rad = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
                T vrad = (vx[i] *  x[i] + vy[i] * y[i] + vz[i] * z[i]) / rad;
                outputFile << rad << ' ' << vrad;// << std::endl;

                outputFile << " " << neighbors[i].size() << std::endl;
            }
        }

        #ifdef USE_MPI
            if(rank == 0) resize(count);
        #endif 
    }

    int n, side, count; // Number of particles
    std::vector<T> x, y, z, x_m1, y_m1, z_m1; // Positions
    std::vector<T> vx, vy, vz; // Velocities
    std::vector<T> ro, ro_0; // Density
    std::vector<T> u; // Internal Energy
    std::vector<T> p, p_0; // Pressure
    std::vector<T> h; // Smoothing Length
    std::vector<T> m; // Mass
    std::vector<T> c; // Speed of sound

    std::vector<T> grad_P_x, grad_P_y, grad_P_z; //gradient of the pressure
    std::vector<T> du, du_m1; //variation of the energy
    std::vector<T> dt, dt_m1;

    T ttot, etot, ecin, eint;

    sphexa::BBox<T> bbox;
    std::vector<std::vector<int>> neighbors; // List of neighbor indices per particle.

    #ifdef USE_MPI
        MPI_Comm comm;
        int nrank = 0, pnamelen = 0;
        char pname[MPI_MAX_PROCESSOR_NAME];
        std::vector<int> workload;
    #endif
    
    int rank = 0;

    std::vector<std::vector<T>*> data;
    T sincIndex = 6.0;
    T K = sphexa::compute_3d_k(sincIndex);
    T Kcour = 0.2;
    T maxDtIncrease = 1.1;
    int stabilizationTimesteps = 15;
    unsigned int ngmin = 5, ng0 = 500, ngmax = 800;
};
