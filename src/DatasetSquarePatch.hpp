#pragma once

#include <vector>
#include <fstream>
#include "BBox.hpp"

namespace sphexa
{

class DatasetSquarePatch
{
public:

    DatasetSquarePatch() = delete;
    ~DatasetSquarePatch() = default;

    DatasetSquarePatch(int n, const char *filename) : 
    	n(n), x(n), y(n), z(n), x_m1(n), y_m1(n), z_m1(n), vx(n), vy(n), vz(n), 
    	ro(n), ro_0(n), u(n), p(n), p_0(n), h(n), m(n), c(n), temp(n), 
    	grad_P_x(n), grad_P_y(n), grad_P_z(n), 
    	du(n), du_m1(n), dt(n), dt_m1(n), neighbors(n)
    {
        for(auto i : neighbors)
            i.reserve(ngmax);

        // input file stream
        std::ifstream inputfile(filename, std::ios::binary);

        // read the contents of the file into the vectors
        inputfile.read(reinterpret_cast<char*>(x.data()), sizeof(double)*x.size());
        inputfile.read(reinterpret_cast<char*>(y.data()), sizeof(double)*y.size());
        inputfile.read(reinterpret_cast<char*>(z.data()), sizeof(double)*z.size());
        inputfile.read(reinterpret_cast<char*>(vx.data()), sizeof(double)*vx.size());
        inputfile.read(reinterpret_cast<char*>(vy.data()), sizeof(double)*vy.size());
        inputfile.read(reinterpret_cast<char*>(vz.data()), sizeof(double)*vz.size());
        inputfile.read(reinterpret_cast<char*>(p_0.data()), sizeof(double)*ro.size());


        std::fill(h.begin(), h.end(), 2.0);
        std::fill(temp.begin(), temp.end(), 1.0);
        std::fill(ro_0.begin(), ro_0.end(), 1.0);
        std::fill(c.begin(), c.end(), 3500.0);
        std::fill(m.begin(), m.end(), 1.0);

        for(int i=0; i<n; i++)
        {
            p_0[i] = p_0[i] * 10.0;
            x[i] = x[i] * 100.0;
            y[i] = y[i] * 100.0;
            z[i] = z[i] * 100.0;
            vx[i] = vx[i] * 100.0;
            vy[i] = vy[i] * 100.0;
            vz[i] = vz[i] * 100.0;
        }

        std::fill(grad_P_x.begin(), grad_P_x.end(), 0.0);
        std::fill(grad_P_y.begin(), grad_P_y.end(), 0.0);
        std::fill(grad_P_z.begin(), grad_P_z.end(), 0.0);

        std::fill(du.begin(), du.end(), 0.0);
        std::fill(du_m1.begin(), du_m1.end(), 0.0);

        std::fill(dt.begin(), dt.end(), 1e-6);
        std::fill(dt_m1.begin(), dt_m1.end(), 1e-6);

        for(int i=0; i<n; i++)
        {
            x_m1[i] = x[i] - vx[i] * dt[0];
            y_m1[i] = y[i] - vy[i] * dt[0];
            z_m1[i] = z[i] - vz[i] * dt[0];
        }

        iteration = 0;
    }

    template<typename T>
    void reorderSwap(const std::vector<int> &ordering, std::vector<T> &data)
    {
        std::vector<T> tmp(ordering.size());
        for(unsigned int i=0; i<ordering.size(); i++)
            tmp[i] = data[ordering[i]];
        tmp.swap(data);
    }

    void reorder(const std::vector<int> &ordering)
    {
        reorderSwap(ordering, x);
        reorderSwap(ordering, y);
        reorderSwap(ordering, z);
        reorderSwap(ordering, x_m1);
        reorderSwap(ordering, y_m1);
        reorderSwap(ordering, z_m1);
        reorderSwap(ordering, vx);
        reorderSwap(ordering, vy);
        reorderSwap(ordering, vz);
        reorderSwap(ordering, ro);
        reorderSwap(ordering, ro_0);
        reorderSwap(ordering, u);
        reorderSwap(ordering, p);
        reorderSwap(ordering, p_0);
        reorderSwap(ordering, h);
        reorderSwap(ordering, m);
        reorderSwap(ordering, c);
        reorderSwap(ordering, temp);
        reorderSwap(ordering, grad_P_x);
        reorderSwap(ordering, grad_P_y);
        reorderSwap(ordering, grad_P_z);
        reorderSwap(ordering, du);
        reorderSwap(ordering, du_m1);
        reorderSwap(ordering, dt);
        reorderSwap(ordering, dt_m1);
    }

    int n; // Number of particles
    std::vector<double> x, y, z, x_m1, y_m1, z_m1; // Positions
    std::vector<double> vx, vy, vz; // Velocities
    std::vector<double> ro, ro_0; // Density
    std::vector<double> u; // Internal Energy
    std::vector<double> p, p_0; // Pressure
    std::vector<double> h; // Smoothing Length
    std::vector<double> m; // Mass
    std::vector<double> c; // Speed of sound
    std::vector<double> temp; // Temperature

    std::vector<double> grad_P_x, grad_P_y, grad_P_z; //gradient of the pressure
    std::vector<double> du, du_m1; //variation of the energy
    std::vector<double> dt, dt_m1;

    int ngmax = 550; // Maximum number of neighbors per particle
    std::vector<std::vector<int>> neighbors; // List of neighbor indices per particle.

    // Domain box
    BBox bbox;

    // Periodic boundary conditions
    bool PBCx = false, PBCy = false, PBCz = true;
    
    // Global bounding box (of the domain)
    double xmin = -10.0, xmax = 10.0, ymin = -10.0, ymax = 10.0, zmin = -10.0, zmax = 10.0;

    int iteration;
};

}

