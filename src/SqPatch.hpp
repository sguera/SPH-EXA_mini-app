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
    SqPatch(int n, const char *filename) : 
    	n(n), x(n), y(n), z(n), x_m1(n), y_m1(n), z_m1(n), vx(n), vy(n), vz(n), 
    	ro(n), ro_0(n), u(n), p(n), p_0(n), h(n), m(n), c(n), temp(n), 
    	grad_P_x(n), grad_P_y(n), grad_P_z(n), 
    	du(n), du_m1(n), dt(n), dt_m1(n), neighbors(n)
    {
        for(auto i : neighbors)
            i.reserve(ngmax);

        // input file stream
        std::ifstream inputfile(filename, std::ios::binary);
        if(!inputfile)
        {
            std::cout << "Couldn't open file " << filename << std::endl;
            exit(1);
        }

        // read the contents of the file into the vectors
        inputfile.read(reinterpret_cast<char*>(x.data()), sizeof(T)*x.size());
        inputfile.read(reinterpret_cast<char*>(y.data()), sizeof(T)*y.size());
        inputfile.read(reinterpret_cast<char*>(z.data()), sizeof(T)*z.size());
        inputfile.read(reinterpret_cast<char*>(vx.data()), sizeof(T)*vx.size());
        inputfile.read(reinterpret_cast<char*>(vy.data()), sizeof(T)*vy.size());
        inputfile.read(reinterpret_cast<char*>(vz.data()), sizeof(T)*vz.size());
        inputfile.read(reinterpret_cast<char*>(p_0.data()), sizeof(T)*p_0.size());

        std::fill(h.begin(), h.end(), 2.0);
        std::fill(temp.begin(), temp.end(), 1.0);
        std::fill(ro_0.begin(), ro_0.end(), 1.0);
        std::fill(ro.begin(), ro.end(), 0.0);
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

        bbox.zmin = -50;
        bbox.zmax = 50;
    }

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

    void writeFile(std::ofstream &outputFile)
    {
        for(int i=0; i<n; i++)
        {
            outputFile << x[i] << ' ' << y[i] << ' ' << z[i] << ' ';
            outputFile << vx[i] << ' ' << vy[i] << ' ' << vz[i] << ' ';
            outputFile << h[i] << ' ' << ro[i] << ' ' << u[i] << ' ' << p[i] << ' ' << c[i] << ' ';
            outputFile << grad_P_x[i] << ' ' << grad_P_y[i] << ' ' << grad_P_z[i] << ' ';
            T rad = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
            T vrad = (vx[i] *  x[i] + vy[i] * y[i] + vz[i] * z[i]) / rad;
            outputFile << rad << ' ' << vrad << std::endl;  
        }
    }

    int n; // Number of particles
    std::vector<T> x, y, z, x_m1, y_m1, z_m1; // Positions
    std::vector<T> vx, vy, vz; // Velocities
    std::vector<T> ro, ro_0; // Density
    std::vector<T> u; // Internal Energy
    std::vector<T> p, p_0; // Pressure
    std::vector<T> h; // Smoothing Length
    std::vector<T> m; // Mass
    std::vector<T> c; // Speed of sound
    std::vector<T> temp; // Temperature

    std::vector<T> grad_P_x, grad_P_y, grad_P_z; //gradient of the pressure
    std::vector<T> du, du_m1; //variation of the energy
    std::vector<T> dt, dt_m1;

    T etot, ecin, eint;

    sphexa::BBox<T> bbox;
    std::vector<std::vector<int>> neighbors; // List of neighbor indices per particle.

    const T K = sphexa::compute_3d_k(6.0);
    const T maxDtIncrease = 1.1;
    const int stabilizationTimesteps = 15;
    const int ngmin = 450, ng0 = 500, ngmax = 550;
    const bool PBCx = false, PBCy = false, PBCz = false;
};
