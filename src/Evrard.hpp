#pragma once

#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>

#include "BBox.hpp"

template<typename T>
class Evrard
{
public:

    Evrard() = delete;
    ~Evrard() = default;

    Evrard(int n, const char *filename) : 
    	n(n), x(n), y(n), z(n), x_m1(n), y_m1(n), z_m1(n), vx(n), vy(n), vz(n), 
    	ro(n), u(n), p(n), h(n), m(n), c(n), cv(n), temp(n), mue(n), mui(n), 
    	grad_P_x(n), grad_P_y(n), grad_P_z(n), 
    	du(n), du_m1(n), dt(n), dt_m1(n), neighbors(n)
    {
        for(auto i : neighbors)
            i.reserve(ngmax);

        // input file stream
        std::ifstream inputfile(filename, std::ios::binary);

        // read the contents of the file into the vectors
        inputfile.read(reinterpret_cast<char*>(x.data()), sizeof(T)*x.size());
        inputfile.read(reinterpret_cast<char*>(y.data()), sizeof(T)*y.size());
        inputfile.read(reinterpret_cast<char*>(z.data()), sizeof(T)*z.size());
        inputfile.read(reinterpret_cast<char*>(vx.data()), sizeof(T)*vx.size());
        inputfile.read(reinterpret_cast<char*>(vy.data()), sizeof(T)*vy.size());
        inputfile.read(reinterpret_cast<char*>(vz.data()), sizeof(T)*vz.size());
        inputfile.read(reinterpret_cast<char*>(ro.data()), sizeof(T)*ro.size());
        inputfile.read(reinterpret_cast<char*>(u.data()), sizeof(T)*u.size());
        inputfile.read(reinterpret_cast<char*>(p.data()), sizeof(T)*p.size());
        inputfile.read(reinterpret_cast<char*>(h.data()), sizeof(T)*h.size());
        inputfile.read(reinterpret_cast<char*>(m.data()), sizeof(T)*m.size());

        std::fill(temp.begin(), temp.end(), 1.0);
        std::fill(mue.begin(), mue.end(), 2.0);
        std::fill(mui.begin(), mui.end(), 10.0);
        std::fill(vx.begin(), vx.end(), 0.0);
        std::fill(vy.begin(), vy.end(), 0.0);
        std::fill(vz.begin(), vz.end(), 0.0);

        std::fill(grad_P_x.begin(), grad_P_x.end(), 0.0);
        std::fill(grad_P_y.begin(), grad_P_y.end(), 0.0);
        std::fill(grad_P_z.begin(), grad_P_z.end(), 0.0);

        std::fill(du.begin(), du.end(), 0.0);
        std::fill(du_m1.begin(), du_m1.end(), 0.0);

        std::fill(dt.begin(), dt.end(), 0.0001);
        std::fill(dt_m1.begin(), dt_m1.end(), 0.0001);

        for(int i=0; i<n; i++)
        {
            x_m1[i] = x[i] - vx[i] * dt[0];
            y_m1[i] = y[i] - vy[i] * dt[0];
            z_m1[i] = z[i] - vz[i] * dt[0];
        }

        etot = ecin = eint = 0.0;

        iteration = 0;
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
        reorderSwap(ordering, u);
        reorderSwap(ordering, p);
        reorderSwap(ordering, h);
        reorderSwap(ordering, m);
        reorderSwap(ordering, c);
        reorderSwap(ordering, cv);
        reorderSwap(ordering, temp);
        reorderSwap(ordering, mue);
        reorderSwap(ordering, mui);
        reorderSwap(ordering, grad_P_x);
        reorderSwap(ordering, grad_P_y);
        reorderSwap(ordering, grad_P_z);
        reorderSwap(ordering, du);
        reorderSwap(ordering, du_m1);
        reorderSwap(ordering, dt);
        reorderSwap(ordering, dt_m1);
    }

    void computeBBox()
    {
        bbox.xmin = INFINITY;
        bbox.xmax = -INFINITY;
        bbox.ymin = INFINITY;
        bbox.ymax = -INFINITY;
        bbox.zmin = INFINITY;
        bbox.zmax = -INFINITY;
        for(int i=0; i<n; i++)
        {
            if(x[i] < bbox.xmin) bbox.xmin = x[i];
            if(x[i] > bbox.xmax) bbox.xmax = x[i];
            if(y[i] < bbox.ymin) bbox.ymin = y[i];
            if(y[i] > bbox.ymax) bbox.ymax = y[i];
            if(z[i] < bbox.zmin) bbox.zmin = z[i];
            if(z[i] > bbox.zmax) bbox.zmax = z[i];
        }
    }

    void writeFile()
    {
        if(iteration % 10 == 0)
        {
            std::ofstream outputFile;
            std::ostringstream oss;
            oss << "output" << iteration << ".txt";
            outputFile.open(oss.str());
        
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
            outputFile.close();
        }
    }

    int n; // Number of particles
    std::vector<T> x, y, z, x_m1, y_m1, z_m1; // Positions
    std::vector<T> vx, vy, vz; // Velocities
    std::vector<T> ro; // Density
    std::vector<T> u; // Internal Energy
    std::vector<T> p; // Pressure
    std::vector<T> h; // Smoothing Length
    std::vector<T> m; // Mass
    std::vector<T> c; // Speed of sound
    std::vector<T> cv; // Specific heat
    std::vector<T> temp; // Temperature
    std::vector<T> mue; // Mean molecular weigh of electrons
    std::vector<T> mui; // Mean molecular weight of ions
    std::vector<T> grad_P_x, grad_P_y, grad_P_z; //gradient of the pressure
    std::vector<T> du, du_m1; //variation of the energy
    std::vector<T> dt, dt_m1;

    T etot, ecin, eint;

    int ngmin = 50, ng0 = 100, ngmax = 150; // Minimum, target and maximum number of neighbors per particle
    std::vector<std::vector<int>> neighbors; // List of neighbor indices per particle.

    // Domain box
    sphexa::BBox bbox;

    int iteration;
};
