#pragma once

#include <vector>
#include <fstream>
#include "BBox.hpp"

namespace sphexa
{

class Dataset
{
public:
    Dataset(int n, const char *filename) : 
    	n(n), x.resize(n), y(n), z(n), x_m1(n), y_m1(n), z_m1(n), vx(n), vy(n), vz(n), 
    	ro(n), u(n), p(n), h(n), m(n), c(n), cv(n), temp(n), mue(n), mui(n), 
    	grad_P_x(n), grad_P_y(n), grad_P_z(n), 
    	du(n), du_m1(n), dt(n), dt_m1(n), neighbors(n)
    {
        for(auto i : neighbors)
            i.reserve(ngmax);

        // input file stream
        std::ifstream inputfile(filename, std::ios::binary);

        // read the contents of the file into the vectors
        inputfile.read(reinterpret_cast<double>(x.data()), x.size());
        inputfile.read(reinterpret_cast<double>(y.data()), y.size());
        inputfile.read(reinterpret_cast<double>(z.data()), z.size());
        inputfile.read(reinterpret_cast<double>(vx.data()), vx.size());
        inputfile.read(reinterpret_cast<double>(vy.data()), vy.size());
        inputfile.read(reinterpret_cast<double>(vz.data()), vz.size());
        inputfile.read(reinterpret_cast<double>(ro.data()), ro.size());
        inputfile.read(reinterpret_cast<double>(u.data()), u.size());
        inputfile.read(reinterpret_cast<double>(p.data()), p.size());
        inputfile.read(reinterpret_cast<double>(h.data()), h.size());
        inputfile.read(reinterpret_cast<double>(m.data()), m.size());

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

        iteration = 0;
    }

    ~Dataset() {}

    int n; // Number of particles
    std::vector<double> x, y, z, x_m1, y_m1, z_m1; // Positions
    std::vector<double> vx, vy, vz; // Velocities
    std::vector<double> ro; // Density
    std::vector<double> u; // Internal Energy
    std::vector<double> p; // Pressure
    std::vector<double> h; // Smoothing Length
    std::vector<double> m; // Mass
    std::vector<double> c; // Speed of sound
    std::vector<double> cv; // Specific heat
    std::vector<double> temp; // Temperature
    std::vector<double> mue; // Mean molecular weigh of electrons
    std::vector<double> mui; // Mean molecular weight of ions
    std::vector<double> grad_P_x, grad_P_y, grad_P_z; //gradient of the pressure
    std::vector<double> du, du_m1; //variation of the energy
    std::vector<double> dt, dt_m1;

    int ngmax = 150; // Maximum number of neighbors per particle
    std::vector<std::vector<int>> neighbors; // List of neighbor indices per particle.

    // Domain box
    BBox bbox;

    // Periodic boundary conditions
    bool PBCx = false, PBCy = false, PBCz = false;
    
    // Global bounding box (of the domain)
    double xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0, zmin = -1.0, zmax = 1.0;

    int iteration;
};

}