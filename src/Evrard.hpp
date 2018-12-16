#pragma once

#include <vector>
#include "BBox.hpp"

namespace sphexa
{

class Evrard
{
public:
    Evrard(const char *filename)
    {
        n = 1e6;

        //reserve the space for each vector
        x.resize(n);
        y.resize(n);
        z.resize(n);
        x_m1.resize(n);
        y_m1.resize(n);
        z_m1.resize(n);
        vx.resize(n);
        vy.resize(n);
        vz.resize(n);
        ro.resize(n);
        u.resize(n);
        p.resize(n);
        h.resize(n);
        m.resize(n);

        c.resize(n);
        cv.resize(n);
        temp.resize(n);
        mue.resize(n);
        mui.resize(n);

        grad_P_x.resize(n);
        grad_P_y.resize(n);
        grad_P_z.resize(n);

        du.resize(n);
        du_m1.resize(n);

        dt.resize(n);
        dt_m1.resize(n);

        neighbors.resize(n);
        for(auto i : neighbors)
            i.reserve(ngmax);

        FILE *f = fopen(filename, "rb");
        if(f)
        {
            fread(&x[0], sizeof(double), n, f);
            fread(&y[0], sizeof(double), n, f);
            fread(&z[0], sizeof(double), n, f);
            fread(&vx[0], sizeof(double), n, f);
            fread(&vy[0], sizeof(double), n, f);
            fread(&vz[0], sizeof(double), n, f);
            fread(&ro[0], sizeof(double), n, f);
            fread(&u[0], sizeof(double), n, f);
            fread(&p[0], sizeof(double), n, f);
            fread(&h[0], sizeof(double), n, f);
            fread(&m[0], sizeof(double), n, f);

            fclose(f);
        }
        else
        {
            printf("Error opening file %s\n", filename);
            exit(EXIT_FAILURE);
        }

        // // input file stream
        // ifstream inputfile(filename, std::ios::binary);

        // // read the contents of the file into the vectors
        // inputfile.read(reinterpret_cast<char*>(x.data()), x.size());
        // inputfile.read(reinterpret_cast<char*>(y.data()), y.size());
        // inputfile.read(reinterpret_cast<char*>(z.data()), z.size());
        // inputfile.read(reinterpret_cast<char*>(x_m1.data()), x_m1.size());
        // inputfile.read(reinterpret_cast<char*>(y_m1.data()), y_m1.size());
        // inputfile.read(reinterpret_cast<char*>(z_m1.data()), z_m1.size());
        // inputfile.read(reinterpret_cast<char*>(vx.data()), vx.size());
        // inputfile.read(reinterpret_cast<char*>(vy.data()), vy.size());
        // inputfile.read(reinterpret_cast<char*>(vz.data()), vz.size());
        // inputfile.read(reinterpret_cast<char*>(ro.data()), ro.size());
        // inputfile.read(reinterpret_cast<char*>(u.data()), u.size());
        // inputfile.read(reinterpret_cast<char*>(p.data()), p.size());
        // inputfile.read(reinterpret_cast<char*>(h.data()), h.size());
        // inputfile.read(reinterpret_cast<char*>(m.data()), m.size());
        // inputfile.read(reinterpret_cast<char*>(c.data()), c.size());
        // inputfile.read(reinterpret_cast<char*>(cv.data()), cv.size());
        // inputfile.read(reinterpret_cast<char*>(temp.data()), temp.size());
        // inputfile.read(reinterpret_cast<char*>(mue.data()), mue.size());
        // inputfile.read(reinterpret_cast<char*>(mui.data()), mui.size());

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

    ~Evrard(){}

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

