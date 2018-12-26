#include <iostream>
#include <fstream>
#include <string>

#include "sphexa.hpp"
#include "Evrard.hpp"

using namespace std;
using namespace sphexa;

int main()
{
    typedef double Real;
    typedef Octree<Real> Tree;
    typedef Evrard<Real> Dataset;

    int n = 1e6;
    Dataset d(n, "bigfiles/Evrard3D.bin");
    Domain<Real, Tree> domain(d.ngmin, d.ng0, d.ngmax);
    Density<Real> density(d.K);
    EquationOfState<Real> equationOfState;
    MomentumEnergy<Real> momentumEnergy(d.K);
    Timestep<Real> timestep(d.K, d.maxDtIncrease);
    UpdateQuantities<Real> updateQuantities(d.stabilizationTimesteps);
    EnergyConservation<Real> energyConservation;

    for(int iteration = 0; iteration < 200; iteration++)
    {
        timer::TimePoint start = timer::Clock::now();

        cout << "Iteration: " << iteration << endl;

        REPORT_TIME(domain.buildTree(d.x, d.y, d.z, d.h, d.bbox), "BuildTree");
        REPORT_TIME(domain.reorderParticles(d), "ReorderParticles");
        REPORT_TIME(domain.findNeighbors(d.x, d.y, d.z, d.h, d.neighbors), "FindNeighbors");
        REPORT_TIME(density.compute(d.neighbors, d.x, d.y, d.z, d.h, d.m, d.ro), "Density");
        REPORT_TIME(equationOfState.compute(d.ro, d.mui, d.temp, d.u, d.p, d.c, d.cv), "EquationOfState");
        REPORT_TIME(momentumEnergy.compute(d.neighbors, d.x, d.y, d.z, d.h, d.vx, d.vy, d.vz, d.ro, d.p, d.c, d.m, d.grad_P_x, d.grad_P_y, d.grad_P_z, d.du), "MomentumEnergy");
        REPORT_TIME(timestep.compute(d.h, d.c, d.dt_m1, d.dt), "Timestep");
        REPORT_TIME(updateQuantities.compute(iteration, d.grad_P_x, d.grad_P_y, d.grad_P_z, d.dt, d.du, d.bbox, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.u, d.du_m1, d.dt_m1), "UpdateQuantities");
        REPORT_TIME(energyConservation.compute(d.u, d.vx, d.vy, d.vz, d.m, d.etot, d.ecin, d.eint), "EnergyConservation");

        int totalNeighbors = 0;
        #pragma omp parallel for reduction (+:totalNeighbors)
        for(unsigned int i=0; i<d.neighbors.size(); i++)
            totalNeighbors += d.neighbors[i].size();

        cout << "### Check ### Computational domain: ";
        cout << d.bbox.xmin << " " << d.bbox.xmax << " ";
        cout << d.bbox.ymin << " " << d.bbox.ymax << " ";
        cout << d.bbox.zmin << " " << d.bbox.zmax << endl;
        cout << "### Check ### Avg. number of neighbours: " << totalNeighbors/n << "/" << d.ng0 << endl;
        cout << "### Check ### New Time-step: " << d.dt[0] << endl;
        cout << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", cinetic: " << d.ecin << ")" << endl;

        if(iteration % 10 == 0)
        {
            std::ofstream outputFile("output" + to_string(iteration) + ".txt");
            REPORT_TIME(d.writeFile(outputFile), "writeFile");
            outputFile.close();
        }

        timer::TimePoint stop = timer::Clock::now();
        cout << "=== Total time for iteration " << timer::duration(start, stop) << "s" << endl << endl;
    }

    return 0;
}

