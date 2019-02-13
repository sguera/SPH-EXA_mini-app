#include <iostream>
#include <fstream>
#include <string>

#include "sphexa.hpp"
#include "debug.hpp"

#include "SqPatch.hpp"

using namespace std;
using namespace sphexa;

#define REPORT_TIME(rank, expr, name) \
    if (rank == 0) sphexa::timer::report_time([&](){ expr; }, name); \
    else { expr; }

int main(int argc, char **argv)
{
    ArgParser parser(argc, argv);

    int n = parser.getInt("-n", 1e6);
    int maxStep = parser.getInt("-s", 1e5);
    int writeFrequency = parser.getInt("-w", 250);
    std::string inputFilename = parser.getString("-f", "bigfiles/squarepatch3D_1M.bin");

    #ifdef _JENKINS
        maxStep = 0;
        writeFrequency = -1;
    #endif

	debug();

    typedef double Real;
    typedef Octree<Real> Tree;
    typedef SqPatch<Real> Dataset;

    #ifdef USE_MPI
        MPI_Init(NULL, NULL);
    #endif

    Dataset d(n, inputFilename);
    DistributedDomain<Real, Tree> domain(d.ngmin, d.ng0, d.ngmax);
    Density<Real> density(d.sincIndex, d.K);
    EquationOfStateSqPatch<Real> equationOfState(d.stabilizationTimesteps);
    MomentumEnergySqPatch<Real> momentumEnergy(d.stabilizationTimesteps, d.sincIndex, d.K);
    Timestep<Real> timestep(d.Kcour, d.maxDtIncrease);
    UpdateQuantities<Real> updateQuantities(d.stabilizationTimesteps);
    EnergyConservation<Real> energyConservation;

    vector<int> clist(d.x.size());
    for(int i=0; i<(int)clist.size(); i++)
        clist[i] = i;

    for(int iteration = 0; iteration <= maxStep; iteration++)
    {
        timer::TimePoint start = timer::Clock::now();

        if(d.rank == 0) cout << "Iteration: " << iteration << endl;
        
        #ifdef USE_MPI
            REPORT_TIME(d.rank, domain.build(d.workload, d.bbox, d.x, d.y, d.z, d.h, clist, d.data, false), "mpi::build");
            REPORT_TIME(d.rank, domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.m), "mpi::synchronizeHalos");
            d.count = clist.size();
            if(d.rank == 0) cout << "# mpi::clist.size: " << clist.size() << " halos: " << domain.haloCount << endl;
        #endif

        REPORT_TIME(d.rank, domain.buildTree(d.x, d.y, d.z, d.h, d.bbox), "BuildTree");


        // REPORT_TIME(d.rank, mpi.reorder(d.data), "ReorderParticles");
        REPORT_TIME(d.rank, domain.findNeighbors(clist, d.bbox, d.x, d.y, d.z, d.h, d.neighbors), "FindNeighbors");
        REPORT_TIME(d.rank, density.compute(clist, d.bbox, d.neighbors, d.x, d.y, d.z, d.h, d.m, d.ro), "Density");
        REPORT_TIME(d.rank, equationOfState.compute(clist, iteration, d.ro_0, d.p_0, d.ro, d.p, d.u, d.c), "EquationOfState");
        
        #ifdef USE_MPI
            d.resize(d.count); // Discard old neighbors
            REPORT_TIME(d.rank, domain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c), "mpi::synchronizeHalos");
        #endif

        REPORT_TIME(d.rank, momentumEnergy.compute(clist, d.bbox, iteration, d.neighbors, d.x, d.y, d.z, d.h, d.vx, d.vy, d.vz, d.ro, d.p, d.c, d.m, d.grad_P_x, d.grad_P_y, d.grad_P_z, d.du), "MomentumEnergy");
        REPORT_TIME(d.rank, timestep.compute(clist, d.h, d.c, d.dt_m1, d.dt, d.ttot), "Timestep");
        REPORT_TIME(d.rank, updateQuantities.compute(clist, iteration, d.grad_P_x, d.grad_P_y, d.grad_P_z, d.dt, d.du, d.bbox, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.u, d.du_m1, d.dt_m1), "UpdateQuantities");
        REPORT_TIME(d.rank, energyConservation.compute(clist, d.u, d.vx, d.vy, d.vz, d.m, d.etot, d.ecin, d.eint), "EnergyConservation");
        REPORT_TIME(d.rank, domain.updateSmoothingLength(clist, d.neighbors, d.h), "SmoothingLength");

        int totalNeighbors = domain.neighborsSum(clist, d.neighbors);

        if(d.rank == 0)
        {
            cout << "### Check ### Computational domain: ";
            cout << d.bbox.xmin << " " << d.bbox.xmax << " ";
            cout << d.bbox.ymin << " " << d.bbox.ymax << " ";
            cout << d.bbox.zmin << " " << d.bbox.zmax << endl;
            cout << "### Check ### Total number of neighbours: " << totalNeighbors << endl;
            cout << "### Check ### Total time: " << d.ttot << ", current time-step: " << d.dt[0] << endl;
            cout << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", cinetic: " << d.ecin << ")" << endl;
        }

        if(writeFrequency > 0 && iteration % writeFrequency == 0)
        {
            std::ofstream outputFile("output" + to_string(iteration) + ".txt");
            REPORT_TIME(d.rank, d.writeFile(clist, outputFile), "writeFile");
            outputFile.close();
        }

        timer::TimePoint stop = timer::Clock::now();
        
        if(d.rank == 0) cout << "=== Total time for iteration " << timer::duration(start, stop) << "s" << endl << endl;
    }

    #ifdef USE_MPI
        MPI_Finalize();
    #endif

    return 0;
}

