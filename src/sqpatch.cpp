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

    int cubeSide = parser.getInt("-n", 50);
    int maxStep = parser.getInt("-s", 10);
    int writeFrequency = parser.getInt("-w", -1);

    #ifdef _JENKINS
        maxStep = 0;
        writeFrequency = -1;
    #endif

    typedef double Real;
    typedef Octree<Real> Tree;
    typedef SqPatch<Real> Dataset;

    #ifdef USE_MPI
        MPI_Init(NULL, NULL);
    #endif

    Dataset d(cubeSide);
    #ifdef USE_MPI
        DistributedDomain<Real, Tree> domain(d.ngmin, d.ng0, d.ngmax);
    #else
        Domain<Real, Tree> domain(d.ngmin, d.ng0, d.ngmax);
    #endif
    Density<Real> density(d.sincIndex, d.K);
    EquationOfStateSqPatch<Real> equationOfState;
    MomentumEnergySqPatch<Real> momentumEnergy(d.dx, d.sincIndex, d.K);
    Timestep<Real> timestep(d.Kcour, d.maxDtIncrease);
    UpdateQuantities<Real> updateQuantities;
    EnergyConservation<Real> energyConservation;

    vector<int> clist(d.count);
    for(int i=0; i<(int)clist.size(); i++)
        clist[i] = i;

    std::ofstream constants("constants.txt");

    if(d.rank == 0) cout << "Calibration of Density..." << endl; 
    #ifdef USE_MPI
        domain.build(d.workload, d.x, d.y, d.z, d.h, d.bbox, clist, d.data, false);
        domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.m);
        domain.buildTree(d.bbox, d.x, d.y, d.z, d.h);
        d.count = clist.size();
    #else
        domain.build(clist, d.x, d.y, d.z, d.h, d.bbox);
    #endif
    domain.findNeighbors(clist, d.bbox, d.x, d.y, d.z, d.h, d.neighbors, d.neighborsCount);
    density.compute(clist, d);

    #pragma omp parallel for
    for(int pi=0; pi<(int)clist.size(); pi++)
        d.ro_0[clist[pi]] = d.ro[clist[pi]];

    for(int iteration = 0; iteration <= maxStep; iteration++)
    {
        timer::TimePoint start = timer::Clock::now();

        if(d.rank == 0) cout << "Iteration: " << iteration << endl;

        #ifdef USE_MPI
            REPORT_TIME(d.rank, domain.build(d.workload, d.x, d.y, d.z, d.h, d.bbox, clist, d.data, false), "domain::build");
            REPORT_TIME(d.rank, domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.m), "mpi::synchronizeHalos");
            REPORT_TIME(d.rank, domain.buildTree(d.bbox, d.x, d.y, d.z, d.h), "BuildTree");
            d.count = clist.size();
            if(d.rank == 0) cout << "# mpi::clist.size: " << clist.size() << " halos: " << domain.haloCount << endl;
        #else
            REPORT_TIME(d.rank, domain.build(clist, d.x, d.y, d.z, d.h, d.bbox), "BuildTree");
        #endif

        // REPORT_TIME(d.rank, mpi.reorder(d.data), "ReorderParticles");
        REPORT_TIME(d.rank, domain.findNeighbors(clist, d.bbox, d.x, d.y, d.z, d.h, d.neighbors, d.neighborsCount), "FindNeighbors");
        REPORT_TIME(d.rank, density.compute(clist, d), "Density");
        REPORT_TIME(d.rank, equationOfState.compute(clist, d.ro_0, d.p_0, d.ro, d.p, d.u, d.c), "EquationOfState");
        
        #ifdef USE_MPI
            d.resize(d.count); // Discard old neighbors
            REPORT_TIME(d.rank, domain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c), "mpi::synchronizeHalos");
        #endif

        REPORT_TIME(d.rank, momentumEnergy.compute(clist, d), "MomentumEnergy");
        REPORT_TIME(d.rank, timestep.compute(clist, d.h, d.c, d.dt_m1, d.dt, d.ttot), "Timestep");
        REPORT_TIME(d.rank, updateQuantities.compute(clist, d.grad_P_x, d.grad_P_y, d.grad_P_z, d.dt, d.du, d.bbox, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.u, d.du_m1, d.dt_m1), "UpdateQuantities");
        REPORT_TIME(d.rank, energyConservation.compute(clist, d.u, d.vx, d.vy, d.vz, d.m, d.etot, d.ecin, d.eint), "EnergyConservation");
        //REPORT_TIME(d.rank, domain.updateSmoothingLength(clist, d.neighbors, d.h), "SmoothingLength");

        long long int totalNeighbors = domain.neighborsSum(clist, d.neighborsCount);
        if(d.rank == 0)
        {
            cout << "### Check ### Computational domain: ";
            cout << d.bbox.xmin << " " << d.bbox.xmax << " ";
            cout << d.bbox.ymin << " " << d.bbox.ymax << " ";
            cout << d.bbox.zmin << " " << d.bbox.zmax << endl;
            cout << "### Check ### Avg neighbor count per particle: " << totalNeighbors/d.n << endl;
            cout << "### Check ### Total time: " << d.ttot << ", current time-step: " << d.dt[0] << endl;
            cout << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", cinetic: " << d.ecin << ")" << endl;
        }

        if(writeFrequency > 0 && iteration % writeFrequency == 0)
        {
            std::ofstream dump("dump" + to_string(iteration) + ".txt");
            REPORT_TIME(d.rank, d.writeData(clist, dump), "writeFile");
            dump.close();
        }

        d.writeConstants(iteration, totalNeighbors, constants);

        timer::TimePoint stop = timer::Clock::now();
        
        if(d.rank == 0) cout << "=== Total time for iteration " << timer::duration(start, stop) << "s" << endl << endl;
    }

    constants.close();

    #ifdef USE_MPI
        MPI_Finalize();
    #endif

    return 0;
}

