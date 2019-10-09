#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "sphexa.hpp"
#include "SqPatch.hpp"

#include "gnuplot.hpp"

using namespace std;
using namespace sphexa;

int main(int argc, char **argv)
{
    ArgParser parser(argc, argv);
    int cubeSide = parser.getInt("-n", 50);
    int maxStep = parser.getInt("-s", 10);
    int writeFrequency = parser.getInt("-w", -1);

#ifdef USE_GNUPLOT
    FILE *gp; gnuplot_init(gp);
#endif

#ifdef _JENKINS
    maxStep = 0;
    writeFrequency = -1;
#endif

    typedef double Real;
    typedef SqPatch<Real> Dataset;

#ifdef USE_MPI
    MPI_Init(NULL, NULL);
#endif

    Dataset d(cubeSide);
    DistributedDomain<Real> distributedDomain;
    // Domain<Real, Tree> domain(d.ngmin, d.ng0, d.ngmax);

    vector<int> clist(d.count);
    for (unsigned int i = 0; i < clist.size(); i++)
        clist[i] = i;

    std::ofstream constants("constants.txt");

    //d.bbox.setBox(0, 0, 0, 0, d.bbox.zmin, d.bbox.zmax, false, false, true);
    d.bbox.setBox(0, 0, 0, 0, d.bbox.zmin, d.bbox.zmax, false, false, false);
    d.bbox.computeGlobal(clist, d.x, d.y, d.z);
    distributedDomain.create(clist, d);

    MPITimer timer(d.rank);
    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();

        #ifdef USE_GNUPLOT
            gnuplot_plot<double>(gp, clist, d);
        #endif

        d.bbox.computeGlobal(clist, d.x, d.y, d.z);
        distributedDomain.distribute(clist, d);
        //timer.step("domain::distribute");
        distributedDomain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h);
        //timer.step("mpi::synchronizeHalos");
        distributedDomain.buildTree(d);
        //timer.step("domain::buildTree");
        distributedDomain.findNeighbors(clist, d);
        //timer.step("FindNeighbors");
        sph::computeDensity<Real>(clist, d);
        if (d.iteration == 0) { sph::initFluidDensityAtRest<Real>(clist, d); }
        //timer.step("Density");
        sph::computeEquationOfState<Real>(clist, d);
        //timer.step("EquationOfState");
        distributedDomain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c);
        //timer.step("mpi::synchronizeHalos");
        sph::computeIAD<Real>(clist, d);
        //timer.step("IAD");
        distributedDomain.synchronizeHalos(&d.c11, &d.c12, &d.c13, &d.c22, &d.c23, &d.c33);
        //timer.step("mpi::synchronizeHalos");
        sph::computeMomentumAndEnergyIAD<Real>(clist, d);
        //timer.step("MomentumEnergyIAD");
        sph::computeTimestep<Real>(clist, d);
        //timer.step("Timestep"); // AllReduce(min:dt)
        sph::computePositions<Real>(clist, d);
        //timer.step("UpdateQuantities");
        sph::computeTotalEnergy<Real>(clist, d);
        //timer.step("EnergyConservation"); // AllReduce(sum:ecin,ein)

        long long int totalNeighbors = distributedDomain.neighborsSum(clist, d);
        // if (d.rank == 0)
        // {
        //     cout << "### Check ### Global Tree Nodes: " << distributedDomain.octree.globalNodeCount << ", Particles: " << clist.size()
        //          << ", Halos: " << distributedDomain.haloCount << endl;
        //     cout << "### Check ### Computational domain: " << d.bbox.xmin << " " << d.bbox.xmax << " " << d.bbox.ymin << " " << d.bbox.ymax
        //          << " " << d.bbox.zmin << " " << d.bbox.zmax << endl;
        //     cout << "### Check ### Total neighbors " << totalNeighbors << ", Avg count per particle: " << totalNeighbors / d.n << endl;
        //     cout << "### Check ### Total time: " << d.ttot << ", current time-step: " << d.minDt << endl;
        //     cout << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", cinetic: " << d.ecin << ")" << endl;
        // }

        if (writeFrequency > 0 && d.iteration % writeFrequency == 0)
        {
            d.writeData(clist, "dump" + to_string(d.iteration) + ".txt");
            timer.step("writeFile");
        }
        d.writeConstants(d.iteration, totalNeighbors, constants);

        timer.stop();
        if (d.rank == 0) cout << "=== Total time for iteration(" << d.iteration << ") " << timer.duration() << "s" << endl;// << endl;

        #ifdef USE_GNUPLOT
            gnuplot_wait();
        #endif
    }

    constants.close();

#ifdef USE_MPI
    MPI_Finalize();
#endif

#ifdef USE_GNUPLOT
    gnuplot_destroy(gp);
#endif
    
    return 0;
}
