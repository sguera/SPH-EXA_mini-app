#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "sphexa.hpp"
#include "SqPatchDataGenerator.hpp"

using namespace std;
using namespace sphexa;

int main(int argc, char **argv)
{
    ArgParser parser(argc, argv);
    const size_t cubeSide = parser.getInt("-n", 50);
    const size_t maxStep = parser.getInt("-s", 10);
    const int writeFrequency = parser.getInt("-w", -1);

    using Real = double;
    using Dataset = ParticlesData<Real>;

#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    DistributedDomain<Real, Dataset> domain;
#else
    Domain<Real, Dataset> domain;
#endif

    auto d = SqPatchDataGenerator<Real>::generate(cubeSide);
    Printer<Dataset> printer(d);
    MasterProcessTimer timer(d.rank);

    std::ofstream constantsFile("constants.txt");

    std::vector<Task> taskList;

    domain.create(d);

    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();

        domain.update(d);
        timer.step("domain::distribute");
        domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h);
        timer.step("mpi::synchronizeHalos");
        domain.buildTree(d);
        timer.step("domain::buildTree");
        domain.createTasks(taskList, 64);
        timer.step("domain::createTasks");
        sph::findNeighbors(domain.octree, taskList, d);
        timer.step("FindNeighbors");
        sph::computeDensity<Real>(taskList, d);
        if (d.iteration == 0) { sph::initFluidDensityAtRest<Real>(taskList, d); }
        timer.step("Density");
        sph::computeEquationOfState<Real>(taskList, d);
        timer.step("EquationOfState");
        domain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c);
        timer.step("mpi::synchronizeHalos");
        sph::computeIAD<Real>(taskList, d);
        timer.step("IAD");
        domain.synchronizeHalos(&d.c11, &d.c12, &d.c13, &d.c22, &d.c23, &d.c33);
        timer.step("mpi::synchronizeHalos");
        sph::computeMomentumAndEnergyIAD<Real>(taskList, d);
        timer.step("MomentumEnergyIAD");
        sph::computeTimestep<Real>(taskList, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        sph::computePositions<Real>(taskList, d);
        timer.step("UpdateQuantities");
        sph::computeTotalEnergy<Real>(taskList, d);
        timer.step("EnergyConservation"); // AllReduce(sum:ecin,ein)
        sph::updateSmoothingLength<Real>(taskList, d);
        timer.step("UpdateSmoothingLength");

        size_t totalNeighbors = sph::neighborsSum(taskList);
        if (d.rank == 0)
        {
            printer.printCheck(d.count, domain.octree.globalNodeCount, d.x.size() - d.count, totalNeighbors, std::cout);
            printer.printConstants(d.iteration, totalNeighbors, constantsFile);
        }

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
            printer.printAllDataToFile(domain.clist, "dump" + to_string(d.iteration) + ".txt");
            timer.step("writeFile");
        }

        timer.stop();

        if (d.rank == 0) printer.printTotalIterationTime(timer.duration(), std::cout);
    }

    constantsFile.close();

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
