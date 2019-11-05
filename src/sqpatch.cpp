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
    const int cubeSide = parser.getInt("-n", 50);
    const int maxStep = parser.getInt("-s", 10);
    const int writeFrequency = parser.getInt("-w", -1);

#ifdef _JENKINS
    maxStep = 0;
    writeFrequency = -1;
#endif

    using Real = double;
    using Dataset = ParticlesData<Real>;

#ifdef USE_MPI
    MPI_Init(NULL, NULL);
#endif

    auto d = SqPatchDataGenerator<Real>::generate(cubeSide);
    DistributedDomain<Real> distributedDomain;
    Printer<Dataset> printer(d);
    MPITimer timer(d.rank);

    std::ofstream constantsFile("constants.txt");

    std::vector<Task> taskList;

    // Easiest way to go back to one task. Every loop add this after buildTree:
    // Task bigTask(distributedDomain.clist.size());
    // bigTask.clist = distributedDomain.clist;
    // taskList.clear();
    // taskList.push_back(bigTask);

    distributedDomain.create(d);

    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();

        distributedDomain.distribute(d);
        timer.step("domain::distribute");
        distributedDomain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h);
        timer.step("mpi::synchronizeHalos");
        distributedDomain.buildTree(d);
        timer.step("domain::buildTree");
        distributedDomain.createTasks(taskList, 480);
        timer.step("domain::createTasks");

        // distributedDomain.findNeighbors(taskList, d);
        // timer.step("FindNeighbors");

#pragma omp parallel
#pragma omp single
        {
            for (auto &task : taskList)
            {
#pragma omp task
                {
                    distributedDomain.findNeighborsImpl(task, d);
                    timer.step("FindNeighbors");
                    sph::computeDensityImpl<Real>(task, d);
                    if (d.iteration == 0) { sph::initFluidDensityAtRestImpl<Real>(task, d); }
                    timer.step("Density");
                    sph::computeEquationOfStateImpl<Real>(task, d);
                    timer.step("EquationOfState");
                }
            }
        }
        // sph::computeDensity<Real>(taskList, d);
        // if (d.iteration == 0) { sph::initFluidDensityAtRest<Real>(taskList, d); }
        // timer.step("Density");
        // sph::computeEquationOfState<Real>(taskList, d);
        // timer.step("EquationOfState");

        distributedDomain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c);
        timer.step("mpi::synchronizeHalos");
        sph::computeIAD<Real>(taskList, d);
        timer.step("IAD");
        distributedDomain.synchronizeHalos(&d.c11, &d.c12, &d.c13, &d.c22, &d.c23, &d.c33);
        timer.step("mpi::synchronizeHalos");
        sph::computeMomentumAndEnergyIAD<Real>(taskList, d);
        timer.step("MomentumEnergyIAD");
        sph::computeTimestep<Real>(taskList, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        sph::computePositions<Real>(taskList, d);
        timer.step("UpdateQuantities");
        sph::computeTotalEnergy<Real>(taskList, d);
        timer.step("EnergyConservation"); // AllReduce(sum:ecin,ein)

        long long int totalNeighbors = distributedDomain.neighborsSum(taskList);
        if (d.rank == 0)
        {
            printer.printCheck(distributedDomain.clist.size(), distributedDomain.octree.globalNodeCount, distributedDomain.haloCount,
                               totalNeighbors, std::cout);
            printer.printConstants(d.iteration, totalNeighbors, constantsFile);
        }

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
            printer.printAllDataToFile(distributedDomain.clist, "dump" + to_string(d.iteration) + ".txt");
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
