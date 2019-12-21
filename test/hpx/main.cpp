#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "mpi.h"

#ifndef USE_MPI
#define USE_MPI
#endif

#ifndef USE_HPX
#define USE_HPX
#endif

#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_reduce.hpp>
//#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/util.hpp>
//#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>


#include <pthread.h>

#include "sphexa.hpp"
#include "SqPatchDataGenerator.hpp"

using namespace sphexa;


int hpx_main(boost::program_options::variables_map& vm)
{
    int cubeSide = vm["cubeside"].as<int>();
    int maxStep = vm["maxstep"].as<int>();
    int writeFrequency = vm["writefrequency"].as<int>();

    using Real = double;
    using Dataset = ParticlesData<Real>;

    DistributedDomain<Real, Dataset> domain;

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
            printer.printAllDataToFile(domain.clist, "dump" + std::to_string(d.iteration) + ".txt");
            timer.step("writeFile");
        }

        timer.stop();

        if (d.rank == 0) printer.printTotalIterationTime(timer.duration(), std::cout);
    }

    constantsFile.close();

    return hpx::finalize();
}

namespace po = boost::program_options;

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);

    po::options_description
        desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("cubeside,n", po::value<int>()->default_value(50),
            "number of particles per cube side")
        ("maxstep,s", po::value<int>()->default_value(10),
            "number of SPH iterations to be performed")
        ("writefrequency,w", po::value<int>()->default_value(-1),
            "write output every \"w\" steps")
        ;

    hpx::init(desc_commandline, argc, argv);

    MPI_Finalize();
}
