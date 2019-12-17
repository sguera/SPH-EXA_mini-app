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

    auto d = SqPatchDataGenerator<Real>::generate(cubeSide);
    DistributedDomain<Real, Dataset> domain;

    MasterProcessTimer timer(d.rank);

    domain.create(d);

    //for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    //{
        timer.start();

        domain.update(d);
        timer.step("domain::distribute");
        domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h);
        timer.step("mpi::synchronizeHalos");
        domain.buildTreeInc(d);
        timer.step("domain::buildTree");
    //}


    //std::vector<std::vector<Real> *> data;
    //makeDataArray(data, &d.x, &d.y, &d.z);

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
