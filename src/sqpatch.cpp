#include <iostream>
#include <fstream>
#include <string>
#include <functional>
#include <chrono>
#include <thread>

//#include <boost/program_options.hpp>

#include <hpx/hpx_init.hpp>
//#include <hpx/hpx_start.hpp>
#include <hpx/hpx_suspend.hpp>
#include <hpx/include/parallel_algorithm.hpp>
//#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/util.hpp>
//#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>
//#include <hpx/include/lcos.hpp>

#include "sphexa.hpp"
#include "SqPatch.hpp"

using namespace std;
using namespace sphexa;

int hpx_main(boost::program_options::variables_map& vm)
{
    int cubeSide = vm["cubeside"].as<int>();
    int maxStep = vm["maxstep"].as<int>();
    int writeFrequency = vm["writefrequency"].as<int>();

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
    DistributedDomain<Real> distributedDomain;
    Domain<Real, Tree> domain(d.ngmin, d.ng0, d.ngmax);

    vector<int> clist(d.count);
    for (unsigned int i = 0; i < clist.size(); i++)
        clist[i] = i;

    std::ofstream constants("constants.txt");

    MPITimer timer(d.rank);
    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();
        d.resize(d.count); // Discard halos
        distributedDomain.distribute(clist, d);
        timer.step("domain::build");
        distributedDomain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.m);
        timer.step("mpi::synchronizeHalos");

        //domain.buildTree(d);
        hpx::future<void> f = hpx::async([](Domain<Real, Tree>& dom_, Dataset const& data_) -> void
                                         { dom_.buildTree(data_); }, std::ref(domain), std::cref(d));
        f.get();

        timer.step("BuildTree");
        domain.findNeighbors(clist, d);
        timer.step("FindNeighbors");

        sph::computeDensity<Real>(clist, d);
        timer.step("Density");
        sph::computeEquationOfState<Real>(clist, d);
        timer.step("EquationOfState");

        distributedDomain.resizeArrays(d.count, &d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c); // Discard halos
        distributedDomain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c);
        timer.step("mpi::synchronizeHalos");

        sph::computeMomentumAndEnergy<Real>(clist, d);
        timer.step("MomentumEnergy");
        sph::computeTimestep<Real>(clist, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        sph::computePositions<Real>(clist, d);
        timer.step("UpdateQuantities");
        sph::computeTotalEnergy<Real>(clist, d);
        timer.step("EnergyConservation"); // AllReduce(sum:ecin,ein)

        long long int totalNeighbors = domain.neighborsSum(clist, d);
        if (d.rank == 0)
        {
            cout << "### Check ### Particles: " << clist.size() << ", Halos: " << distributedDomain.haloCount << endl;
            cout << "### Check ### Computational domain: " << d.bbox.xmin << " " << d.bbox.xmax << " " << d.bbox.ymin << " " << d.bbox.ymax
                 << " " << d.bbox.zmin << " " << d.bbox.zmax << endl;
            cout << "### Check ### Avg neighbor count per particle: " << totalNeighbors / d.n << endl;
            cout << "### Check ### Total time: " << d.ttot << ", current time-step: " << d.dt[0] << endl;
            cout << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", cinetic: " << d.ecin << ")" << endl;
        }

        if (writeFrequency > 0 && d.iteration % writeFrequency == 0)
        {
            std::ofstream dump("dump" + to_string(d.iteration) + ".txt");
            d.writeData(clist, dump);
            timer.step("writeFile");
            dump.close();
        }
        d.writeConstants(d.iteration, totalNeighbors, constants);

        timer.stop();
        if (d.rank == 0) cout << "=== Total time for iteration(" << d.iteration << ") " << timer.duration() << "s" << endl << endl;
    }

    constants.close();

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return hpx::finalize();
}

namespace po = boost::program_options;

int main(int argc, char ** argv)
{
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

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc_commandline), vm);

    return hpx::init(desc_commandline, argc, argv);
}
