#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "sphexa.hpp"
#include "SedovDataGenerator.hpp"
#include "SedovDataFileWriter.hpp"

using namespace sphexa;

void printHelp(char *binName, int rank);

int main(int argc, char **argv)
{
    const int rank = initAndGetRankId();

    const ArgParser parser(argc, argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printHelp(argv[0], rank);
        return exitSuccess();
    }

    const size_t cubeSide = parser.getInt("-n", 50);
    const size_t maxStep = parser.getInt("-s", 10);
    const int writeFrequency = parser.getInt("-w", -1);
    const bool quiet = parser.exists("--quiet");
    const std::string outDirectory = parser.getString("--outDir");

    std::ofstream nullOutput("/dev/null");
    std::ostream &output = quiet ? nullOutput : std::cout;

    using Real = double;
    using Dataset = ParticlesData<Real>;
    using Tree = Octree<Real>;

#ifdef USE_MPI
    DistributedDomain<Real, Dataset, Tree> domain;
    const IFileWriter<Dataset> &fileWriter = SedovMPIFileWriter<Dataset>();
#else
    Domain<Real, Dataset, Tree> domain;
    const IFileWriter<Dataset> &fileWriter = SedovFileWriter<Dataset>();
#endif

    auto d = SedovDataGenerator<Real>::generate(cubeSide);
    const Printer<Dataset> printer(d);

    if(d.rank == 0) std::cout << "Data generated." << std::endl;

    MasterProcessTimer timer(output, d.rank), totalTimer(output, d.rank);

    std::ofstream constantsFile(outDirectory + "constants.txt");

    // -n 350, 42M per node
    Tree::bucketSize = 64;
    Tree::minGlobalBucketSize = std::pow(2,12);
    Tree::maxGlobalBucketSize = std::pow(2,14);//1048576;
    domain.create(d);

    if(d.rank == 0) std::cout << "Domain created." << std::endl;

    const size_t nTasks = 256;
    const size_t ngmax = 150;
    const size_t ng0 = 100;
    TaskList taskList = TaskList(domain.clist, nTasks, ngmax, ng0);

    if(d.rank == 0) std::cout << "Starting main loop." << std::endl;

    totalTimer.start();
    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();

        domain.update(d);
        timer.step("domain::distribute");
        domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h);
        timer.step("mpi::synchronizeHalos");
        domain.buildTree(d);
        timer.step("domain::buildTree");
        taskList.update(domain.clist);
        timer.step("updateTasks");
        sph::findNeighbors(domain.octree, taskList.tasks, d);
        timer.step("FindNeighbors");
        if(domain.clist.size() > 0) sph::computeDensity<Real>(domain.octree, taskList.tasks, d);
        timer.step("Density");
        sph::computeEquationOfStateEvrard<Real>(taskList.tasks, d);
        timer.step("EquationOfState");
        domain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c);
        timer.step("mpi::synchronizeHalos");
        if(domain.clist.size() > 0) sph::computeIAD<Real>(domain.octree, taskList.tasks, d);
        timer.step("IAD");
        domain.synchronizeHalos(&d.c11, &d.c12, &d.c13, &d.c22, &d.c23, &d.c33);
        timer.step("mpi::synchronizeHalos");
        if(domain.clist.size() > 0) sph::computeMomentumAndEnergyIAD<Real>(domain.octree, taskList.tasks, d);
        timer.step("MomentumEnergyIAD");
        sph::computeTimestep<Real, sph::TimestepPress2ndOrder<Real, Dataset>>(taskList.tasks, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        sph::computePositions<Real, sph::computeAcceleration<Real, Dataset>>(taskList.tasks, d);
        timer.step("UpdateQuantities");
        sph::computeTotalEnergy<Real>(taskList.tasks, d);
        timer.step("EnergyConservation"); // AllReduce(sum:ecin,ein)
        sph::updateSmoothingLength<Real>(taskList.tasks, d);
        timer.step("UpdateSmoothingLength");

        const size_t totalNeighbors = sph::neighborsSum(taskList.tasks);

        if (d.rank == 0)
        {
            printer.printCheck(d.count, domain.octree.globalNodeCount, d.x.size() - d.count, totalNeighbors, output);
            printer.printConstants(d.iteration, totalNeighbors, constantsFile);
        }

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
            //fileWriter.dumpParticleDataToAsciiFile(d, domain.clist, outDirectory + "dump_Sedov" + std::to_string(d.iteration) + ".txt");
            fileWriter.dumpParticleDataToBinFile(d, outDirectory + "dump_Sedov" + std::to_string(d.iteration) + ".bin");
            timer.step("writeFile");
        }

        timer.stop();

        if (d.rank == 0) printer.printTotalIterationTime(timer.duration(), output);
    }

    totalTimer.step("Total execution time of " + std::to_string(maxStep) + " iterations of Sedov");

    constantsFile.close();

    return exitSuccess();
}

void printHelp(char *name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n");
        printf("\t-n NUM \t\t\t NUM^3 Number of particles\n");
        printf("\t-s NUM \t\t\t NUM Number of iterations (time-steps)\n");
        printf("\t-w NUM \t\t\t Dump particles data every NUM iterations (time-steps)\n\n");

        printf("\t--quiet \t\t Don't print anything to stdout\n\n");

        printf("\t--outDir PATH \t\t Path to directory where output will be saved.\
                    \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                    \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n");
    }
}
