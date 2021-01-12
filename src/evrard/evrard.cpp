#include <fstream>
#include <string>

#include "sphexa.hpp"
#include "EvrardCollapseInputFileReader.hpp"
#include "EvrardCollapseFileWriter.hpp"

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

    const size_t maxStep = parser.getInt("-s", 10);
    const size_t nParticles = parser.getInt("-n", 65536);
    const int writeFrequency = parser.getInt("-w", -1);
    const int checkpointFrequency = parser.getInt("-c", -1);
    const bool quiet = parser.exists("--quiet");
    const bool timeRemoteGravitySteps = parser.exists("--timeRemoteGravity");
    const std::string checkpointInput = parser.getString("--cinput");
    const std::string inputFilePath = parser.getString("--input", "bigfiles/Test3DEvrardRel.bin");
    const std::string outDirectory = parser.getString("--outDir");

    std::ofstream nullOutput("/dev/null");
    std::ostream &output = quiet ? nullOutput : std::cout;

    using Real = double;
    using Dataset = ParticlesDataEvrard<Real>;
    using Tree = GravityOctree<Real>;

#ifdef USE_MPI
    DistributedDomain<Real, Dataset, Tree> domain;
    const IFileReader<Dataset> &fileReader = EvrardCollapseMPIInputFileReader<Dataset>();
    const IFileWriter<Dataset> &fileWriter = EvrardCollapseMPIFileWriter<Dataset>();
#else
    Domain<Real, Dataset, Tree> domain;
    const IFileReader<Dataset> &fileReader = EvrardCollapseInputFileReader<Dataset>();
    const IFileWriter<Dataset> &fileWriter = EvrardCollapseFileWriter<Dataset>();
#endif

    auto d = checkpointInput.empty() ? fileReader.readParticleDataFromBinFile(inputFilePath, nParticles)
                                     : fileReader.readParticleDataFromCheckpointBinFile(checkpointInput);

    const Printer<Dataset> printer(d);

    MasterProcessTimer timer(output, d.rank), totalTimer(output, d.rank);

    std::ofstream constantsFile(outDirectory + "constants.txt");

    Tree::bucketSize = 1;
    Tree::minGlobalBucketSize = 512;
    Tree::maxGlobalBucketSize = 2048;
    // Tree::minGlobalBucketSize = 1;
    // Tree::maxGlobalBucketSize = 1;

    domain.create(d);

    const size_t nTasks = 64;
    const size_t ng0 = 100;
    const size_t ngmax = 150;
    TaskList taskList = TaskList(domain.clist, nTasks, ngmax, ng0);
    using namespace std::chrono_literals;

    totalTimer.start();
    for (d.iteration = 0; d.iteration <= maxStep; d.iteration++)
    {
        timer.start();
        domain.update(d);
        timer.step("domain::update");
        domain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h, &d.m);
        timer.step("mpi::synchronizeHalos");
        domain.buildTree(d);
        timer.step("BuildTree");
        domain.octree.buildGlobalGravityTree(d.x, d.y, d.z, d.m);
        timer.step("BuildGlobalGravityTree");
        taskList.update(domain.clist);
        timer.step("updateTasks");
        auto rankToParticlesForRemoteGravCalculations = sph::gravityTreeWalk(taskList.tasks, domain.octree, d);
        timer.step("Gravity (self)");
        sph::remoteGravityTreeWalks<Real>(domain.octree, d, rankToParticlesForRemoteGravCalculations, timeRemoteGravitySteps);
        timer.step("Gravity (remote contributions)");
#ifdef USE_MPI
        // This barrier is only just to check how imbalanced gravity is.
        // Can be removed safely if not needed.
        MPI_Barrier(d.comm);
        timer.step("Gravity (remote contributions) Barrier");
#endif
        sph::findNeighbors(domain.octree, taskList.tasks, d);
        timer.step("FindNeighbors");
        sph::computeDensity<Real>(taskList.tasks, d);
        if (d.iteration == 0) { sph::initFluidDensityAtRest<Real>(taskList.tasks, d); }
        timer.step("Density");
        sph::computeEquationOfStateEvrard<Real>(taskList.tasks, d);
        timer.step("EquationOfState");
        domain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c);
        timer.step("mpi::synchronizeHalos");
        sph::computeIAD<Real>(taskList.tasks, d);
        timer.step("IAD");
        domain.synchronizeHalos(&d.c11, &d.c12, &d.c13, &d.c22, &d.c23, &d.c33);
        timer.step("mpi::synchronizeHalos");
        sph::computeMomentumAndEnergyIAD<Real>(taskList.tasks, d);
        timer.step("MomentumEnergyIAD");
        sph::computeTimestep<Real, sph::TimestepKCourant<Real, Dataset>>(taskList.tasks, d);
        timer.step("Timestep"); // AllReduce(min:dt)
        sph::computePositions<Real, sph::computeAccelerationWithGravity<Real, Dataset>, Dataset>(taskList.tasks, d);
        timer.step("UpdateQuantities");
        sph::computeTotalEnergyWithGravity<Real>(taskList.tasks, d);
        timer.step("EnergyConservation"); // AllReduce(sum:ecin,ein)
        sph::updateSmoothingLength<Real>(taskList.tasks, d);
        timer.step("UpdateSmoothingLength"); // AllReduce(sum:ecin,ein)

        const size_t totalNeighbors = sph::neighborsSum(taskList.tasks);
        if (d.rank == 0)
        {
            // printer.printRadiusAndGravityForce(domain.clist, fxFile);
            // printer.printTree(domain.octree, treeFile);
            printer.printCheck(d.count, domain.octree.globalNodeCount, d.x.size() - d.count, totalNeighbors, output);
            printer.printConstants(d.iteration, totalNeighbors, constantsFile);
        }

        if ((writeFrequency > 0 && d.iteration % writeFrequency == 0) || writeFrequency == 0)
        {
            fileWriter.dumpParticleDataToAsciiFile(d, domain.clist, outDirectory + "dump_evrard" + std::to_string(d.iteration) + ".txt");
            timer.step("writeFile");
        }
        if (checkpointFrequency > 0 && d.iteration % checkpointFrequency == 0)
        {
            fileWriter.dumpCheckpointDataToBinFile(d, outDirectory + "checkpoint_evrard" + std::to_string(d.iteration) + ".bin");
            timer.step("Save Checkpoint File");
        }

        timer.stop();

        if (d.rank == 0) printer.printTotalIterationTime(timer.duration(), output);
    }

    totalTimer.step("Total execution time of " + std::to_string(maxStep) + " iterations of Evrard Collapse");

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
        printf("\t-n NUM \t\t\t NUM Number of particles\n");
        printf("\t-s NUM \t\t\t NUM Number of iterations (time-steps)\n");
        printf("\t-w NUM \t\t\t Dump particles data every NUM iterations (time-steps)\n");
        printf("\t-c NUM \t\t\t Create checkpoint every NUM iterations (time-steps)\n\n");

        printf("\t--quiet \t\t Don't print anything to stdout\n");
        printf("\t--timeRemoteGravity \t Print times of gravity treewalk steps for each node\n\n");

        printf("\t--input PATH \t\t Path to input file\n");
        printf("\t--cinput PATH \t\t Path to checkpoint input file\n");
        printf("\t--outDir PATH \t\t Path to directory where output will be saved.\
                    \n\t\t\t\t Note that directory must exist and be provided with ending slash.\
                    \n\t\t\t\t Example: --outDir /home/user/folderToSaveOutputFiles/\n");
    }
}
