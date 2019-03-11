#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>

#include "sphexa.hpp"
#include "debug.hpp"

#include "SqPatch.hpp"

using namespace std;
using namespace sphexa;

#define REPORT_TIME(rank, expr, name) \
    if (rank == 0) sphexa::timer::report_time([&](){ expr; }, name); \
    else { expr; }

void gnuplot_init(FILE *&gp)
{
    gp = popen("gnuplot -persistent", "w");
    //fprintf(gp, "set pm3d map;\n");
    //fprintf(gp, "set terminal wxt size 600,600\n");
    //fprintf(gp, "set cbrange [300.0:700.0];\n");
    //fprintf(gp, "set palette defined ( 0 \"black\", 1 \"red\", 2 \"yellow\", 3 \"white\" );\n");
}

void gnuplot_destroy(FILE *gp)
{
    fprintf(gp, "exit\n");
    pclose(gp);
}

template<typename T>
void gnuplot_plot(FILE *gp, const std::vector<int> &clist, const BBox<T> &bbox, const vector<T> &x, const vector<T> &y, const vector<T> &z, const vector<T> &c)
{
    // //int n = ceil(pow(x.size(), 1.0/3.0));

    // int n = ceil(pow(x.size(), 1.0/3.0));
    // n *= n;

    // double xy[n*3];
    // for(int i=0; i<n; i++)
    // {
    //     xy[i*3] = x[i];
    //     xy[i*3+1] = y[i];
    //     xy[i*3+2] = z[i];
    // }

    // //fprintf(gp, "\n");
    // // fprintf(gp, "set xrange [%f:%f];\n", bbox.xmin, bbox.xmax);
    // // fprintf(gp, "set yrange [%f:%f];\n", bbox.ymin, bbox.ymax);
    // fprintf(gp, "plot '-' binary format='%%double%%double%%double' record=%d u 1:2:3 pt 7 ps 1 lc palette z\n", n);
    // fwrite(xy, sizeof(double), 3*n, gp);
    // //fwrite(&y[0], sizeof(double), n, gp);
    // //fwrite(&data[0][1], sizeof(double), n, gp);
    // //fwrite(&data[0][2], sizeof(double), n, gp);
    // fprintf(gp, "\n");
    // fflush(gp);

    int n = clist.size();

    double xy[n*4];
    for(int i=0; i<n; i++)
    {
        xy[i*4] = x[clist[i]];
        xy[i*4+1] = y[clist[i]];
        xy[i*4+2] = z[clist[i]];
        xy[i*4+3] = c[clist[i]];
    }

    //fprintf(gp, "\n");
    fprintf(gp, "set xrange [%f:%f];\n", bbox.xmin, bbox.xmax);
    fprintf(gp, "set yrange [%f:%f];\n", bbox.ymin, bbox.ymax);
    fprintf(gp, "set zrange [%f:%f];\n", bbox.zmin, bbox.zmax);
    fprintf(gp, "set cbrange [-50.0:50.0];\n");
    fprintf(gp, "splot '-' binary format='%%double%%double%%double' record=%d u 1:2:3:4 pt 7 ps 1 lc palette z\n", n);
    fwrite(xy, sizeof(double), 4*n, gp);
    //fwrite(&y[0], sizeof(double), n, gp);
    //fwrite(&data[0][1], sizeof(double), n, gp);
    //fwrite(&data[0][2], sizeof(double), n, gp);
    fprintf(gp, "\n");
    fflush(gp);
}

int main(int argc, char **argv)
{
    FILE *gp; gnuplot_init(gp);

    ArgParser parser(argc, argv);

    int cubeSide = parser.getInt("-n", 100);
    int maxStep = parser.getInt("-s", 1e5);
    int writeFrequency = parser.getInt("-w", 250);
    //int targetNeighbors = parser.getInt("-nn", 500);
    //std::string inputFilename = parser.getString("-f", "bigfiles/squarepatch3D_1M.bin");

    #ifdef _JENKINS
        maxStep = 0;
        writeFrequency = -1;
    #endif

	//debug();

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

    // if(d.rank == 0)
    //     cout << "Total number of ranks: " << domain.comm_size << endl;
    
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
    domain.findNeighbors(clist, d.bbox, d.x, d.y, d.z, d.h, d.neighbors);
    density.compute(clist, d.bbox, d.neighbors, d.x, d.y, d.z, d.h, d.m, d.ro);

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
        REPORT_TIME(d.rank, domain.findNeighbors(clist, d.bbox, d.x, d.y, d.z, d.h, d.neighbors), "FindNeighbors");
        REPORT_TIME(d.rank, density.compute(clist, d.bbox, d.neighbors, d.x, d.y, d.z, d.h, d.m, d.ro), "Density");
        REPORT_TIME(d.rank, equationOfState.compute(clist, d.ro_0, d.p_0, d.ro, d.p, d.u, d.c), "EquationOfState");
        
        #ifdef USE_MPI
            d.resize(d.count); // Discard old neighbors
            REPORT_TIME(d.rank, domain.synchronizeHalos(&d.vx, &d.vy, &d.vz, &d.ro, &d.p, &d.c), "mpi::synchronizeHalos");
        #endif

        REPORT_TIME(d.rank, momentumEnergy.compute(clist, d.bbox, d.neighbors, d.x, d.y, d.z, d.h, d.vx, d.vy, d.vz, d.ro, d.p, d.c, d.m, d.grad_P_x, d.grad_P_y, d.grad_P_z, d.du), "MomentumEnergy");
        REPORT_TIME(d.rank, timestep.compute(clist, d.h, d.c, d.dt_m1, d.dt, d.ttot), "Timestep");
        REPORT_TIME(d.rank, updateQuantities.compute(clist, d.grad_P_x, d.grad_P_y, d.grad_P_z, d.dt, d.du, d.bbox, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.u, d.du_m1, d.dt_m1), "UpdateQuantities");
        REPORT_TIME(d.rank, energyConservation.compute(clist, d.u, d.vx, d.vy, d.vz, d.m, d.etot, d.ecin, d.eint), "EnergyConservation");
        //REPORT_TIME(d.rank, domain.updateSmoothingLength(clist, d.neighbors, d.h), "SmoothingLength");

        int totalNeighbors = domain.neighborsSum(clist, d.neighbors);

        if(d.rank == 0)
        {
            cout << "### Check ### Computational domain: ";
            cout << d.bbox.xmin << " " << d.bbox.xmax << " ";
            cout << d.bbox.ymin << " " << d.bbox.ymax << " ";
            cout << d.bbox.zmin << " " << d.bbox.zmax << endl;
            cout << "### Check ### Total number of neighbours: " << totalNeighbors/d.n << endl;
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
        
        gnuplot_plot(gp, clist, d.bbox, d.x, d.y, d.z, d.z);
    }

    constants.close();

    #ifdef USE_MPI
        MPI_Finalize();
    #endif

    gnuplot_destroy(gp);

    return 0;
}

