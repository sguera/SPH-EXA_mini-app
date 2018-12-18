#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>

#include "TaskScheduler.hpp"
#include "TaskLoop.hpp"
#include "DistributedDataset.hpp"
#include "DistributedDomain.hpp"
#include "tree/Octree.hpp"
#include "tree/HTree.hpp"
#include "BBox.hpp"
#include "Density.hpp"
#include "EOS.hpp"
#include "Momentum.hpp"
#include "Energy.hpp"
#include "Timestep.hpp"
#include "H.hpp"
#include "UpdateQuantities.hpp"

using namespace std;
using namespace sphexa;

class LambdaTask : public Task
{
public:
	LambdaTask(const std::function<void()> func) : func(func) {}
	virtual void compute() { func(); }
	const std::function<void()> func; 
};

class LambdaTaskLoop : public TaskLoop
{
public:
    LambdaTaskLoop(const std::vector<int> &computeList, const std::function<void(int i)> func) : TaskLoop(computeList), func(func) {}
    virtual void compute(int i) { func(i); }
    const std::function<void(int i)> func; 
};

int main()
{
    MPI_Init(NULL, NULL);

    // Dataset: contains arrays (x, y, z, vx, vy, vz, ro, u, p, h, m, temp, mue, mui)
    DistributedDataset d(1e6, "bigfiles/Evrard3D.bin", MPI_COMM_WORLD);

    // This is the right way...
    DistributedDomain<DistributedDataset> domain(MPI_COMM_WORLD, d);

    Octree<double> tree(d.x, d.y, d.z, d.h, Octree<double>::Params(/*max neighbors*/d.ngmax, /*bucketSize*/128));
    //HTree<double> tree(d.x, d.y, d.z, d.h, HTree<double>::Params(/*max neighbors*/d.ngmax, /*bucketSize*/128));

    LambdaTask tbalance([&]()
    {
        d.resize(d.computeList.size());
        domain.build(d.workload, d.computeList);
        domain.synchronize();
        domain.discardHalos();

        d.computeList.resize(d.size());
        for(unsigned int i=0; i<d.computeList.size(); i++)
            d.computeList[i] = i;

        domain.build(d.workload, d.computeList);
        domain.synchronizeHalos(d.computeList, true);

        d.neighbors.resize(d.size());
        d.computeBBox();

        if(d.comm_rank == 0) cout << "ComputeList.size: " << d.computeList.size() << ", Halos: " << domain.haloCount << endl;
    });

    LambdaTask tsynchronizeHalos([&]()
    {
        d.resize(d.computeList.size());
        domain.synchronizeHalos(d.computeList, false);
    });

    LambdaTask tbuild([&]()
    {
        d.computeBBox();
        tree.build(d.bbox);
    });

    // LambdaTask treorder([&]()
    // {
    //     d.reorder(*tree.ordering);
    //     // Assuming computeList start from 0 before reordering (true at the moment)
    //     int j = 0;
    //     std::vector<int> tmp(d.computeList.size());
    //     for(unsigned int i=0; i<(*tree.ordering).size(); i++)
    //     {
    //         if((*tree.ordering)[i] < (int)d.computeList.size())
    //             tmp[j++] = i;
    //     }
    //     tmp.swap(d.computeList);

    //     for(unsigned int i=0; i<(*tree.ordering).size(); i++)
    //     	(*tree.ordering)[i] = i;
    // });

    LambdaTaskLoop tfind(d.computeList, [&](int i)
    {
        d.neighbors[i].resize(0);
        tree.findNeighbors(d.x[i], d.y[i], d.z[i], 2*d.h[i], d.neighbors[i]);
    });

    LambdaTask tprintBBox([&]()
    {
        if(d.comm_rank == 0)
        {
            cout << "### Check ### Computational domain: ";
            cout << d.bbox.xmin << " " << d.bbox.xmax << " ";
            cout << d.bbox.ymin << " " << d.bbox.ymax << " ";
            cout << d.bbox.zmin << " " << d.bbox.zmax << endl;
        }
    });

    LambdaTask tcheckNeighbors([&]()
    { 
        int sum = 0;
        for(unsigned int i=0; i<d.computeList.size(); i++)
            sum += d.neighbors[d.computeList[i]].size();

        MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if(d.comm_rank == 0) cout << "### Check ### Total number of neighbours: " << sum << endl;
    });

    Density<double> tdensity(d.computeList, d.x, d.y, d.z, d.h, d.m, d.neighbors, d.ro);
    EOS<double> teos(d.computeList, d.ro, d.u, d.mui, d.p, d.temp, d.c, d.cv);
    Momentum<double> tmomentum(d.computeList, d.x, d.y, d.z, d.h, d.vx, d.vy, d.vz, d.ro, d.p, d.c, d.m, d.neighbors, d.grad_P_x, d.grad_P_y, d.grad_P_z);
    Energy<double> tenergy(d.computeList, d.x, d.y, d.z, d.h, d.vx, d.vy, d.vz, d.ro, d.p, d.c, d.m, d.neighbors, d.du);
    Timestep<double> ttimestep(d.computeList, d.h, d.c, d.dt_m1, d.dt);

    LambdaTask tcheckTimestep([&]()
    {
        // All local dt are set to min after Timestep
        MPI_Allreduce(MPI_IN_PLACE, &d.dt[0], 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        for(unsigned int i=0; i<d.dt.size(); i++)
            d.dt[i] = d.dt[0];

        if(d.comm_rank == 0) cout << "### Check ### Old Time-step: " << d.dt_m1[0] << ", New time-step: " << d.dt[0] << endl;
    });

    H<double> tH(d.computeList, d.neighbors, d.h, H<double>::Params(/*Target No of neighbors*/100));
    UpdateQuantities<double> tupdate(d.computeList, d.grad_P_x, d.grad_P_y, d.grad_P_z, d.dt, d.du, d.iteration, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.u, d.du_m1, d.dt_m1);

    LambdaTask tcheckConservation([&]()
    { 
        double e_tot = 0.0, e_cin = 0.0, e_int = 0.0;

        for(unsigned int i=0; i<d.computeList.size(); i++)
        {
            int id = d.computeList[i];

            double vmod2 = 0.0;
            vmod2 = d.vx[id] * d.vx[id] + d.vy[id] * d.vy[id] + d.vz[id] * d.vz[id];
            e_cin += 0.5 * d.m[i] * vmod2;
            e_int += d.u[id] * d.m[id]; 
        }

        MPI_Allreduce(MPI_IN_PLACE, &e_cin, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &e_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        e_tot += e_cin + e_int;

        if(d.comm_rank == 0) cout << "### Check ### Total energy: " << e_tot << ", (internal: " << e_int << ", cinetic: " << e_cin << ")" << endl;
    });

    LambdaTask twriteFile([&]()
    {
        std::vector<double> x, y, z, vx, vy, vz, h, ro, u, p, c, grad_P_x, grad_P_y, grad_P_z;
        std::vector<int> workload(d.comm_size);

        int load = (int)d.computeList.size();
        MPI_Allgather(&load, 1, MPI_INT, &workload[0], 1, MPI_INT, MPI_COMM_WORLD);

        // if(d.comm_rank == 0)
        // {
        //     printf("Workloads:\n");
        //     for(int i=0; i<d.comm_size; i++)
        //         printf("%d ", workload[i]);
        //     printf("\n");
        // }

        std::vector<int> displs(d.comm_size);

        displs[0] = 0;
        for(int i=1; i<d.comm_size; i++)
            displs[i] = displs[i-1]+workload[i-1];

        x.resize(d.n); MPI_Gatherv(&d.x[0], (int)d.computeList.size(), MPI_DOUBLE, &x[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        y.resize(d.n); MPI_Gatherv(&d.y[0], (int)d.computeList.size(), MPI_DOUBLE, &y[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        z.resize(d.n); MPI_Gatherv(&d.z[0], (int)d.computeList.size(), MPI_DOUBLE, &z[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        vx.resize(d.n); MPI_Gatherv(&d.vx[0], (int)d.computeList.size(), MPI_DOUBLE, &vx[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        vy.resize(d.n); MPI_Gatherv(&d.vy[0], (int)d.computeList.size(), MPI_DOUBLE, &vy[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        vz.resize(d.n); MPI_Gatherv(&d.vz[0], (int)d.computeList.size(), MPI_DOUBLE, &vz[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        h.resize(d.n); MPI_Gatherv(&d.h[0], (int)d.computeList.size(), MPI_DOUBLE, &h[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        ro.resize(d.n); MPI_Gatherv(&d.ro[0], (int)d.computeList.size(), MPI_DOUBLE, &ro[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        u.resize(d.n); MPI_Gatherv(&d.u[0], (int)d.computeList.size(), MPI_DOUBLE, &u[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        p.resize(d.n); MPI_Gatherv(&d.p[0], (int)d.computeList.size(), MPI_DOUBLE, &p[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        c.resize(d.n); MPI_Gatherv(&d.c[0], (int)d.computeList.size(), MPI_DOUBLE, &c[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        grad_P_x.resize(d.n); MPI_Gatherv(&d.grad_P_x[0], (int)d.computeList.size(), MPI_DOUBLE, &grad_P_x[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        grad_P_y.resize(d.n); MPI_Gatherv(&d.grad_P_y[0], (int)d.computeList.size(), MPI_DOUBLE, &grad_P_y[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        grad_P_z.resize(d.n); MPI_Gatherv(&d.grad_P_z[0], (int)d.computeList.size(), MPI_DOUBLE, &grad_P_z[0], &workload[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(d.comm_rank == 0)
        {
        	if(d.iteration % 10 == 0)
        	{
		        ofstream outputFile;
		        ostringstream oss;
		        oss << "output" << d.iteration << ".txt";
		        outputFile.open(oss.str());
		    
		        for(int i=0; i<d.n; i++)
		        {
		            outputFile << x[i] << ' ' << y[i] << ' ' << z[i] << ' ';
		            // outputFile << vx[i] << ' ' << vy[i] << ' ' << vz[i] << ' ';
		            // outputFile << h[i] << ' ' << ro[i] << ' ' << u[i] << ' ' << p[i] << ' ' << c[i] << ' ';
		            // outputFile << grad_P_x[i] << ' ' << grad_P_y[i] << ' ' << grad_P_z[i] << ' ';
		            double rad = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
		            double vrad = (vx[i] *  x[i] + vy[i] * y[i] + vz[i] * z[i]) / rad;
		            outputFile << rad << ' ' << vrad << endl;  
		        }
		        outputFile.close();
		    }
	    }
    });

    TaskScheduler taskSched(d.comm_rank);
    taskSched.add(&tbalance, TaskScheduler::Params(1, "BalanceDomain"));
    taskSched.add(&tprintBBox);
    taskSched.add(&tbuild, TaskScheduler::Params(1, "BuildTree"));
    // taskSched.add(&treorder, TaskScheduler::Params(1, "Reorder"));
    taskSched.add(&tfind, TaskScheduler::Params(1, "FindNeighbors"));
    taskSched.add(&tcheckNeighbors, TaskScheduler::Params(1, "CheckNeighbors"));
    	taskSched.add(&tsynchronizeHalos);
    taskSched.add(&tdensity, TaskScheduler::Params(1, "Compute Density"));
        taskSched.add(&tsynchronizeHalos);
    taskSched.add(&teos, TaskScheduler::Params(1, "Compute EOS"));
        taskSched.add(&tsynchronizeHalos);
    taskSched.add(&tmomentum, TaskScheduler::Params(1, "Compute Momentum"));
        taskSched.add(&tsynchronizeHalos);
    taskSched.add(&tenergy, TaskScheduler::Params(1, "Compute Energy"));
        taskSched.add(&tsynchronizeHalos);
    taskSched.add(&ttimestep, TaskScheduler::Params(1, "Update Time-step"));
    taskSched.add(&tcheckTimestep, TaskScheduler::Params(1, "Update Time-step"));
        taskSched.add(&tsynchronizeHalos);
    taskSched.add(&tH, TaskScheduler::Params(1, "Update H"));
        taskSched.add(&tsynchronizeHalos);
    taskSched.add(&tupdate, TaskScheduler::Params(1, "UpdateQuantities"));
    taskSched.add(&tcheckConservation, TaskScheduler::Params(1, "CheckConservation"));
    taskSched.add(&twriteFile, TaskScheduler::Params(1, "WriteFile"));

    for(d.iteration = 0; d.iteration < 100; d.iteration++)
    {
		if(d.comm_rank == 0) cout << "Iteration: " << d.iteration << endl;
        taskSched.exec();
        if(d.comm_rank == 0) cout << endl;
    }

    MPI_Finalize();

    return 0;
}
