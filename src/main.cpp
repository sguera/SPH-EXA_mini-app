#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>

#include "TaskScheduler.hpp"
#include "TaskLoop.hpp"
#include "Dataset.hpp"
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
    LambdaTaskLoop(int n, const std::function<void(int i)> func) : TaskLoop(n), func(func) {}
    virtual void compute(int i) { func(i); }
    const std::function<void(int i)> func; 
};

int main()
{
    // Dataset: contains arrays (x, y, z, vx, vy, vz, ro, u, p, h, m, temp, mue, mui)
    Dataset d(1e6, "bigfiles/Evrard3D.bin");

    Octree<double> tree(d.x, d.y, d.z, d.h, Octree<double>::Params(/*max neighbors*/d.ngmax, /*bucketSize*/128));
    //HTree<double> tree(d.x, d.y, d.z, d.h, HTree<double>::Params(/*max neighbors*/d.ngmax, /*bucketSize*/128));

    LambdaTask tbuild([&]()
    {
        d.computeBBox();
        tree.build(d.bbox);
    });

    LambdaTask treorder([&]()
    {
        d.reorder(*tree.ordering);
        for(unsigned i=0; i<d.x.size(); i++)
            (*tree.ordering)[i] = i;
    });

    LambdaTaskLoop tfind(d.n, [&](int i)
    {
        d.neighbors[i].resize(0);
        tree.findNeighbors(i, d.neighbors[i]);
    });

    LambdaTask tprintBBox([&]()
    { 
        cout << "### Check ### Computational domain: ";
        cout << d.bbox.xmin << " " << d.bbox.xmax << " ";
        cout << d.bbox.ymin << " " << d.bbox.ymax << " ";
        cout << d.bbox.zmin << " " << d.bbox.zmax << endl;
    });

    LambdaTask tcheckNeighbors([&]()
    { 
        int sum = 0;
        for(unsigned int i=0; i<d.neighbors.size(); i++)
            sum += d.neighbors[i].size();
        cout << "### Check ### Total number of neighbours: " << sum << endl;
    });

    Density<double> tdensity(d.x, d.y, d.z, d.h, d.m, d.neighbors, d.ro);
    EOS<double> teos(d.ro, d.u, d.mui, d.p, d.temp, d.c, d.cv);
    Momentum<double> tmomentum(d.x, d.y, d.z, d.h, d.vx, d.vy, d.vz, d.ro, d.p, d.c, d.m, d.neighbors, d.grad_P_x, d.grad_P_y, d.grad_P_z);
    Energy<double> tenergy(d.x, d.y, d.z, d.h, d.vx, d.vy, d.vz, d.ro, d.p, d.c, d.m, d.neighbors, d.du);
    Timestep<double> ttimestep(d.h, d.c, d.dt_m1, d.dt);

    LambdaTask tcheckTimestep([&]()
    { 
        cout << "### Check ### Old Time-step: " << d.dt_m1[0] << ", New time-step: " << d.dt[0] << endl;
    });

    H<double> tH(d.neighbors, d.h, H<double>::Params(/*Target No of neighbors*/100));
    UpdateQuantities<double> tupdate(d.grad_P_x, d.grad_P_y, d.grad_P_z, d.dt, d.du, d.iteration, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.u, d.du_m1, d.dt_m1);

    LambdaTask tcheckConservation([&]()
    { 
        double e_tot = 0.0, e_cin = 0.0, e_int = 0.0;

        for(int i=0; i<d.n; i++)
        {
            double vmod2 = 0.0;
            vmod2 = d.vx[i] * d.vx[i] + d.vy[i] * d.vy[i] + d.vz[i] * d.vz[i];
            e_cin += 0.5 * d.m[i] * vmod2;
            e_int += d.u[i] * d.m[i]; 
        }

        e_tot += e_cin + e_int;
    
        cout << "### Check ### Total energy: " << e_tot << ", (internal: " << e_int << ", cinetic: " << e_cin << ")" << endl;
    });

    LambdaTask twriteFile([&]()
    {
        ofstream outputFile;
        ostringstream oss;
        oss << "output" << d.iteration << ".txt";
        outputFile.open(oss.str());
    
        for(int i=0; i<d.n; i++)
        {
            outputFile << d.x[i] << ' ' << d.y[i] << ' ' << d.z[i] << ' ';
            outputFile << d.vx[i] << ' ' << d.vy[i] << ' ' << d.vz[i] << ' ';
            outputFile << d.h[i] << ' ' << d.ro[i] << ' ' << d.u[i] << ' ' << d.p[i] << ' ' << d.c[i] << ' ';
            outputFile << d.grad_P_x[i] << ' ' << d.grad_P_y[i] << ' ' << d.grad_P_z[i] << ' ';
            double rad = sqrt(d.x[i] * d.x[i] + d.y[i] * d.y[i] + d.z[i] * d.z[i]);
            double vrad = (d.vx[i] *  d.x[i] + d.vy[i] * d.y[i] + d.vz[i] * d.z[i]) / rad;
            outputFile << rad << ' ' << vrad << endl;  
        }
        outputFile.close();
    });

    TaskScheduler taskSched;
    taskSched.add(&tbuild, TaskScheduler::Params(1, "BuildTree"));
    taskSched.add(&tprintBBox);
    taskSched.add(&treorder, TaskScheduler::Params(1, "Reorder"));
    taskSched.add(&tfind, TaskScheduler::Params(1, "FindNeighbors"));
    taskSched.add(&tcheckNeighbors, TaskScheduler::Params(1, "CheckNeighbors"));
    taskSched.add(&tdensity, TaskScheduler::Params(1, "Compute Density"));
    taskSched.add(&teos, TaskScheduler::Params(1, "Compute EOS"));
    taskSched.add(&tmomentum, TaskScheduler::Params(1, "Compute Momentum"));
    taskSched.add(&tenergy, TaskScheduler::Params(1, "Compute Energy"));
    taskSched.add(&ttimestep, TaskScheduler::Params(1, "Update Time-step"));
    taskSched.add(&tcheckTimestep, TaskScheduler::Params(1, "Update Time-step"));
    taskSched.add(&tH, TaskScheduler::Params(1, "Update H"));
    taskSched.add(&tupdate, TaskScheduler::Params(1, "UpdateQuantities"));
    taskSched.add(&tcheckConservation, TaskScheduler::Params(1, "CheckConservation"));
    //taskSched.add(&twriteFile, TaskScheduler::Params(1, "WriteFile"));

    for(d.iteration = 0; d.iteration < 200; d.iteration++)
    {
        cout << "Iteration: " << d.iteration << endl;
        taskSched.exec();
        cout << endl;
    }

    return 0;
}

