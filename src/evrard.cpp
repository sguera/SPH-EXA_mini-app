#include <iostream>

#include "sphexa.hpp"
#include "Evrard.hpp"

using namespace std;
using namespace sphexa;

template<typename T>
void reorderParticles(Evrard<T> &d, std::vector<int> &ordering)
{
    d.reorder(ordering);
    for(unsigned i=0; i<d.x.size(); i++)
        ordering[i] = i;
}

int main()
{
    typedef double Real;
    typedef Octree<Real> Tree;
    //typedef HTree<Real> Tree;

    Evrard<Real> d(1e6, "bigfiles/Evrard3D.bin");

    Tree tree(d.x, d.y, d.z, d.h, Tree::Params(/*max neighbors*/d.ngmax, /*bucketSize*/128));

    // Main computational tasks
    LambdaTask tComputeBBox([&](){ d.computeBBox(); });
    BuildTree<Tree> tBuildTree(d.bbox, tree);
    LambdaTask tReorderParticles([&](){ reorderParticles(d, *tree.ordering); });
    FindNeighbors<Tree> tFindNeighbors(tree, d.neighbors);
    Density<Real> tDensity(d.x, d.y, d.z, d.h, d.m, d.neighbors, d.ro);
    EquationOfState<Real> tEquationOfState(d.ro, d.u, d.mui, d.p, d.temp, d.c, d.cv);
    Momentum<Real> tMomentum(d.x, d.y, d.z, d.h, d.vx, d.vy, d.vz, d.ro, d.p, d.c, d.m, d.neighbors, d.grad_P_x, d.grad_P_y, d.grad_P_z);
    Energy<Real> tEnergy(d.x, d.y, d.z, d.h, d.vx, d.vy, d.vz, d.ro, d.p, d.c, d.m, d.neighbors, d.du);
    Timestep<Real> tTimestep(d.h, d.c, d.dt_m1, d.dt);
    SmoothingLength<Real> tSmoothingLength(d.neighbors, d.h, SmoothingLength<Real>::Params(/*Target No of neighbors*/d.ng0));
    UpdateQuantities<Real> tUpdateQuantities(d.grad_P_x, d.grad_P_y, d.grad_P_z, d.dt, d.du, d.iteration, d.x, d.y, d.z, d.vx, d.vy, d.vz, d.x_m1, d.y_m1, d.z_m1, d.u, d.du_m1, d.dt_m1);
    EnergyConservation<Real> tEnergyConservation(d.u, d.vx, d.vy, d.vz, d.m, d.etot, d.ecin, d.eint);
    
    // Small tasks to check the result
    LambdaTask tPrintBBox([&](){
        cout << "### Check ### Computational domain: ";
        cout << d.bbox.xmin << " " << d.bbox.xmax << " ";
        cout << d.bbox.ymin << " " << d.bbox.ymax << " ";
        cout << d.bbox.zmin << " " << d.bbox.zmax << endl;
    });
    LambdaTask tCheckNeighbors([&]()
    { 
        int sum = 0;
        for(unsigned int i=0; i<d.neighbors.size(); i++)
            sum += d.neighbors[i].size();
        cout << "### Check ### Avg. number of neighbours: " << sum/d.n << "/" << d.ng0 << endl;
    });
    LambdaTask tCheckTimestep([&](){ cout << "### Check ### Old Time-step: " << d.dt_m1[0] << ", New time-step: " << d.dt[0] << endl; });
    LambdaTask tCheckConservation([&](){ cout << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", cinetic: " << d.ecin << ")" << endl; });
    LambdaTask twriteFile([&](){ d.writeFile(); });

    // We build the workflow
    Workflow work;
    work.add(&tComputeBBox);
    work.add(&tBuildTree, Workflow::Params(1, "BuildTree"));
    work.add(&tReorderParticles, Workflow::Params(1, "Reorder"));
    work.add(&tFindNeighbors, Workflow::Params(1, "FindNeighbors"));
    work.add(&tDensity, Workflow::Params(1, "Compute Density"));
    work.add(&tEquationOfState, Workflow::Params(1, "Compute EOS"));
    work.add(&tMomentum, Workflow::Params(1, "Compute Momentum"));
    work.add(&tEnergy, Workflow::Params(1, "Compute Energy"));
    work.add(&tTimestep, Workflow::Params(1, "Update Time-step"));
    work.add(&tSmoothingLength, Workflow::Params(1, "Update H"));
    work.add(&tUpdateQuantities, Workflow::Params(1, "UpdateQuantities"));
    work.add(&tEnergyConservation, Workflow::Params(1, "Compute Total Energy"));
    work.add(&tPrintBBox);
    work.add(&tCheckNeighbors);
    work.add(&tCheckTimestep);
    work.add(&tCheckConservation);

    //taskSched.add(&twriteFile, Workflow::Params(1, "WriteFile"));

    for(d.iteration = 0; d.iteration < 200; d.iteration++)
    {
        cout << "Iteration: " << d.iteration << endl;
        work.exec();
        cout << endl;
    }

    return 0;
}

