#include <iostream>
#include <functional>

#include "TaskScheduler.hpp"
#include "TaskLoop.hpp"
#include "Evrard.hpp"
#include "BBox.hpp"
#include "Density.hpp"
#include "EOS.hpp"
#include "Momentum.hpp"
#include "Energy.hpp"
#include "Timestep.hpp"
#include "tree/Octree.hpp"

using namespace std;
using namespace sphexa;

// template<class TreeType>
// class BuildTree : public ComputeTask
// {
// public:
//     BuildTree(const BBox &bbox, TreeType &tree) : 
//         tree(tree), bbox(bbox) {}

//     virtual void compute()
//     {
//         tree.build(bbox);
//     }

//     const BBox &bbox;
//     TreeType &tree;
// };

// template<class TreeType>
// class FindNeighbors : public ComputeParticleTask
// {
// public:
//     FindNeighbors(const TreeType &tree, std::vector<std::vector<int>> &neighbors) : 
//         ComputeParticleTask(neighbors.size()), tree(tree), neighbors(neighbors) {}

//     virtual void compute(int i)
//     {
//         neighbors[i].resize(0);
//         tree.findNeighbors(i, neighbors[i]);
//     }

//     const TreeType &tree;
//     std::vector<std::vector<int>> &neighbors;
// };

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
    Evrard d("bigfiles/Evrard3D.bin");

    Octree<double> tree(d.x, d.y, d.z, d.h, Octree<double>::Params(/*max neighbors*/d.ngmax, /*bucketSize*/128));

    LambdaTask tbuild([&]()
    {
        tree.build(d.bbox);
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
        for(int i=0; i<d.neighbors.size(); i++)
            sum += d.neighbors[i].size();
        cout << "### Check ### Total Number of Neighbours : " << sum << endl;
    });

    Density<double> tdensity(d.x, d.y, d.z, d.h, d.m, d.neighbors, d.ro);
    EOS<double> teos(d.ro, d.u, d.mui, d.p, d.temp, d.c, d.cv);
    Momentum<double> tmomentum(d.x, d.y, d.z, d.h, d.vx, d.vy, d.vz, d.ro, d.p, d.c, d.m, d.neighbors, d.grad_P_x, d.grad_P_y, d.grad_P_z);
    Energy<double> tenergy(d.x, d.y, d.z, d.h, d.vx, d.vy, d.vz, d.ro, d.p, d.c, d.m, d.neighbors, d.u);
    Timestep<double> ttimestep(d.h, d.c, d.dt_m1, d.dt);

    // TaskScheduler::Params(/*show time=yes*/1, /*name*/"TaskName"));

    TaskScheduler taskSched;
    taskSched.add(&tprintBBox);
    taskSched.add(&tbuild, TaskScheduler::Params(1, "BuildTree"));
    taskSched.add(&tfind, TaskScheduler::Params(1, "FindNeighbors"));
    taskSched.add(&tcheckNeighbors);
    taskSched.add(&tdensity, TaskScheduler::Params(1, "Compute Density"));
    taskSched.add(&teos, TaskScheduler::Params(1, "Compute EOS"));
    taskSched.add(&tmomentum, TaskScheduler::Params(1, "Compute Momentum"));
    taskSched.add(&tenergy, TaskScheduler::Params(1, "Compute Timestep"));

    for(int timeloop = 0; timeloop < 1; timeloop++)
    {
        cout << "Iteration: " << timeloop << endl;
        taskSched.exec();
    }

    return 0;
}


// #include <iostream>
// #include <functional>
// #include <fstream>
// #include <chrono>
// #include <cstring>
// #include <cmath>
// #include <algorithm>
// #include <sstream>
// #include <cassert>

// #include "utils.hpp"
// #include "tree/KdTree.hpp"

// #include "Evrard.hpp"

// using namespace std;
// using namespace std::chrono;
// using namespace sphexa;

// #define PI 3.141592653589793
// #define K53D 0.617013
// #define GAMMA (5.0/3.0)
// #define NV0 100
// #define MAX_DT_INCREASE 1.1
// #define STABILIZATION_TIMESTEPS 15

// typedef std::chrono::high_resolution_clock Clock;
// typedef std::chrono::time_point<Clock> TimePoint;
// typedef std::chrono::duration<float> Time;


// template <typename Callable>
// float task(Callable f)
// {
//     TimePoint start = Clock::now();
//     f();
//     TimePoint stop = Clock::now();
//     return duration_cast<duration<float>>(stop-start).count();
// }

// template <typename Callable>
// float task_loop(int n, Callable f)
// {
//     TimePoint start = Clock::now();
//     #pragma omp parallel for schedule(static)
//     for(int i=0; i<n; i++)
//         f(i);
//     TimePoint stop = Clock::now();
//     return duration_cast<duration<float>>(stop-start).count();
// }

// inline double compute_3d_k(double n)
// {
//     //b0, b1, b2 and b3 are defined in "SPHYNX: an accurate density-based SPH method for astrophysical applications", DOI: 10.1051/0004-6361/201630208
//     double b0 = 2.7012593e-2;
//     double b1 = 2.0410827e-2;
//     double b2 = 3.7451957e-3;
//     double b3 = 4.7013839e-2;

//     return b0 + b1 * sqrt(n) + b2 * n + b3 * sqrt(n*n*n);
// }

// inline double wharmonic(double v, double h, double K)
// {
//     double value = (PI/2.0) * v;
//     return K/(h*h*h) * pow((sin(value)/value), 5);
// }

// inline double wharmonic_derivative(double v, double h, double K)
// {
//     double value = (PI/2.0) * v;
//     // r_ih = v * h
//     // extra h at the bottom comes from the chain rule of the partial derivative
//     double kernel = wharmonic(v, h, K);

//     return 5.0 * (PI/2.0) * kernel / (h * h) / v * ((1.0 / tan(value)) - (1.0 / value));
// }

// inline void eos(double ro, double R, double u, double mui, double &pressure, double &temperature, double &soundspeed, double &cv)
// {
//     cv = (GAMMA - 1) * R / mui;
//     temperature = u / cv;
//     double tmp = u * (GAMMA - 1);
//     pressure = ro * tmp;
//     soundspeed = sqrt(tmp);
// }

// inline double artificial_viscosity(double ro_i, double ro_j, double h_i, double h_j, double c_i, double c_j, double rv, double r_square)
// {
//     double alpha = 1.0;
//     double beta = 2.0;
//     double epsilon = 0.01;

//     double ro_ij = (ro_i + ro_j) / 2.0;
//     double c_ij = (c_i + c_j) / 2.0;
//     double h_ij = (h_i + h_j) / 2.0;


//     //calculate viscosity_ij according to Monaghan & Gringold 1983
//     double viscosity_ij = 0.0;
//     if (rv < 0.0){
//         //calculate muij
//         double mu_ij = (h_ij * rv) / (r_square + epsilon * h_ij * h_ij);
//         viscosity_ij = (-alpha * c_ij * mu_ij + beta * mu_ij * mu_ij) / ro_ij;
//     }

//     return viscosity_ij;
// }


// /*
//     Momentum Equation according to SPH from Variational Principle, Rossweg 2009
// */
// inline void computeMomentum(int i, Evrard &d)
// {
//     static const double gradh_i = 1.0;
//     static const double gradh_j = 1.0;
//     static const double K = compute_3d_k(5.0);
//     double ro_i = d.ro[i];
//     double p_i = d.p[i];
//     double x_i = d.x[i];
//     double y_i = d.y[i];
//     double z_i = d.z[i];
//     double vx_i = d.vx[i];
//     double vy_i = d.vy[i];
//     double vz_i = d.vz[i];
//     double h_i = d.h[i];
//     double momentum_x = 0.0;
//     double momentum_y = 0.0;
//     double momentum_z = 0.0;

//     for(int j=0; j<d.nvi[i]; j++)
//     {
//         // retrive the id of a neighbor
//         int neigh_id = d.ng[i*d.ngmax+j];
//         if(neigh_id == i) continue;

//         double ro_j = d.ro[neigh_id];
//         double p_j = d.p[neigh_id];
//         double x_j = d.x[neigh_id];
//         double y_j = d.y[neigh_id];
//         double z_j = d.z[neigh_id];
//         double h_j = d.h[neigh_id];

//         // calculate the scalar product rv = rij * vij
//         double r_ijx = (x_i - x_j);
//         double r_ijy = (y_i - y_j);
//         double r_ijz = (z_i - z_j);

//         double v_ijx = (vx_i - d.vx[neigh_id]);
//         double v_ijy = (vy_i - d.vy[neigh_id]);
//         double v_ijz = (vz_i - d.vz[neigh_id]);

//         double rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

//         double r_square = (r_ijx * r_ijx) + (r_ijy * r_ijy) + (r_ijz * r_ijz);

//         double viscosity_ij = artificial_viscosity(ro_i, ro_j, d.h[i], d.h[neigh_id], d.c[i], d.c[neigh_id], rv, r_square);

//         double r_ij = sqrt(r_square);
//         double v_i = r_ij / h_i;
//         double v_j = r_ij / h_j;

//         double derivative_kernel_i = wharmonic_derivative(v_i, h_i, K);
//         double derivative_kernel_j = wharmonic_derivative(v_j, h_j, K);
        
//         double grad_v_kernel_x_i = r_ijx * derivative_kernel_i;
//         double grad_v_kernel_x_j = r_ijx * derivative_kernel_j;
//         double grad_v_kernel_y_i = r_ijy * derivative_kernel_i;
//         double grad_v_kernel_y_j = r_ijy * derivative_kernel_j;
//         double grad_v_kernel_z_i = r_ijz * derivative_kernel_i;
//         double grad_v_kernel_z_j = r_ijz * derivative_kernel_j;

//         if (isnan(grad_v_kernel_x_i) || isnan(grad_v_kernel_x_j) || isnan(grad_v_kernel_y_i) || isnan(grad_v_kernel_y_j) || isnan(grad_v_kernel_z_i) || isnan(grad_v_kernel_z_j))
//             cout << "ERROR: " << grad_v_kernel_x_i << ' ' << grad_v_kernel_x_j << ' ' << grad_v_kernel_y_i << ' ' << grad_v_kernel_y_j << ' ' << grad_v_kernel_z_i << ' ' << grad_v_kernel_z_j << endl;
//         if (isnan(p_i) || isnan(ro_i) || isnan(ro_j) || isnan(viscosity_ij))
//             cout << "ERROR: " << p_i << ' ' << ro_i << ' ' << ro_j << ' ' << viscosity_ij << endl;
//         momentum_x +=  (p_i/(gradh_i * ro_i * ro_i) * grad_v_kernel_x_i) + (p_j/(gradh_j * ro_j * ro_j) * grad_v_kernel_x_j) + viscosity_ij * (grad_v_kernel_x_i + grad_v_kernel_x_j)/2.0;
//         momentum_y +=  (p_i/(gradh_i * ro_i * ro_i) * grad_v_kernel_y_i) + (p_j/(gradh_j * ro_j * ro_j) * grad_v_kernel_y_j) + viscosity_ij * (grad_v_kernel_y_i + grad_v_kernel_y_j)/2.0;
//         momentum_z +=  (p_i/(gradh_i * ro_i * ro_i) * grad_v_kernel_z_i) + (p_j/(gradh_j * ro_j * ro_j) * grad_v_kernel_z_j) + viscosity_ij * (grad_v_kernel_z_i + grad_v_kernel_z_j)/2.0;

//     }

//     d.grad_P_x[i] = momentum_x * d.m[i];
//     d.grad_P_y[i] = momentum_y * d.m[i];
//     d.grad_P_z[i] = momentum_z * d.m[i];
// }



// inline void computeEnergy(int i, Evrard &d)
// {
//     // note that practically all these variables are already calculated in
//     // computeMomentum, so it would make sens to fuse momentum and energy
//     static const double gradh_i = 1.0;
//     static const double K = compute_3d_k(5.0);

//     double ro_i = d.ro[i];
//     double p_i = d.p[i];
//     double vx_i = d.vx[i];
//     double vy_i = d.vy[i];
//     double vz_i = d.vz[i];
//     double x_i = d.x[i];
//     double y_i = d.y[i];
//     double z_i = d.z[i];
//     double h_i = d.h[i];

//     double energy = 0.0;

//     for(int j=0; j<d.nvi[i]; j++)
//     {
//         // retrive the id of a neighbor
//         int neigh_id = d.ng[i*d.ngmax+j];
//         if(neigh_id == i) continue;

//         double ro_j = d.ro[neigh_id];
//         double m_j = d.m[neigh_id];
//         double x_j = d.x[neigh_id];
//         double y_j = d.y[neigh_id];
//         double z_j = d.z[neigh_id];

//         //calculate the velocity difference
//         double v_ijx = (vx_i - d.vx[neigh_id]);
//         double v_ijy = (vy_i - d.vy[neigh_id]);
//         double v_ijz = (vz_i - d.vz[neigh_id]);

//         double r_ijx = (x_i - x_j);
//         double r_ijy = (y_i - y_j);
//         double r_ijz = (z_i - z_j);

//         double rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

//         double r_square = (r_ijx * r_ijx) + (r_ijy * r_ijy) + (r_ijz * r_ijz);

//         double viscosity_ij = artificial_viscosity(ro_i, ro_j, d.h[i], d.h[neigh_id], d.c[i], d.c[neigh_id], rv, r_square);

//         double r_ij = sqrt(r_square);
//         double v_i = r_ij / h_i;

//         double derivative_kernel_i = wharmonic_derivative(v_i, h_i, K);

//         double grad_v_kernel_x_i = r_ijx * derivative_kernel_i;
//         double grad_v_kernel_y_i = r_ijy * derivative_kernel_i;
//         double grad_v_kernel_z_i = r_ijz * derivative_kernel_i;

//         energy +=  m_j * (1 + 0.5 * viscosity_ij) * (v_ijx * grad_v_kernel_x_i + v_ijy * grad_v_kernel_y_i + v_ijz * grad_v_kernel_z_i);
//     }

//     d.d_u[i] =  energy * (-p_i/(gradh_i * ro_i * ro_i));

// }

// // inline void buildTree(Evrard &d, BroadTree &t)
// inline void buildTree(Evrard &d, KdTree &t)
// {
//     d.xmin = INFINITY, d.xmax = -INFINITY, d.ymin = INFINITY, d.ymax = -INFINITY, d.zmin = INFINITY, d.zmax = -INFINITY;
//     for(int i=0; i<d.n; i++)
//     {
//         if(d.x[i] < d.xmin) d.xmin = d.x[i];
//         if(d.x[i] > d.xmax) d.xmax = d.x[i];
//         if(d.y[i] < d.ymin) d.ymin = d.y[i];
//         if(d.y[i] > d.ymax) d.ymax = d.y[i];
//         if(d.z[i] < d.zmin) d.zmin = d.z[i];
//         if(d.z[i] > d.zmax) d.zmax = d.z[i];
//     }

//     printf("Domain x[%f %f]\n", d.xmin, d.xmax);
//     printf("Domain y[%f %f]\n", d.ymin, d.ymax);
//     printf("Domain z[%f %f]\n", d.zmin, d.zmax);
 
//     t.setBox(d.xmin, d.xmax, d.ymin, d.ymax, d.zmin, d.zmax);
//     t.build(d.n, &d.x[0], &d.y[0], &d.z[0]);//, d.h);
//     //cout << "CELLS: " << t.cellCount() << endl;
// }

// // inline void findNeighbors(int i, Evrard &d, BroadTree &t)
// inline void findNeighbors(int i, Evrard &d, KdTree &t)
// {
//     t.findNeighbors(d.x[i], d.y[i], d.z[i], 2.0*d.h[i], d.ngmax, &d.ng[(long)i*d.ngmax], d.nvi[i], d.PBCx, d.PBCy, d.PBCz);
// }

// //templating it with the kernel to use would be an option
// inline void computeDensity(int i, Evrard &d)
// {
//     static const double K = compute_3d_k(5.0);

//     //double old = d.ro[i];
//     double roloc = 0.0;
//     d.ro[i] = 0.0;

//     for(int j=0; j<d.nvi[i]; j++)
//     {
//         // retrive the id of a neighbor
//         int neigh_id = d.ng[i*d.ngmax+j];
//         if(neigh_id == i) continue;

//         // later can be stores into an array per particle
//         double dist =  distance(d.x[i], d.y[i], d.z[i], d.x[neigh_id], d.y[neigh_id], d.z[neigh_id]); //store the distance from each neighbor

//         // calculate the v as ratio between the distance and the smoothing length
//         double vloc = dist / d.h[i];
        
//         //assert(vloc<=2);
//         double value = wharmonic(vloc, d.h[i], K);
//         roloc += value * d.m[neigh_id];
//     }
//     d.ro[i] = roloc + d.m[i] * K/(d.h[i]*d.h[i]*d.h[i]);
//     //printf("ro[%d] = %f %f\n", i, old, d.ro[i]);
// }

// inline void computeEOS(int i, Evrard &d)
// {
//     static const double R = 8.317e7;

//     eos(d.ro[i], R, d.u[i], d.mui[i], d.p[i], d.temp[i], d.c[i], d.cv[i]);

// }

// //computes the timestep using the Courant condition
// inline void computeTimestep(int i, Evrard &d)
// {
//     static const double CHI = 0.2;
//     d.timestep[i] = CHI * (d.h[i]/d.c[i]);
// }


// //computes the smoothing lenght
// inline void computeH(int i, Evrard &d)
// {

//     static const double c0 = 7.0;
//     static const double exp = 1.0/3.0;//1.0/log(c0 + 1)

//     double ka = pow((1 + c0 * NV0 / d.nvi[i]), exp);

//     d.h[i] = d.h[i] * 0.5 * ka;


// }


// //updates the quantities
// inline void updateQuantities(int i, Evrard &d)
// {

//     double t_m1 = d.timestep_m1[i];
//     double t_0 = d.timestep[i];
//     double x = d.x[i];
//     double y = d.y[i];
//     double z = d.z[i];

//     // in case it is the first iteration we do something different
//     if(d.iteration == 0){
//         t_0 = 0.0001;
//         t_m1 = t_0;

//         d.d_u_m1[i] = d.d_u[i];
//         d.x_m1[i] = x - d.vx[i] * t_0;
//         d.y_m1[i] = y - d.vy[i] * t_0;
//         d.z_m1[i] = z - d.vz[i] * t_0;
//     }

//     // ADD COMPONENT DUE TO THE GRAVITY HERE
//     double ax = - (d.grad_P_x[i]); //-G * fx
//     double ay = - (d.grad_P_y[i]); //-G * fy
//     double az = - (d.grad_P_z[i]); //-G * fz

//     // if(d.iteration < STABILIZATION_TIMESTEPS){
//     //     ax = 0.0;
//     //     ay = 0.0;
//     //     az = 0.0;
//     // }

//     if (isnan(ax) || isnan(ay) || isnan(az))
//         cout << "ERROR: " << ax << ' ' << ay << ' ' << az << endl;

//     //update positions according to Press (2nd order)
//     double deltaA = t_0 + 0.5 * t_m1;
//     double deltaB = 0.5 * (t_0 + t_m1);

//     double valx = (x - d.x_m1[i]) / t_m1;
//     double valy = (y - d.y_m1[i]) / t_m1;
//     double valz = (z - d.z_m1[i]) / t_m1;

//     double vx = valx + ax * deltaA;
//     double vy = valy + ay * deltaA;
//     double vz = valz + az * deltaA;
   
//     d.vx[i] = vx;
//     d.vy[i] = vy;
//     d.vz[i] = vz;
    
//     d.x[i] = x + t_0 * valx + (vx - valx) * t_0 * deltaB / deltaA;
//     d.x_m1[i] = x;

//     d.y[i] = y + t_0 * valy + (vy - valy) * t_0 * deltaB / deltaA;
//     d.y_m1[i] = y;

//     d.z[i] = z + t_0 * valz + (vz - valz) * t_0 * deltaB / deltaA;
//     d.z_m1[i] = z;

//     //update the energy according to Adams-Bashforth (2nd order)
//     deltaA = 0.5 * t_0 * t_0 / t_m1;
//     deltaB = t_0 + deltaA;

//     d.u[i] = d.u[i] + 0.5 * d.d_u[i] * deltaB - 0.5 * d.d_u_m1[i] * deltaA;
//     d.d_u_m1[i] = d.d_u[i];


//     //update positions
//     d.x_m1[i] = x;
//     d.y_m1[i] = y;
//     d.z_m1[i] = z;

//     d.timestep_m1[i] = t_0;

// }

// //applies the conservation laws
// inline double conservation(Evrard &d){
//     double e_tot = 0.0;
//     double e_cin = 0.0;
//     double e_int = 0.0;
    

//     #pragma omp parallel for reduction(+:e_cin,e_int)
//     for(int i=0; i<d.n; i++){
//         double vmod2 = 0.0;
//         vmod2 = d.vx[i] * d.vx[i] + d.vy[i] * d.vy[i] + d.vz[i] * d.vz[i];
//         e_cin += 0.5 * d.m[i] * vmod2;
//         e_int += d.u[i] * d.m[i];
        
//     }
//     e_tot += e_cin + e_int;

//     return e_tot;
// }


// int main()
// {
//     // Domain (x, y, z, vx, vy, vz, ro, u, p, h, m, temp, mue, mui)
//     Evrard evrard("bigfiles/Evrard3D.bin");

//     // Tree structure
//     KdTree tree;



//     for(int timeloop = 0; timeloop < 100; timeloop++){

//         cout << "Time-step : " << evrard.iteration << endl;

//         // cout << endl << "Building tree..." << endl;

//         float ms = task([&]()
//         {
//             buildTree(evrard, tree);
//         });

//         cout << "# Total Time (s) to build the tree : " << ms << endl;

//         // cout << endl << "Finding neighbors..." << endl;

//         for(int i=0; i<evrard.n; i++)
//             evrard.nvi[i] = 0;

//         ms = task_loop(evrard.n, [&](int i)
//         {
//             findNeighbors(i, evrard, tree);
//         });

//         cout << "# Total Time (s) to find the neighbors : " << ms << endl;

//         //DEBUG
//         int sum = 0;
//         for(int i=0; i<evrard.n; i++)
//             sum+= evrard.nvi[i];

//         cout << "# Total Number of Neighbours : " << sum << endl;


//         // cout << endl << "Conservation Laws..." << endl;
//         double e_tot = 0.0;

//         ms = task([&]()
//         {
//             e_tot = conservation(evrard);
//         });

//         cout << "# Total energy resulting from the Conservation Laws : " << e_tot << endl;
//         // END DEBUG

//         // cout << endl << "Computing Density..." << endl;

//         ms = task_loop(evrard.n, [&](int i)
//         {
//             computeDensity(i, evrard);
//         });

//         cout << "# Total Time (s) to compute the density : " << ms << endl;

//         // cout << endl << "Computing EOS..." << endl;

//         ms = task_loop(evrard.n, [&](int i)
//         {
//             computeEOS(i, evrard);
//         });

//         cout << "# Total Time (s) to compute the EOS : " << ms << endl;

//         // cout << endl << "Computing Momentum..." << endl;

//         ms = task_loop(evrard.n, [&](int i)
//         {
//             computeMomentum(i, evrard);
//         });     

//         cout << "# Total Time (s) to compute the Momentum : " << ms << endl;

//         // cout << endl << "Computing Energy..." << endl;

//         ms = task_loop(evrard.n, [&](int i)
//         {
//             computeEnergy(i, evrard);
//         });

//         cout << "# Total Time (s) to compute the Energy : " << ms << endl;

//         // cout << endl << "Computing Timestep..." << endl;

//         ms = task_loop(evrard.n, [&](int i)
//         {
//             computeTimestep(i, evrard);
//         });
//         //find the minimum timestep between the ones of each particle and use that one as new timestep
//         TimePoint start1 = Clock::now();
//         auto result = *std::min_element(evrard.timestep.begin(), evrard.timestep.end());
//         //int index = std::distance(evrard.timestep.begin(), it);
//         // double min = evrard.timestep[index];

//         //all particles have the same time-step so we just take the one of particle 0
//         double min = std::min(result, MAX_DT_INCREASE * evrard.timestep_m1[0]);

//         // double min = 10.0;
//         //int minIndex = evrard.n + 1;
//         // for (int i = 0; i < evrard.n; ++i){
//         //     if (evrard.timestep[i] < min){
//         //         min = evrard.timestep[i];
//         // //        minIndex = i;
//         //     }
//         // }
//         std::fill(evrard.timestep.begin(), evrard.timestep.end(), min);

//         //cout << "# Total Time (s) to compute the Timestep : " << ms << endl;
//         TimePoint stop1 = Clock::now();
//         cout << "# Total Time (s) to compute the Timestep : " << ms + duration_cast<duration<float>>(stop1-start1).count() << endl;

//         // cout << endl << "Computing Smoothing Length..." << endl;

//         ms = task_loop(evrard.n, [&](int i)
//         {
//             computeH(i, evrard);
//         });

//         cout << "# Total Time (s) to compute the Smoothing Length : " << ms << endl;

//         // cout << endl << "Updating Quantities..." << endl;

//         ms = task_loop(evrard.n, [&](int i)
//         {
//             updateQuantities(i, evrard);
//         });
//         evrard.iteration += 1;

//         cout << "# Total Time (s) to update the Quantities : " << ms << endl;

//         // cout << "t_0 : " << evrard.timestep[0] << endl;
//         // cout << "t_m1 : " << evrard.timestep_m1[0] << endl;


//     // double rad = 0.0;
//         ofstream outputFile;
//         ostringstream oss;
//         oss << "output" << evrard.iteration <<".txt";
//         outputFile.open(oss.str());
    
//         for(int i=0; i<evrard.n; i++)
//         {
//             outputFile << evrard.x[i] << ' ' << evrard.y[i] << ' ' << evrard.z[i] << ' ' << evrard.grad_P_x[i] << ' ' << evrard.grad_P_y[i] << ' ' << evrard.grad_P_z[i] << ' ' << (evrard.vx[i] *  evrard.x[i] + evrard.vy[i] * evrard.y[i] + evrard.vz[i] * evrard.z[i]) / sqrt(evrard.x[i]*evrard.x[i] + evrard.y[i] * evrard.y[i] + evrard.z[i] * evrard.z[i]) << ' ' << sqrt(evrard.x[i]*evrard.x[i] + evrard.y[i] * evrard.y[i] + evrard.z[i] * evrard.z[i]) << ' ' << evrard.ro[i] << ' ' << endl;  
//         }
//         outputFile.close();
//         cout << endl;

//         } //closes timeloop
    

//     return 0;
// }

