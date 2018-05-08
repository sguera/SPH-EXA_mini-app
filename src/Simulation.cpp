#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include "Simulation.hpp"
#include <cmath>

#define NEIGHBOURS_SIZE 100

using namespace std;

Simulation::Simulation() : 
    n(1e3), simTime(0.0), targetTime(1.0),
    x(n, 0.0), y(n, 0.0), z(n, 0.0),
    vx(n, 0.0), vy(n, 0.0), vz(n, 0.0),
    grad_x(n, 0.0), grad_y(n, 0.0), grad_z(n, 0.0),
    mass(n, 0.0), pressure(n, 0.0), volume(n, 0.0), h(n, 0.0)
{}

void Simulation::init(const string &filename)
{
    printf("init(%s)\n", &filename[0]);

    ifstream inFile;
    inFile.open(filename);
    if (!inFile)
    {
        cout << "Unable to open file: " << filename;
        exit(1); // terminate with error
    }
    
    for(int i=0; i<n; i++)
    {
        inFile >> x[i] >> y[i] >> z[i] >> vx[i] >> vy[i] >> vz[i] >> pressure[i] >> h[i] >> volume[i] >> mass[i];
//        cout << x[i] << " " << y[i] << " " << z[i] << " " << vx[i] << " " << vy[i] << " " << vz[i] << " ";
//        cout << pressure[i] << " " << h[i] << " " << volume[i] << mass[i] << endl;
    }
    
    inFile.close();
}

bool Simulation::advance()
{
    printf("::advance()\n");
    simTime += 0.5;
    return (simTime < targetTime? true : false);
}

void Simulation::find_neighbors()
{
    printf("::find_neighbors()\n");
    real d;
    vector<vector<real> > neighbours(n, vector<real>(NEIGHBOURS_SIZE, std::numeric_limits<real>::max()));
    vector<vector<real> > neighbours_x(n, vector<real>(NEIGHBOURS_SIZE, std::numeric_limits<real>::max()));
    vector<vector<real> > neighbours_y(n, vector<real>(NEIGHBOURS_SIZE, std::numeric_limits<real>::max()));
    vector<vector<real> > neighbours_z(n, vector<real>(NEIGHBOURS_SIZE, std::numeric_limits<real>::max()));
    int index;

    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<n; ++j){
            if (i != j) {
                d = sqrt(pow((x[i] - x[j]), 2) + pow((y[i] - y[j]), 2) + pow((z[i] - z[j]), 2));
                
                //based on https://stackoverflow.com/questions/47982127/c-how-to-insert-a-new-element-in-a-sorted-vector
                auto pos = std::find_if(neighbours[i].begin(), neighbours[i].end(), [d](auto s) {
                    return s < d;
                });
                if(pos != neighbours[i].end()){
                    neighbours[i].insert(pos, d);
                    index = distance(neighbours[i].begin(),pos);
                
                    neighbours_x[i][index] = x[i];
                    neighbours_y[i][index] = y[i];
                    neighbours_z[i][index] = z[i];
                }
                
            }
        }
    }
}
