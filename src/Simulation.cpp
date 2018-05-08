#include <iostream>
#include <iomanip>
#include <fstream>
#include "Simulation.hpp"

using namespace std;

Simulation::Simulation() : 
    n(100), simTime(0.0), targetTime(1.0),
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
        cout << x[i] << " " << y[i] << " " << z[i] << " " << vx[i] << " " << vy[i] << " " << vz[i] << " ";
        cout << pressure[i] << " " << h[i] << " " << volume[i] << mass[i] << endl;
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
}