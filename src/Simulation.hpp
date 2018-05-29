#ifndef _SIMULATION_HPP
#define _SIMULATION_HPP

#include <vector>
#include <string>
#include <iostream>
#include "Octree"

typedef float real;

class Simulation 
{
public:

	Simulation();

	void init(const std::string &filename);
	bool advance();
	void find_neighbors();

	int n;		//number of particles

	real simTime;	//total time of the simulation
	real targetTime;

	std::vector<real> x, y, z;
	std::vector<real> vx, vy, vz;
	std::vector<real> grad_x, grad_y, grad_z;
	std::vector<real> mass, pressure, volume, h;
    
private:
    Octree octree;
    
};

#endif
