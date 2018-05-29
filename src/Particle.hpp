#include <vector>
#include "Types.hpp"

#ifndef _PARTICLE_H_
#define _PARTICLE_H_
struct Particle
{
public:
    Particle() { init();}
    void init()
    {}
    //position in each dimension
    std::vector<real> *x, *y, *z;
    
    //velocity in each dimension
    std::vector<real> *vx, *vy, *vz;
    
    //gradient of pressure
    std::vector<real> *gradx, *grady, *gradz;
    
    // mass, pressure and volume
    std::vector<real> *mass, *pressure, *volume;
    
    //smoothing length
    std::vector<real> *h;
    
    //gravitational force
    std::vector<real> *gravfx, *gravfy, *gravfz;

};

#endif
