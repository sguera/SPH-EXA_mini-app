#ifndef _SIMULATION_HPP
#define _SIMULATION_HPP

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>

#include <omp.h>

#define R_MAX 1<<16
#include "particle.hpp"

class Simulation 
{
public:
  Simulation();
  ~Simulation();
  
  void init();
  void set_number_of_particles(int N);
  void set_number_of_steps(int N);
  void start();
  
private:
  Particle *particles;
  
  int nparts;		//number of particles
  int nsteps;		//number of integration steps
  real_type tstep;		//time step of the simulation

  int sfreq;		//sample frequency
  
  real_type kenergy;		//kinetic energy
  
  double totTime;		//total time of the simulation
  double totFlops;		//total number of flops 
   
  void init_pos();	
  void init_vel();
  void init_acc();
  void init_mass();
    
  inline void set_nparts(const int &N){ nparts = N; }
  inline int get_nparts() const {return nparts; }
  
  inline void set_tstep(const real_type &dt){ tstep = dt; }
  inline real_type get_tstep() const {return tstep; }
  
  inline void set_nsteps(const int &n){ nsteps = n; }
  inline int get_nsteps() const {return nsteps; }
  
  inline void set_sfreq(const int &sf){ sfreq = sf; }
  inline int get_sfreq() const {return sfreq; }
  
  void print_header();
  
};

#endif
