#include <iostream>

#include "Simulation.hpp"



int main(int argc, char** argv)
{
    int N;            //number of particles
    int nsteps;         //number ot integration steps
    
    Simulation sim;
    
    if(argc>1)
    {
        N=atoi(argv[1]);
        sim.set_number_of_particles(N);
        if(argc==3)
        {
            nsteps=atoi(argv[2]);
            sim.set_number_of_steps(nsteps);
        }
    }
    
    
    
    sim.start();
    
    return 0;
}

int newmain(int argc, char** argv)
{
    int N;            //number of particles
    int nsteps;         //number ot integration steps
    
    Simulation sim;
    
    if(argc>1)
    {
        N=atoi(argv[1]);
        sim.set_number_of_particles(N);
        if(argc==3)
        {
            nsteps=atoi(argv[2]);
            sim.set_number_of_steps(nsteps);
        }
    }
    
//call testparameters
//    call init
//
//    call readdata
//    call init_scenario
//
//    timepass=0.d0
//
//
//    !.....Start iterations.
//    do 100 l=iterini,nnl
//        call buildtree
//        call findneighbors
//
//        call calculate_density !Density
//
//        call calculate_omega      !Omega
//
//        call calculate_IAD       !6 - IAD terms
//
//        call eostot
//
//        call calculate_divv    !9 - Div-v
//
//
//        call momeqn            !11 - Mom + Energy eqs
//
//
//
//
//        if (gravity) then
//            call treewalk          !13 - Gravity calculation
//
//
//        call update            !15 - Update
//
//        call timectrl          !16 - Timestep
//
//
//        if(conserv)call conservation      !17 - Conservation laws
    
                
    
    return 0;
}
