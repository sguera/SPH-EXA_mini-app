#include <iostream>
#include "Simulation.hpp"

int main(int argc, char** argv)
{
    Simulation sim;

    sim.init("squarepatch3D");
    //sim.readdata();
    //sim.init_scenario();

    do
    {
       //sim.buildtree();
       sim.find_neighbors();
       //sim.calculate_density();
       //sim.calculate_omega();
       //sim.calculate_IAD();
       //sim.eostot();
       //sim.calculate_divv();
       //sim.momeqn();

       //if(gravity)
           //sim.treewalk();

       //sim.update();
       //sim.timectrl();
    } while(sim.advance());
    
    return 0;
}
