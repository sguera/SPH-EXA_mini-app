#include <cmath>
#include "Types.hpp"


struct Particle
{
  public:
    Particle() { init();}
    void init() 
    {
      pos_x = NULL; pos_y = NULL; pos_z = NULL;
      vel_x = NULL; vel_y = NULL; vel_z = NULL;
      //acc_x = NULL; acc_y = NULL; acc_z = NULL;
      grad_p_x = NULL; grad_p_y = NULL; grad_p_z = NULL;
      grav_f_x = NULL; grav_f_y= NULL; grav_f_z = NULL;
      mass  = NULL;
    }
    real_type *pos_x, *pos_y, *pos_z;
    real_type *vel_x, *vel_y, *vel_z;
//real_type *acc_x, *acc_y, *acc_z;
    
    //gradient of pressure
    real_type *grad_p_x, *grad_p_y, *grad_p_z;
    
    //gravitational force
    real_type *grav_f_x, *grav_f_y, *grav_f_z;
    
    //smoothing length
    real_type *smooth_length;
    
    real_type *mass;
};
