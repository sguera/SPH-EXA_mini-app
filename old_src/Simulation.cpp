#include <stdio.h>
#include <stdlib.h>
#include "Simulation.hpp"
#include "cpu_time.hpp"
#include <omp.h>


Simulation :: Simulation()
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Simulation" << std::endl;
  set_nparts(2000); 
  set_nsteps(500);
  set_tstep(0.1); 
  set_sfreq(50);
}

void Simulation :: set_number_of_particles(int N)  
{
  set_nparts(N);
}

void Simulation :: set_number_of_steps(int N)  
{
  set_nsteps(N);
}

void Simulation :: init_pos()  
{
  int gen = 42; 
  srand(gen);
  real_type max = static_cast<real_type> ( R_MAX );
  
  for(int i=0; i<get_nparts(); ++i)
  {
    real_type r = static_cast<real_type>(rand()) / static_cast<real_type>(RAND_MAX); 
    r = (max - 1.0f) * r + 1.0f;
    particles->pos_x[i] = -1.0f + 2.0f * r / max;
    particles->pos_y[i] = -1.0f + 2.0f * r / max;
    particles->pos_z[i] = -1.0f + 2.0f * r / max;
  }
}

void Simulation :: init_vel()  
{
  int gen = 42;
  srand(gen);
  real_type max = static_cast<real_type> (RAND_MAX);

  for(int i=0; i<get_nparts(); ++i)
  {
    real_type r = static_cast<real_type>(rand()) / static_cast<real_type>(RAND_MAX); 
    r = (max - 1.0f) * r + 1.0f;
    particles->vel_x[i] = -1.0e-4f + 2.0f * r / max * 1.0e-4f;  
    particles->vel_y[i] = -1.0e-4f + 2.0f * r / max * 1.0e-4f; 
    particles->vel_z[i] = -1.0e-4f + 2.0f * r / max * 1.0e-4f; 
  }
}

void Simulation :: init_grad()
{
  for(int i=0; i<get_nparts(); ++i)
  {
    particles->grad_p_x[i] = 0.f;
    particles->grad_p_y[i] = 0.f;
    particles->grad_p_z[i] = 0.f;
  }
}

void Simulation :: init_grav()
{
    for(int i=0; i<get_nparts(); ++i)
    {
        particles->grav_f_x[i] = 0.f;
        particles->grav_f_x[i] = 0.f;
        particles->grav_f_x[i] = 0.f;
    }
}

void Simulation :: init_mass() 
{
  int gen = 42;
  srand(gen);
  real_type n   = static_cast<real_type> (get_nparts());
  real_type max = static_cast<real_type> (RAND_MAX);

  for(int i=0; i<get_nparts(); ++i)
  {
    real_type r = static_cast<real_type>(rand()) / static_cast<real_type>(RAND_MAX); 
    r = (max - 1.0f) * r + 1.0f;
    particles->mass[i] =  n + n * r / max; 
  }
}


void Simulation :: find_neighbours()
{
    for(int i=0; i<get_nparts(); ++i)
    {
        
    }
}

void Simulation :: start() 
{
  real_type energy;
  real_type dt = get_tstep();
  int n = get_nparts();
  int i,j;
  
  const int alignment = 32; // depends on real_type?

  //particles = static_cast<Particle*>(std::aligned_alloc(alignment, sizeof(Particle)));
  //particles = (Particle*) aligned_alloc(alignment, sizeof(Particle));
  //particles = (Particle*) _mm_malloc(sizeof(Particle),alignment);
  particles = (Particle*) malloc(sizeof(Particle));

  // particles->pos_x = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  // particles->pos_y = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  // particles->pos_z = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  // particles->vel_x = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  // particles->vel_y = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  // particles->vel_z = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  // particles->acc_x = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  // particles->acc_y = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  // particles->acc_z = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  // particles->mass  = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  particles->pos_x = (real_type*) malloc(n*sizeof(real_type));
  particles->pos_y = (real_type*) malloc(n*sizeof(real_type));
  particles->pos_z = (real_type*) malloc(n*sizeof(real_type));
  particles->vel_x = (real_type*) malloc(n*sizeof(real_type));
  particles->vel_y = (real_type*) malloc(n*sizeof(real_type));
  particles->vel_z = (real_type*) malloc(n*sizeof(real_type));
  particles->grad_p_x = (real_type*) malloc(n*sizeof(real_type));
  particles->grad_p_y = (real_type*) malloc(n*sizeof(real_type));
  particles->grad_p_z = (real_type*) malloc(n*sizeof(real_type));
  particles->grav_f_x = (real_type*) malloc(n*sizeof(real_type));
  particles->grav_f_y = (real_type*) malloc(n*sizeof(real_type));
  particles->grav_f_z = (real_type*) malloc(n*sizeof(real_type));
  particles->mass  = (real_type*) malloc(n*sizeof(real_type));
  
  init_pos();	
  init_vel();
  init_grad();
  init_grav();
  init_mass();
  
  print_header();
  
  totTime = 0.; 
  
  const float softeningSquared = 1.e-3f;
  const float G = 6.67259e-11f;
  
  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  double nd = double(n);
  double gflops = 1e-9 * ( (11. + 18. ) * nd*nd  +  nd * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;
  
  const int tileSize = 8; //depends on real_type

  const double t0 = time.start();
  for (int s=1; s<=get_nsteps(); ++s)
  {   
    ts0 += time.start();
    #pragma omp parallel for 
    for (int ii = 0; ii < n; ii += tileSize )
    {
      real_type acc_xtile[tileSize];
      real_type acc_ytile[tileSize] ;
      real_type acc_ztile[tileSize];
      #pragma omp simd
      for(int s=0; s<tileSize; s++)
      {
        acc_xtile[s] = 0.0f;
        acc_ytile[s] = 0.0f;
        acc_ztile[s] = 0.0f;
      }
      // __assume_aligned(particles->pos_x, alignment);
      // __assume_aligned(particles->pos_y, alignment);
      // __assume_aligned(particles->pos_z, alignment);
      // __assume_aligned(particles->acc_x, alignment);
      // __assume_aligned(particles->acc_y, alignment);
      // __assume_aligned(particles->acc_z, alignment);
      // __assume_aligned(particles->mass, alignment);
      __builtin_assume_aligned(particles->pos_x, alignment);
      __builtin_assume_aligned(particles->pos_y, alignment);
      __builtin_assume_aligned(particles->pos_z, alignment);
      __builtin_assume_aligned(particles->grad_p_x, alignment);
      __builtin_assume_aligned(particles->grad_p_y, alignment);
      __builtin_assume_aligned(particles->grad_p_z, alignment);
      __builtin_assume_aligned(particles->grav_f_x, alignment);
      __builtin_assume_aligned(particles->grav_f_y, alignment);
      __builtin_assume_aligned(particles->grav_f_z, alignment);
      __builtin_assume_aligned(particles->mass, alignment);

      real_type ax_i = particles->acc_x[i];
      real_type ay_i = particles->acc_y[i];
      real_type az_i = particles->acc_z[i];
      #pragma omp simd
      for (j = 0; j < n; j++)
      {
        for (int i = ii; i < ii + tileSize; i++)
        {
          real_type dx, dy, dz;
          real_type distanceSqr = 0.0f;
          real_type distanceInv = 0.0f;

          dx = particles->pos_x[j] - particles->pos_x[i];	//1flop
          dy = particles->pos_y[j] - particles->pos_y[i];	//1flop	
          dz = particles->pos_z[j] - particles->pos_z[i];	//1flop

          distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
          distanceInv = 1.0f / sqrtf(distanceSqr);			//1div+1sqrt

        	//acc_xtile[i-ii] += dx * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
        	//acc_ytile[i-ii] += dy * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
        	//acc_ztile[i-ii] += dz * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
        }
      }
     #pragma omp simd
      for(int s=0; s<tileSize; s++)
      {
        //particles->acc_x[s+ii] = acc_xtile[s];
        //particles->acc_y[s+ii] = acc_ytile[s];
        //particles->acc_z[s+ii] = acc_ztile[s];
      }
    }
    energy = 0;
    #pragma omp parallel for reduction(+:energy)
    for (i = 0; i < n; ++i)// update position
    {
      particles->vel_x[i] += particles->acc_x[i] * dt; //2flops
      particles->vel_y[i] += particles->acc_y[i] * dt; //2flops
      particles->vel_z[i] += particles->acc_z[i] * dt; //2flops

      particles->pos_x[i] += particles->vel_x[i] * dt; //2flops
      particles->pos_y[i] += particles->vel_y[i] * dt; //2flops
      particles->pos_z[i] += particles->vel_z[i] * dt; //2flops

      particles->grad_p_x[i] = 0.;
      particles->grad_p_y[i] = 0.;
      particles->grad_p_z[i] = 0.;
      
      particles->grav_f_x[i] = 0.;
      particles->grav_f_y[i] = 0.;
      particles->grav_f_z[i] = 0.;

      energy += particles->mass[i] * (
         particles->vel_x[i]*particles->vel_x[i] + 
               particles->vel_y[i]*particles->vel_y[i] +
               particles->vel_z[i]*particles->vel_z[i]); //7flops
    }
  
    kenergy = 0.5 * energy; 
    
    ts1 += time.stop();
    if(!(s%get_sfreq()) ) 
    {
      nf += 1;      
      std::cout << " " 
		<<  std::left << std::setw(8)  << s
		<<  std::left << std::setprecision(5) << std::setw(8)  << s*get_tstep()
		<<  std::left << std::setprecision(5) << std::setw(12) << kenergy
		<<  std::left << std::setprecision(5) << std::setw(12) << (ts1 - ts0)
		<<  std::left << std::setprecision(5) << std::setw(12) << gflops*get_sfreq()/(ts1 - ts0)
		<<  std::endl;
      if(nf > 2) 
      {
	av  += gflops*get_sfreq()/(ts1 - ts0);
	dev += gflops*get_sfreq()*gflops*get_sfreq()/((ts1-ts0)*(ts1-ts0));
      }
      
      ts0 = 0;
      ts1 = 0;
    }
  
  } //end of the time step loop
  
  const double t1 = time.stop();
  totTime  = (t1-t0);
  totFlops = gflops*get_nsteps();
  
  av/=(double)(nf-2);
  dev=sqrt(dev/(double)(nf-2)-av*av);
  
  int nthreads=1;
  #pragma omp parallel
  nthreads=omp_get_num_threads();
  
  std::cout << std::endl;
  std::cout << "# Number Threads     : " << nthreads << std::endl;	   
  std::cout << "# Total Time (s)     : " << totTime << std::endl;
  std::cout << "# Average Perfomance : " << av << " +- " <<  dev << std::endl;
  std::cout << "===============================" << std::endl;

}


void Simulation :: print_header()
{
	    
  std::cout << " nParts = " << get_nparts()  << "; " 
	    << "nSteps = " << get_nsteps() << "; " 
	    << "dt = "     << get_tstep()  << std::endl;
	    
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " " 
	    <<  std::left << std::setw(8)  << "s"
	    <<  std::left << std::setw(8)  << "dt"
	    <<  std::left << std::setw(12) << "kenergy"
	    <<  std::left << std::setw(12) << "time (s)"
	    <<  std::left << std::setw(12) << "GFlops"
	    <<  std::endl;
  std::cout << "------------------------------------------------" << std::endl;


}

Simulation :: ~Simulation()
{
  // _mm_free(particles->pos_x);
  // _mm_free(particles->pos_y);
  // _mm_free(particles->pos_z);
  // _mm_free(particles->vel_x);
  // _mm_free(particles->vel_y);
  // _mm_free(particles->vel_z);
  // _mm_free(particles->acc_x);
  // _mm_free(particles->acc_y);
  // _mm_free(particles->acc_z);
  // _mm_free(particles->mass);
  // _mm_free(particles);
  free(particles->pos_x);
  free(particles->pos_y);
  free(particles->pos_z);
  free(particles->vel_x);
  free(particles->vel_y);
  free(particles->vel_z);
  free(particles->grad_p_x);
  free(particles->grad_p_y);
  free(particles->grad_p_z);
  free(particles->grav_f_x);
  free(particles->grav_f_y);
  free(particles->grav_f_z);
  free(particles->mass);
  free(particles);
}
