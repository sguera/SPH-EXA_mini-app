[![Build Status](https://api.travis-ci.org/unibas-dmi-hpc/SPH-EXA_mini-app.svg?branch=develop)](https://travis-ci.org/unibas-dmi-hpc/SPH-EXA_mini-app)


# SPH

The smooth particle hydrodynamics (SPH) technique is a purely Lagrangian method.
SPH discretizes a fluid in a series of interpolation points (SPH particles) 
whose distribution follows the mass density of the fluid and their evolution relies 
on a weighted interpolation over close neighboring particles.

SPH simulations represent computationally demanding calculations. 
Therefore, trade-offs are made between temporal and spatial scales, resolution, 
dimensionality (3-D or 2-D), and approximated versions of the physics involved. 
The parallelization of SPH codes is not trivial due to their boundless nature 
and the absence of a structured particle grid. 
[SPHYNX](https://astro.physik.unibas.ch/sphynx/), 
[ChaNGa](http://faculty.washington.edu/trq/hpcc/tools/changa.html), 
and [SPH-flow](http://www.sph-flow.com) are the three SPH codes selected in the PASC SPH-EXA project proposal. 
The performance of these codes is negatively impacted by factors, such as multiple time-stepping and gravity. 
Therefore, the goal is to extrapolate their common basic SPH features, which are consolidated in a fully optimized, Exascale-ready, MPI+X, pure-SPH, mini-app. 

# SPH-EXA mini-app

SPH-EXA mini-app is a C++14 headers-only code with no external software dependencies. 
The parallelism is currently expressed via following models: MPI, OpenMP, OpenMP4.5 target offloading, OpenACC and CUDA.

[Check our wiki for more details](https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app/wiki)

#### Folder structure

```
SPH-EXA
|   README.md
└───include/        - folder containing all sph functions,
|                     utilities and helpers that are shared between all test cases.
└───src/            - folder containing test cases
    └───testcase/   - folder containing test case main function and 
                      other test case specific functions, 
                      like reading input data from the file
```
#### Compile

Use the following commands to compile and run the SquarePatch example:

* OpenMP: ```make omp```
* OpenMP + CUDA: ```make omp+cuda```
* MPI + OpenMP: ```make mpi+omp```
* MPI + OpenMP + OpenMP 4.5 Offloading: ```mpi+omp+target```
* MPI + OpenMP + CUDA: ```make mpi+omp+cuda```
* MPI + OpenMP + OpenACC: ```make mpi+omp+acc```

Compiled binaries are placed in bin/ in the project root folder.

#### Run

To run the SPH-EXA type ```shell bin/{compiled_parallel_model}.app arguments```

Possible arguments for the Square Patch test case:  
* ```-n NUM``` : Run the simulation with NUM^3 (NUM to the cube) number of particles  
* ```-s NUM``` : Run the simulation with NUM of iterations (time-steps)  
* ```-w NUM``` : Dump particle data every NUM iterations (time-steps)  
* ```--quiet``` : Don't print any output to stdout  

Example usage:  
* ```./bin/omp.app -n 100 -s 1000 -w 10``` Runs the Square Patch simulation with 1 million particles for 1000 iterations (time-steps) with OpenMP and dumps particles data every 10 iterations  
* ```./bin/omp+cuda.app -n 20 -s 500``` Runs the Square Patch simulation with 8 thousands particles for 500 iterations (time-steps) with OpenMP and CUDA  
* ```mpirun bin/mpi+omp+cuda.app -n 500 -s 10``` Runs the Square Patch simulation with 125 million particles for 10 iterations (time-steps) with MPI, OpenMP and CUDA  
* ```mpirun bin/mpi+omp+target.app -n 100 -s 10000``` Runs the Square Patch simulation with 1 million particles for 10000 iterations (time-steps) with MPI, OpenMP and OpenMP4.5 target offloading  

## Authors

* **Danilo Guerrera**
* **Aurelien Cavelan**
* **Michal Grabarczyk**
* **jg piccinali**
* **David Imbert**
* **Ruben Cabezon**
* **Darren Reed**
* **Lucio Mayer**
* **Ali Mohammed**
* **Florina Ciorba**
* **Tom Quinn**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* PASC SPH-EXA project
