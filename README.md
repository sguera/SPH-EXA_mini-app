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
The parallelism is currently expressed via MPI+OpenMP.

You can use the following commands to compile and run the SquarePatch example:

#### Compile

* OpenMP: ```shell make sqpatch```
* MPI+OpenMP: ```shell make mpisqpatch```

#### Run

* OpenMP: ```shell bin/sqpatch.app```
* MPI+OpenMP: ```shell bin/mpisqpatch.app```

## Authors

* **Danilo Guerrera**
* **Aurélien Cavelan**
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
