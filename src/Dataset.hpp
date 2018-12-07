#pragma once

class Dataset
{
	public:
		Dataset(const char *filename)
		{
			n = 1e6;

			//reserve the space for each vector
			x.resize(n);
			y.resize(n);
			z.resize(n);
			x_m1.resize(n);
			y_m1.resize(n);
			z_m1.resize(n);
			vx.resize(n);
			vy.resize(n);
			vz.resize(n);
			ro.resize(n);
			u.resize(n);
			p.resize(n);
			h.resize(n);
			m.resize(n);

			c.resize(n);
			cv.resize(n);
			temp.resize(n);
			mue.resize(n);
			mui.resize(n);

			nvi.resize(n);

			grad_P_x.resize(n);
			grad_P_y.resize(n);
			grad_P_z.resize(n);

			d_u.resize(n);
			d_u_m1.resize(n);

			timestep.resize(n);
			timestep_m1.resize(n);

			ng.resize(n*ngmax);

			FILE *f = fopen(filename, "rb");
			if(f)
			{
			//     //fread(&n, sizeof(int), 1, f);

				fread(&x[0], sizeof(double), n, f);
				fread(&y[0], sizeof(double), n, f);
				fread(&z[0], sizeof(double), n, f);
				fread(&vx[0], sizeof(double), n, f);
				fread(&vy[0], sizeof(double), n, f);
				fread(&vz[0], sizeof(double), n, f);
				fread(&ro[0], sizeof(double), n, f);
				fread(&u[0], sizeof(double), n, f);
				fread(&p[0], sizeof(double), n, f);
				fread(&h[0], sizeof(double), n, f);
				fread(&m[0], sizeof(double), n, f);

				fclose(f);

			//     #pragma omp parallel for
			//     for(int i=0; i<n; i++)
			//     {
			//         temp[i] = 1.0;
			//         mue[i] = 2.0;
			//         mui[i] = 10.0;
			//         vx[i] = 0.0;
			//         vy[i] = 0.0;
			//         vz[i] = 0.0;
			//     }

			//     ngmax = 150;
			//     nvi = new int[n](); //adding the () at the end equals to a memset to 0
				
			//     grad_P_x = new double[n]();
			//     grad_P_y = new double[n]();
			//     grad_P_z = new double[n]();

			//     d_u = new double[n]();
			//     d_u_m1 = new double[n]();

			//     timestep = new double[n]();
			//     timestep_m1 = new double[n]();

			//     ng = new int[n*ngmax];

			//     iteration = 0;
			}
			else
			{
				printf("Error opening file %s\n", filename);
				exit(EXIT_FAILURE);
			}


			// // input file stream
			// ifstream inputfile(filename, std::ios::binary);

			// // read the contents of the file into the vectors
			// inputfile.read(reinterpret_cast<char*>(x.data()), x.size());
			// inputfile.read(reinterpret_cast<char*>(y.data()), y.size());
			// inputfile.read(reinterpret_cast<char*>(z.data()), z.size());
			// inputfile.read(reinterpret_cast<char*>(x_m1.data()), x_m1.size());
			// inputfile.read(reinterpret_cast<char*>(y_m1.data()), y_m1.size());
			// inputfile.read(reinterpret_cast<char*>(z_m1.data()), z_m1.size());
			// inputfile.read(reinterpret_cast<char*>(vx.data()), vx.size());
			// inputfile.read(reinterpret_cast<char*>(vy.data()), vy.size());
			// inputfile.read(reinterpret_cast<char*>(vz.data()), vz.size());
			// inputfile.read(reinterpret_cast<char*>(ro.data()), ro.size());
			// inputfile.read(reinterpret_cast<char*>(u.data()), u.size());
			// inputfile.read(reinterpret_cast<char*>(p.data()), p.size());
			// inputfile.read(reinterpret_cast<char*>(h.data()), h.size());
			// inputfile.read(reinterpret_cast<char*>(m.data()), m.size());
			// inputfile.read(reinterpret_cast<char*>(c.data()), c.size());
			// inputfile.read(reinterpret_cast<char*>(cv.data()), cv.size());
			// inputfile.read(reinterpret_cast<char*>(temp.data()), temp.size());
			// inputfile.read(reinterpret_cast<char*>(mue.data()), mue.size());
			// inputfile.read(reinterpret_cast<char*>(mui.data()), mui.size());


			fill(temp.begin(), temp.end(), 1.0);
			fill(mue.begin(), mue.end(), 2.0);
			fill(mui.begin(), mui.end(), 10.0);
			fill(vx.begin(), vx.end(), 0.0);
			fill(vy.begin(), vy.end(), 0.0);
			fill(vz.begin(), vz.end(), 0.0);

			
			
			fill(nvi.begin(), nvi.end(), 0);


			
			// fill(grad_P_x.begin(), grad_P_x.end(), 0.0);
			fill(grad_P_y.begin(), grad_P_y.end(), 0.0);
			fill(grad_P_z.begin(), grad_P_z.end(), 0.0);

			
			fill(d_u.begin(), d_u.end(), 0.0);
			fill(d_u_m1.begin(), d_u_m1.end(), 0.0);

			
			fill(timestep.begin(), timestep.end(), 0.0);
			fill(timestep_m1.begin(), timestep_m1.end(), 0.0);

			
			fill(ng.begin(), ng.end(), 0);

			iteration = 0;


			
		}

		~Dataset(){}

		DataAccessor<double> pos(x);

	private:
	
		int n; // Number of particles
		std::vector<double> x, y, z, x_m1, y_m1, z_m1; // Positions
		std::vector<double> vx, vy, vz; // Velocities
		std::vector<double> ro; // Density
		std::vector<double> u; // Internal Energy
		std::vector<double> p; // Pressure
		std::vector<double> h; // Smoothing Length
		std::vector<double> m; // Mass
		std::vector<double> c; // Speed of sound
		std::vector<double> cv; // Specific heat
		std::vector<double> temp; // Temperature
		std::vector<double> mue; // Mean molecular weigh of electrons
		std::vector<double> mui; // Mean molecular weight of ions

		std::vector<double> grad_P_x, grad_P_y, grad_P_z; //gradient of the pressure
		std::vector<double> d_u, d_u_m1; //variation of the energy
		std::vector<double> timestep, timestep_m1;

		int ngmax = 150; // Maximum number of neighbors per particle
		std::vector<int> nvi; // Number of neighbors per particle
		std::vector<int> ng; // List of neighbor indices per particle.

		// Periodic boundary conditions
		bool PBCx = false, PBCy = false, PBCz = false;
		
		// Global bounding box (of the domain)
		double xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0, zmin = -1.0, zmax = 1.0;

		int iteration;
};

// template<typename T>
// DataAccessor {
	
// 	DataAccessor(&x);
	
// 	inline const get(int i){
// 		return x[i];
// 	}

// 	inline set(int i, T value){
// 		x[i] = value;

// 	private:
// 		vector<T> &x;
// 	}
// }