#ifndef EVRARD_HPP
#define EVRARD_HPP

class Evrard
{
    public:
        Evrard(const char *filename)
        {
            FILE *f = fopen(filename, "rb");
            if(f)
            {
                //fread(&n, sizeof(int), 1, f);
                n = 1e6;
                
                x = new double[n];
                y = new double[n];
                z = new double[n];
                x_m1 = new double[n];
                y_m1 = new double[n];
                z_m1 = new double[n];
                vx = new double[n];
                vy = new double[n];
                vz = new double[n];
                ro = new double[n];
                u = new double[n];
                p = new double[n];
                h = new double[n];
                m = new double[n];
                c = new double[n];
                cv = new double[n];
                temp = new double[n];
                mue = new double[n];
                mui = new double[n];

                fread(x, sizeof(double), n, f);
                fread(y, sizeof(double), n, f);
                fread(z, sizeof(double), n, f);
                fread(vx, sizeof(double), n, f);
                fread(vy, sizeof(double), n, f);
                fread(vz, sizeof(double), n, f);
                fread(ro, sizeof(double), n, f);
                fread(u, sizeof(double), n, f);
                fread(p, sizeof(double), n, f);
                fread(h, sizeof(double), n, f);
                fread(m, sizeof(double), n, f);
                
                fclose(f);

                #pragma omp parallel for
                for(int i=0; i<n; i++)
                {
                    temp[i] = 1.0;
                    mue[i] = 2.0;
                    mui[i] = 10.0;
                }

                ngmax = 150;
                nvi = new int[n](); //adding the () at the end equals toa memset to 0
                
                grad_P_x = new double[n]();
                grad_P_y = new double[n]();
                grad_P_z = new double[n]();

                d_u_x = new double[n]();
                d_u_y = new double[n]();
                d_u_z = new double[n]();
                d_u_x_m1 = new double[n]();
                d_u_y_m1 = new double[n]();
                d_u_z_m1 = new double[n]();

                timestep = new double[n]();
                timestep_m1 = new double[n]();

                ng = new int[n*ngmax];

                iteration = 0;
            }
            else
            {
                printf("Error opening file %s\n", filename);
                exit(EXIT_FAILURE);
            }
        }

        ~Evrard()
        {
            delete[] x;
            delete[] y;
            delete[] z;
            delete[] x_m1;
            delete[] y_m1;
            delete[] z_m1;
            delete[] vx;
            delete[] vy;
            delete[] vz;
            delete[] ro;
            delete[] u;
            delete[] p;
            delete[] h;
            delete[] m;
            delete[] c;
            delete[] cv;
            delete[] temp;
            delete[] mue;
            delete[] mui;
            delete[] grad_P_x;
            delete[] grad_P_y;
            delete[] grad_P_z;
            delete[] d_u_x;
            delete[] d_u_y;
            delete[] d_u_z;
            delete[] d_u_x_m1;
            delete[] d_u_y_m1;
            delete[] d_u_z_m1;
            delete[] timestep;
            delete[] timestep_m1;
            delete[] nvi;
            delete[] ng;
        }

    int n; // Number of particles
    double *x, *y, *z, *x_m1, *y_m1, *z_m1; // Positions
    double *vx, *vy, *vz; // Velocities
    double *ro; // Density
    double *u; // Internal Energy
    double *p; // Pressure
    double *h; // Smoothing Length
    double *m; // Mass
    double *c; // Speed of sound
    double *cv; // Specific heat
    double *temp; // Temperature
    double *mue; // Mean molecular weigh of electrons
    double *mui; // Mean molecular weight of ions

    double *grad_P_x, *grad_P_y, *grad_P_z; //gradient of the pressure
    double *d_u_x, *d_u_y, *d_u_z, *d_u_x_m1, *d_u_y_m1, *d_u_z_m1; //variation of the energy
    double *timestep, *timestep_m1;

    int ngmax; // Maximum number of neighbors per particle
    int *nvi; // Number of neighbors per particle
    int *ng; // List of neighbor indices per particle.

    // Periodic boundary conditions
    bool PBCx = false, PBCy = false, PBCz = false;
    
    // Global bounding box (of the domain)
    double xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0, zmin = -1.0, zmax = 1.0;

    int iteration;
};

#endif