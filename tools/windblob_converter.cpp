#include <cstdlib>
#include <cstdio>
#include <vector>
#include <fstream> 
#include <iostream>
#include <numeric>

int main()
{
    const int n = 3157385;

    std::vector<double> x(n), y(n), z(n), vx(n), vy(n), vz(n), ro(n), p(n), h(n);
    
    printf("Opening the file\n");
    FILE *f = fopen("../data/windblob_3M.dat", "r");
    
    if(f)
    {
        for(int i=0; i<n; i++)
        {   
            fscanf(f, "%lf %lf %lf %lf %lf %lf %lf %lf %lf\n", &x[i], &y[i], &z[i], &vx[i], &vy[i], &vz[i], &ro[i], &p[i], &h[i]);
            //printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf\n", x[i], y[i], z[i], vx[i], vy[i], vz[i], promro[i], p[i], h[i]);
        }

        fclose(f);

        std::fstream ofs("../data/windblob_3M.bin", std::ofstream::out | std::ofstream::binary);
        if(ofs)
        {
            ofs.write((char*)&x[0], x.size() * sizeof(double));
            ofs.write((char*)&y[0], y.size() * sizeof(double));
            ofs.write((char*)&z[0], z.size() * sizeof(double));
            ofs.write((char*)&vx[0], vx.size() * sizeof(double));
            ofs.write((char*)&vy[0], vy.size() * sizeof(double));
            ofs.write((char*)&vz[0], vz.size() * sizeof(double));
            ofs.write((char*)&ro[0], ro.size() * sizeof(double));
            ofs.write((char*)&p[0], p.size() * sizeof(double));
            ofs.write((char*)&h[0], h.size() * sizeof(double));

            ofs.close();
        }
        else
            printf("Error: couldn't open file for writing.\n");
    }
    else
        printf("Error: couldn't open file for reading.\n");

    return 0;
}