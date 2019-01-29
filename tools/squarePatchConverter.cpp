#include <cstdlib>
#include <cstdio>
#include <vector>
#include <fstream> 
#include <iostream>
#include <numeric>
int main()
{
    const int n = 1000000;


    std::vector<double> x(n), y(n), z(n), vx(n), vy(n), vz(n), p_0(n), h(n), volume(n), m(n);
    
    printf("Opening the file\n");
    FILE *f = fopen("squarepatch3D_1M", "r");
    
    if(f)
    {
        for(int i=0; i<n; i++)
        {   
            fscanf(f, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", &x[i], &y[i], &z[i], &vx[i], &vy[i], &vz[i], &p_0[i], &h[i], &volume[i], &m[i]);
            // printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", x[i], y[i], z[i], vx[i], vy[i], vz[i], p_0[i], h[i], volume[i], m[i]);
        }

        fclose(f);

        std::ofstream ofs("squarepatch3D_1M.bin", std::ofstream::out | std::ofstream::binary);
        
        if(ofs)
        {
            ofs.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(double));
            ofs.write(reinterpret_cast<const char*>(y.data()), y.size() * sizeof(double));
            ofs.write(reinterpret_cast<const char*>(z.data()), z.size() * sizeof(double));
            ofs.write(reinterpret_cast<const char*>(vx.data()), vx.size() * sizeof(double));
            ofs.write(reinterpret_cast<const char*>(vy.data()), vy.size() * sizeof(double));
            ofs.write(reinterpret_cast<const char*>(vz.data()), vz.size() * sizeof(double));
            ofs.write(reinterpret_cast<const char*>(p_0.data()), p_0.size() * sizeof(double));

            ofs.close();
        }

        else
            printf("Error: couldn't open file for writing.\n");
    }
    else
        printf("Error: couldn't open file for reading.\n");

    return 0;
}