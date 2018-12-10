#include <cstdlib>
#include <cstdio>

int main()
{
	int n = 1000000;

	double x[n], y[n], z[n], vx[n], vy[n], vz[n], ro[n], u[n], p[n], h[n], m[n], temp[n], mue[n], mui[n];

	FILE *f = fopen("evrard_1M_ok", "r");
	if(f)
	{
		// a(i),a(i+n),a(i+n2),v(i),v(i+n),v(i+n2),&
	 //    &promro(i),u(i),p(i),h(i),masa(i)
	 //    temp(i)=1.d0
	 //    mue(i)=2.d0
	 //    mui(i)=10.d0
	 //    ballmass(i)=promro(i)*h(i)**3

		for(int i=0; i<n; i++)
		{	
			fscanf(f, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", &x[i], &y[i], &z[i], &vx[i], &vy[i], &vz[i], &ro[i], &u[i], &p[i], &h[i], &m[i]);
			//printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", x[i], y[i], z[i], vx[i], vy[i], vz[i], ro[i], u[i], p[i], h[i], m[i]);
			// temp[i] = 1.0;
			// mue[i] = 2.0;
			// mui[i] = 10.0;
		}

		fclose(f);

		FILE *fo = fopen("Evrard3D.bin", "wb");
		FILE *fh = fopen("Evrard3D.header", "w");
		if(fo && fh)
		{
			// fwrite(&n, sizeof(int), 1, fo);
			// fprintf(fh, "Number of particles (int): %lu bytes\n", sizeof(int));
			
			fwrite(x, sizeof(double), n, fo);
			fprintf(fh, "x (n * double): %lu bytes\n", n*sizeof(double));

			fwrite(y, sizeof(double), n, fo);
			fprintf(fh, "y (n * double): %lu bytes\n", n*sizeof(double));

			fwrite(z, sizeof(double), n, fo);
			fprintf(fh, "z (n * double): %lu bytes\n", n*sizeof(double));

			fwrite(vx, sizeof(double), n, fo);
			fprintf(fh, "vx (n * double): %lu bytes\n", n*sizeof(double));

			fwrite(vy, sizeof(double), n, fo);
			fprintf(fh, "vy (n * double): %lu bytes\n", n*sizeof(double));

			fwrite(vz, sizeof(double), n, fo);
			fprintf(fh, "vz (n * double): %lu bytes\n", n*sizeof(double));

			fwrite(ro, sizeof(double), n, fo);
			fprintf(fh, "ro (n * double): %lu bytes\n", n*sizeof(double));

			fwrite(u, sizeof(double), n, fo);
			fprintf(fh, "u (n * double): %lu bytes\n", n*sizeof(double));

			fwrite(p, sizeof(double), n, fo);
			fprintf(fh, "p (n * double): %lu bytes\n", n*sizeof(double));

			fwrite(h, sizeof(double), n, fo);
			fprintf(fh, "h (n * double): %lu bytes\n", n*sizeof(double));

			fwrite(m, sizeof(double), n, fo);
			fprintf(fh, "m (n * double): %lu bytes\n", n*sizeof(double));

			// fwrite(temp, sizeof(double), n, fo);
			// fprintf(fh, "temp (n * double): %lu bytes\n", n*sizeof(double));

			// fwrite(mue, sizeof(double), n, fo);
			// fprintf(fh, "mue (n * double): %lu bytes\n", n*sizeof(double));

			// fwrite(mui, sizeof(double), n, fo);
			// fprintf(fh, "mui (n * double): %lu bytes\n", n*sizeof(double));

			fclose(fo);
			fclose(fh);
		}
		else
			printf("Error: couldn't open file for writing.\n");
	}
	else
		printf("Error: couldn't open file for reading.\n");

	return 0;
}