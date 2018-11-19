#include <cstdlib>
#include <cstdio>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <cmath>
#include <algorithm>

#include "BBox.hpp"
#include "HTree.hpp"
#include "HTreeScheduler.hpp"

using namespace std;
using namespace sphexa;

#define START omp_get_wtime()
#define STOP (double)(omp_get_wtime()-start)

class EvrardAdaptor
{
public:
	vector<double> x, y, z, h;
	int procSize;
	BBox bbox;

	EvrardAdaptor() = delete;

	EvrardAdaptor(int procSize) :
		procSize(procSize) { resize(procSize); }

	~EvrardAdaptor() = default;

	// Modifiers
	inline void resize(unsigned int n)
	{
		x.resize(n);
		y.resize(n);
		z.resize(n);
		h.resize(n);
	}

	inline void swap(unsigned int i, unsigned int j)
	{
		double xtmp = x[i];
		double ytmp = y[i];
		double ztmp = z[i];
		double htmp = h[i];

		x[i] = x[j];
		y[i] = y[j];
		z[i] = z[j];
		h[i] = h[j];

		x[j] = xtmp;
		y[j] = ytmp;
		z[j] = ztmp;
		h[j] = htmp;
	}

	inline void send(const std::vector<int> &list, int rank, int tag) const
	{
		unsigned int count = list.size();

		double xbuff[count], ybuff[count], zbuff[count], hbuff[count];

		for(unsigned int i=0; i<count; i++)
		{
			xbuff[i] = x[list[i]];
			ybuff[i] = y[list[i]];
			zbuff[i] = z[list[i]];
			hbuff[i] = h[list[i]];
		}

		MPI_Send(xbuff, count, MPI_DOUBLE, rank, tag, MPI_COMM_WORLD);
		MPI_Send(ybuff, count, MPI_DOUBLE, rank, tag+1, MPI_COMM_WORLD);
		MPI_Send(zbuff, count, MPI_DOUBLE, rank, tag+2, MPI_COMM_WORLD);
		MPI_Send(hbuff, count, MPI_DOUBLE, rank, tag+3, MPI_COMM_WORLD);
	}

	inline void recv(int start, int count, int rank, int tag)
	{
		MPI_Status status[4];

		MPI_Recv(&x[start], count, MPI_DOUBLE, rank, tag, MPI_COMM_WORLD, &status[0]);
		MPI_Recv(&y[start], count, MPI_DOUBLE, rank, tag+1, MPI_COMM_WORLD, &status[1]);
		MPI_Recv(&z[start], count, MPI_DOUBLE, rank, tag+2, MPI_COMM_WORLD, &status[2]);
		MPI_Recv(&h[start], count, MPI_DOUBLE, rank, tag+3, MPI_COMM_WORLD, &status[3]);
	}

	template<typename T>
	inline void removeIndices(std::vector<T> &array, const std::vector<int> indices)
	{
		for(unsigned int i=0; i<indices.size(); i++)
		{
			array[indices[i]] = array.back();
			array.pop_back();
		}
	}

	inline void discard(const std::vector<int> discardList)
	{	
		removeIndices(x, discardList);
		removeIndices(y, discardList);
		removeIndices(z, discardList);
		removeIndices(h, discardList);
	}

	// Accessors
	inline unsigned int getCount() const { return x.size(); }
	inline unsigned int getProcSize() const { return procSize; }
	inline double getX(const int i) const { return x[i]; }
	inline double getY(const int i) const { return y[i]; }
	inline double getZ(const int i) const { return z[i]; }
	inline double getH(const int i) const { return h[i]; }
	inline const BBox& getBBox() const { return bbox; }
};

void loadFile(const char *filename, int n, double *x, double *y, double *z, double *h)
{
	FILE *f = fopen(filename, "rb");
	if(f)
	{
		fread(x, sizeof(double), n, f);
		fread(y, sizeof(double), n, f);
		fread(z, sizeof(double), n, f);

        double *dummy = new double[n];

        fread(dummy, sizeof(double), n, f);
        fread(dummy, sizeof(double), n, f);
        fread(dummy, sizeof(double), n, f);
        fread(dummy, sizeof(double), n, f);
        fread(dummy, sizeof(double), n, f);
        fread(dummy, sizeof(double), n, f);
        fread(h, sizeof(double), n, f);
        fread(dummy, sizeof(double), n, f);

        delete[] dummy;

        fclose(f);
	}
	else
	{
		printf("Error opening file.\n");
		exit(1);
	}
}

void printBBox(const BBox &bbox, int comm_rank, int comm_size, const char *processor_name)
{
	printf("(%d/%d,%s) \tx[%f %f]\n", comm_rank, comm_size, processor_name, bbox.xmin, bbox.xmax);
	printf("(%d/%d,%s) \ty[%f %f]\n", comm_rank, comm_size, processor_name, bbox.ymin, bbox.ymax);
	printf("(%d/%d,%s) \tz[%f %f]\n", comm_rank, comm_size, processor_name, bbox.zmin, bbox.zmax);
}

int main()
{
    MPI_Init(NULL, NULL);

	int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

	int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);

    int n = 1e6;
    int proc_size = n/comm_size;

    double start = 0;

	EvrardAdaptor dataset(proc_size);

	start = START;
	// Load file on root node, then scatter data
	{
		double *xt = NULL, *yt = NULL, *zt = NULL, *ht = NULL;

	    if(comm_rank == 0)
	    {
	    	xt = new double[n];
			yt = new double[n];
			zt = new double[n];
			ht = new double[n];

	    	loadFile("../../bigfiles/Evrard3D.bin", n, xt, yt, zt, ht);
	    }

		MPI_Scatter(xt, proc_size, MPI_DOUBLE, &dataset.x[0], proc_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(yt, proc_size, MPI_DOUBLE, &dataset.y[0], proc_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(zt, proc_size, MPI_DOUBLE, &dataset.z[0], proc_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(ht, proc_size, MPI_DOUBLE, &dataset.h[0], proc_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if(comm_rank == 0)
		{
			delete[] xt;
			delete[] yt;
			delete[] zt;
			delete[] ht;
		}

		printf("(%d/%d,%s) Processing %d / %d particles\n", comm_rank, comm_size, processor_name, proc_size, n);
	}
	printf("LOADING AND SCATTERING TIME: %f\n", STOP);

	std::vector<int> nvi;
	std::vector<int> ng;

	// Build global distributed tree
	{
		std::vector<int> computeList(dataset.getCount());

		for(unsigned int i=0; i<computeList.size(); i++)
			computeList[i] = i;

		HTree<EvrardAdaptor> tree(dataset);
		HTreeScheduler<EvrardAdaptor> scheduler(MPI_COMM_WORLD, dataset);

		start = START;
		scheduler.balance(computeList);
		printf("LOAD BALANCE TIME: %f\n", STOP);

		int count = computeList.size();

		printf("computeList.size: %d, dataset.size: %d\n", count, dataset.getCount());

		start = START;
		tree.build(scheduler.globalBBox);
		printf("BUILD TIME: %f\n", STOP);

		int ngmax = 150;
		nvi.resize(count);
		ng.resize(count*150);

		for(int i=0; i<count; i++)
			nvi[i] = 0;

		int totalLoad = count;
		MPI_Allreduce(MPI_IN_PLACE, &totalLoad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		start = START;
		#pragma omp parallel for schedule(guided)
		for(int i=0; i<count; i++)
		{
			int id = computeList[i];
			double xi = dataset.x[id];
			double yi = dataset.y[id];
			double zi = dataset.z[id];
			double hi = dataset.h[id];

			tree.findNeighbors(xi, yi, zi, 2*hi, ngmax, &ng[(long)id*ngmax], nvi[id]);
		}
		printf("FIND TIME: %f\n", STOP);

		int totalNeighbors = 0;
		for(int i=0; i<count; i++)
			totalNeighbors += nvi[i];

		MPI_Allreduce(MPI_IN_PLACE, &totalNeighbors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		printf("Total neighbors found: %d\n", totalNeighbors);
		printf("Total cells: %d\n", tree.cellCount());
		printf("Total buckets: %d\n", tree.bucketCount());

		if(comm_rank == 0)
		{
			printf("Local domain:\n");
			printBBox(scheduler.localBBox, comm_rank, comm_size, processor_name);
			printf("Global domain:\n");
			printBBox(scheduler.globalBBox, comm_rank, comm_size, processor_name);
		}

		// scheduler.balance(computeList);
		// if(comm_rank == 0)
		// {
		// 	printf("Local domain:\n");
		// 	printBBox(scheduler.localBBox, comm_rank, comm_size, processor_name);
		// 	printf("Global domain:\n");
		// 	printBBox(scheduler.globalBBox, comm_rank, comm_size, processor_name);
		// }
	}

    MPI_Finalize();

	return 0;
}
