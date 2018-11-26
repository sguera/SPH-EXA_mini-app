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

	// inline void swap(unsigned int i, unsigned int j)
	// {
	// 	double xtmp = x[i];
	// 	double ytmp = y[i];
	// 	double ztmp = z[i];
	// 	double htmp = h[i];

	// 	x[i] = x[j];
	// 	y[i] = y[j];
	// 	z[i] = z[j];
	// 	h[i] = h[j];

	// 	x[j] = xtmp;
	// 	y[j] = ytmp;
	// 	z[j] = ztmp;
	// 	h[j] = htmp;
	// }

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

	// RecvNew(field, count) : need pointer
	// RecvUpdate(field, count, list) : need pointer
	
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

    double start = 0, stop = 0;

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
	}
	stop = STOP;
	printf("(%d/%d,%s) LOAD SCATTER TIME: %f\n", comm_rank, comm_size, processor_name, stop);

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
		stop = STOP;

		printf("(%d/%d,%s) BALANCE TIME: %f\n", comm_rank, comm_size, processor_name, stop);
		MPI_Barrier(MPI_COMM_WORLD);

		start = START;
		scheduler.findGhosts(computeList);
		stop = STOP;

		printf("(%d/%d,%s) FIND GHOST TIME: %f\n", comm_rank, comm_size, processor_name, stop);
		MPI_Barrier(MPI_COMM_WORLD);

		int before = dataset.getCount();
		start = START;
		scheduler.updateGhosts();
		stop = STOP;
		int after = dataset.getCount();

		printf("(%d/%d,%s) UPDATE GHOST TIME: %f -- compute: %d, ghosts: %d\n", comm_rank, comm_size, processor_name, stop, before, after-before);
		MPI_Barrier(MPI_COMM_WORLD);

		start = START;
		tree.build(scheduler.globalBBox);
		stop = STOP;
		
		printf("(%d/%d,%s) BUILD TIME: %f\n", comm_rank, comm_size, processor_name, stop);
		MPI_Barrier(MPI_COMM_WORLD);

		int count = computeList.size();
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
		stop = STOP;

		printf("(%d/%d,%s) FIND TIME: %f\n", comm_rank, comm_size, processor_name, stop);
		MPI_Barrier(MPI_COMM_WORLD);

		int totalNeighbors = 0;
		for(int i=0; i<count; i++)
			totalNeighbors += nvi[i];

		MPI_Allreduce(MPI_IN_PLACE, &totalNeighbors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		if(comm_rank == 0) printf("(%d/%d,%s) Total neighbors found: %d\n", comm_rank, comm_size, processor_name, totalNeighbors);
	}

    MPI_Finalize();

	return 0;
}
