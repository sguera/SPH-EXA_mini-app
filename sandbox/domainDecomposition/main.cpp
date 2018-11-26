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
	int size, ngmax;
	BBox bbox;

	EvrardAdaptor() = delete;

	EvrardAdaptor(int size, int ngmax) : 
		size(size), ngmax(ngmax) { resize(size); }

	~EvrardAdaptor() = default;

	// Modifiers
	inline void resize(unsigned int n)
	{
		x.resize(n);
		y.resize(n);
		z.resize(n);
		h.resize(n);
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

	// RecvNew(field, count) : need pointer
	// RecvUpdate(field, count, list) : need pointer
	
	inline unsigned int getCount() const { return x.size(); }
	inline double getX(const int i) const { return x[i]; }
	inline double getY(const int i) const { return y[i]; }
	inline double getZ(const int i) const { return z[i]; }
	inline double getH(const int i) const { return h[i]; }
	inline const BBox& getBBox() const { return bbox; }
};

void loadFileSquarePatch(const char *filename, int n, double *x, double *y, double *z, double *h)
{
	FILE *f = fopen(filename, "rb");
	if(f)
	{
		int u1;
		double gm, gh, gd;
		fread(&u1, sizeof(int), 1, f);
		fread(&gm, sizeof(double), 1, f);
		fread(&gh, sizeof(double), 1, f);
		fread(&gd, sizeof(double), 1, f);

		fread(x, sizeof(double), n, f);
		fread(y, sizeof(double), n, f);
		fread(z, sizeof(double), n, f);

		for(int i=0; i<n; i++)
			h[i] = gh*1.5916455;

		fclose(f);
	}
	else
	{
		printf("Error opening file.\n");
		exit(1);
	}
}

void loadFileEvrard(const char *filename, int n, double *x, double *y, double *z, double *h)
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

template <typename LoadFunc>
void scatterFile(LoadFunc load, const char *filename, int comm_rank, int comm_size, int n, const std::vector<int> &procsize, const std::vector<int> &displs, int size, double *x, double *y, double *z, double *h)
{
	double *xt = NULL, *yt = NULL, *zt = NULL, *ht = NULL;

    if(comm_rank == 0)
    {
    	xt = new double[n];
		yt = new double[n];
		zt = new double[n];
		ht = new double[n];

    	load(filename, n, xt, yt, zt, ht);
    }

	MPI_Scatterv(xt, &procsize[0], &displs[0], MPI_DOUBLE, x, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(yt, &procsize[0], &displs[0], MPI_DOUBLE, y, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(zt, &procsize[0], &displs[0], MPI_DOUBLE, z, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(ht, &procsize[0], &displs[0], MPI_DOUBLE, h, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(comm_rank == 0)
	{
		delete[] xt;
		delete[] yt;
		delete[] zt;
		delete[] ht;
	}
}

template <class DatasetAdaptor, class TreeAdaptor>
void findNeighbors(const DatasetAdaptor &dataset, const TreeAdaptor &tree, const std::vector<int> &computeList, std::vector<int> &nvi, std::vector<int> &ng)
{
	for(unsigned int i=0; i<computeList.size(); i++)
		nvi[i] = 0;

	#pragma omp parallel for schedule(guided)
	for(unsigned int i=0; i<computeList.size(); i++)
	{
		int id = computeList[i];
		double xi = dataset.x[id];
		double yi = dataset.y[id];
		double zi = dataset.z[id];
		double hi = dataset.h[id];

		tree.findNeighbors(xi, yi, zi, 2*hi, dataset.ngmax, &ng[(long)id*dataset.ngmax], nvi[id]);
	}
}

void printBBox(const BBox &bbox, int comm_rank, int comm_size, const char *processor_name)
{
	printf("(%d/%d,%s) \tx[%f %f]\n", comm_rank, comm_size, processor_name, bbox.xmin, bbox.xmax);
	printf("(%d/%d,%s) \ty[%f %f]\n", comm_rank, comm_size, processor_name, bbox.ymin, bbox.ymax);
	printf("(%d/%d,%s) \tz[%f %f]\n", comm_rank, comm_size, processor_name, bbox.zmin, bbox.zmax);
}

template <typename Func>
inline double timimg(const Func &f)
{
	MPI_Barrier(MPI_COMM_WORLD);
	double start = START;
	f();
	MPI_Barrier(MPI_COMM_WORLD);
	return STOP;
}

#define TIMEFUNC(f) timimg([&]{ f; })

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

    //const char *filename = "../../bigfiles/Evrard3D.bin";
    //int n = 1e6, ngmax = 150;
    //auto loadFunc = loadFileEvrard;

    const char *filename = "../../bigfiles/squarepatch3D.bin";
    int n = 10077696, ngmax = 150;
    auto loadFunc = loadFileSquarePatch;

    int size = n / comm_size;
    int offset = n % size;

    std::vector<int> procsize(comm_size), displs(comm_size);

    {
	    int chunksize = n / comm_size;
	    int offset = n % chunksize;

	    if(comm_rank == 0) size += offset;

	    procsize[0] = chunksize+offset;
	    displs[0] = 0;

	    for(int i=1; i<comm_size; i++)
	    {
	    	procsize[i] = chunksize;
	    	displs[i] = displs[i-1]+procsize[i-1];
	    }
	}
    
    std::vector<int> nvi;
	std::vector<int> ng;

	std::vector<int> computeList;

	EvrardAdaptor dataset(size, ngmax);

	HTree<EvrardAdaptor> tree(dataset);
	HTreeScheduler<EvrardAdaptor> scheduler(MPI_COMM_WORLD, dataset);

	double totalTime = 0;

	// Load and scatter (load binary file, scatter data)
	{
		// Load file on root node, then scatter data
		double t = TIMEFUNC(scatterFile(loadFunc, filename, comm_rank, comm_size, n, procsize, displs, size, &dataset.x[0], &dataset.y[0], &dataset.z[0], &dataset.h[0]));
		if(comm_rank == 0) printf("(%d/%d,%s) LOAD SCATTER TIME: %f\n", comm_rank, comm_size, processor_name, t);

		computeList.resize(dataset.getCount());
		for(unsigned int i=0; i<computeList.size(); i++)
			computeList[i] = i;
	}

	// Distributed functions (distribute and balance particles accross nodes, find and receive halos particles)
	{

		double t = timimg([&]{ scheduler.balance(procsize, computeList) ;}); totalTime += t;
		if(comm_rank == 0) printf("(%d/%d,%s) BALANCE TIME: %f\n", comm_rank, comm_size, processor_name, t);

		t = TIMEFUNC(scheduler.findHalos(computeList)); totalTime += t;
		if(comm_rank == 0) printf("(%d/%d,%s) FIND GHOST TIME: %f\n", comm_rank, comm_size, processor_name, t);

		t = TIMEFUNC(scheduler.updateHalos()); totalTime += t;
		if(comm_rank == 0) printf("(%d/%d,%s) UPDATE GHOST TIME: %f -- compute: %zu, halos: %d\n", comm_rank, comm_size, processor_name, t, computeList.size(), scheduler.haloCount);
	}

	// Local functions (build tree, find neighbors)
	{
		double t = TIMEFUNC(tree.build(scheduler.globalBBox)); totalTime += t;
		if(comm_rank == 0) printf("(%d/%d,%s) BUILD TIME: %f\n", comm_rank, comm_size, processor_name, t);
	
		int count = computeList.size();
		nvi.resize(count);
		ng.resize(count*ngmax);

		t = TIMEFUNC(findNeighbors(dataset, tree, computeList, nvi, ng)); totalTime += t;
		if(comm_rank == 0) printf("(%d/%d,%s) FIND TIME: %f\n", comm_rank, comm_size, processor_name, t);
	}

	// Final check (total number of neighbors)
	{
		long int totalNeighbors = 0;
		for(unsigned int i=0; i<computeList.size(); i++)
			totalNeighbors += nvi[i];

		MPI_Allreduce(MPI_IN_PLACE, &totalNeighbors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		if(comm_rank == 0) printf("(%d/%d,%s) Total neighbors found: %lu\n", comm_rank, comm_size, processor_name, totalNeighbors);
	}

	if(comm_rank == 0) printf("(%d/%d,%s) Total time: %f\n", comm_rank, comm_size, processor_name, totalTime);

    MPI_Finalize();

	return 0;
}
