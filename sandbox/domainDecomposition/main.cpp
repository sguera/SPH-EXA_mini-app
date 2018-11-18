#include <cstdlib>
#include <cstdio>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <cmath>
#include <algorithm>

#include "HTree.hpp"

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

	// DatasetAdaptor Implementaion
	void computeBoundingBox()
	{
		double xmin = INFINITY;
		double xmax = -INFINITY;
		double ymin = INFINITY;
		double ymax = -INFINITY;
		double zmin = INFINITY;
		double zmax = -INFINITY;

		for(unsigned int i=0; i<x.size(); i++)
		{
			if(x[i] < xmin) xmin = x[i];
			if(x[i] > xmax) xmax = x[i];
			if(y[i] < ymin) ymin = y[i];
			if(y[i] > ymax) ymax = y[i];
			if(z[i] < zmin) zmin = z[i];
			if(z[i] > zmax) zmax = z[i];
		}

		bbox = BBox(xmin, xmax, ymin, ymax, zmin, zmax);
	}

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
		int recvedTotal[4];
		for(int i=0; i<4; i++)
			recvedTotal[i] = 0;

		while(recvedTotal[0] < count)
		{
			MPI_Status status[4];

			MPI_Recv(&x[start+recvedTotal[0]], count, MPI_DOUBLE, rank, tag, MPI_COMM_WORLD, &status[0]);
			MPI_Recv(&y[start+recvedTotal[1]], count, MPI_DOUBLE, rank, tag+1, MPI_COMM_WORLD, &status[1]);
			MPI_Recv(&z[start+recvedTotal[2]], count, MPI_DOUBLE, rank, tag+2, MPI_COMM_WORLD, &status[2]);
			MPI_Recv(&h[start+recvedTotal[3]], count, MPI_DOUBLE, rank, tag+3, MPI_COMM_WORLD, &status[3]);

			int recved[4];
			for(int i=0; i<4; i++)
			{
				MPI_Get_count(&status[i], MPI_DOUBLE, &recved[i]);
				recvedTotal[i] += recved[i];
			}
		}
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

	EvrardAdaptor dataset(proc_size);

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

	// Compute global domain bounding box
	{
		dataset.computeBoundingBox();

		if(comm_rank == 0)
		{
			printf("(%d/%d,%s) Domain x[%f %f]\n", comm_rank, comm_size, processor_name, dataset.bbox.xmin, dataset.bbox.xmax);
			printf("(%d/%d,%s) Domain y[%f %f]\n", comm_rank, comm_size, processor_name, dataset.bbox.ymin, dataset.bbox.ymax);
			printf("(%d/%d,%s) Domain z[%f %f]\n", comm_rank, comm_size, processor_name, dataset.bbox.zmin, dataset.bbox.zmax);
		}
	}

	// Build global distributed tree
	{
		std::vector<int> computeList(dataset.getCount());

		for(unsigned int i=0; i<computeList.size(); i++)
			computeList[i] = i;

		HTree<EvrardAdaptor> tree(dataset);
		tree.build(computeList);
		tree.build(computeList);
		tree.build(computeList);
		tree.build(computeList);
		tree.build(computeList);
		tree.build(computeList);
		tree.build(computeList);
		tree.build(computeList);
		// NEED TO MAINTAIN COMPUTE LIST
	}

    MPI_Finalize();

	return 0;
}
