#pragma once

#include <cmath>
#include <memory>

// TODO
// Make assign rank recursive to assign load more finely:
// If a cell cannot be assigned, call build recursively (else stop)
// Make discardList, exchange, and others recursive as well (global cell id needed)
// Finish building

namespace sphexa
{

class BBox
{
public:
	BBox(double xmin = -1, double xmax = 1, double ymin = -1, double ymax = 1, double zmin = -1, double zmax = 1) : 
		xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax) {}

	double xmin, xmax, ymin, ymax, zmin, zmax;
};

struct HTreeParams
{
	HTreeParams(int bucketSize = 64) :
		bucketSize(bucketSize) {}
	int bucketSize = 64;
};

template<class DatasetAdaptor>
class Cell
{
public:
	DatasetAdaptor &dataset;

	BBox bbox;

	double maxH;

	int ncells;
	int nX, nY, nZ;

	std::vector<std::shared_ptr<Cell>> cells;

	int comm_size, comm_rank, name_len;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	Cell() = delete;
	
	~Cell() = default;

	Cell(DatasetAdaptor &dataset) : 
		dataset(dataset) {}

	inline double normalize(double d, double min, double max)
	{
		return (d-min)/(max-min);
	}

	inline double computeMaxH(const std::vector<int> &list)
	{
		double hmax = 0.0;
		for(unsigned int i=0; i<list.size(); i++)
		{
			double h = dataset.getH(list[i]);
			if(h > hmax)
				hmax = h;
		}
		return hmax;
	}

	inline double computeGlobalMaxH(const std::vector<int> &list)
	{
		double hmax = computeMaxH(list);

	   	MPI_Allreduce(MPI_IN_PLACE, &hmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	   	return hmax;
	}

	inline void distributeParticles(const std::vector<int> &list, std::vector<std::vector<int>> &cellList, std::vector<int> &globalCellCount)
	{
		cellList.resize(ncells);
		globalCellCount.resize(ncells);

		std::vector<int> localCount(ncells);

		for(unsigned int i=0; i<list.size(); i++)
		{
			double xx = std::max(std::min(dataset.getX(i),bbox.xmax),bbox.xmin);
			double yy = std::max(std::min(dataset.getY(i),bbox.ymax),bbox.ymin);
			double zz = std::max(std::min(dataset.getZ(i),bbox.zmax),bbox.zmin);

			double posx = normalize(xx, bbox.xmin, bbox.xmax);
			double posy = normalize(yy, bbox.ymin, bbox.ymax);
			double posz = normalize(zz, bbox.zmin, bbox.zmax);

			int hx = posx*nX;
			int hy = posy*nY;
			int hz = posz*nZ;

			hx = std::min(hx,nX-1);
			hy = std::min(hy,nY-1);
			hz = std::min(hz,nZ-1);

			unsigned int l = hz*nX*nY+hy*nX+hx;

			cellList[l].push_back(list[i]);
			localCount[l] = cellList[l].size();
		}

		MPI_Allreduce(&localCount[0], &globalCellCount[0], ncells, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	}

	inline void assignRanks(const std::vector<int> &globalCellCount, std::vector<int> &assignedRanks, std::vector<int> &rankLoad)
	{
		assignedRanks.resize(ncells);

		int rank = 0;
		int work = 0;
		int procSize = dataset.getProcSize();

		// Assign rank to each cell
		for(int i=0; i<ncells; i++)
		{
			work += globalCellCount[i];
			assignedRanks[i] = rank;

			if(work >= procSize)
			{
			 	rankLoad.push_back(work);
			 	work = 0;
			 	rank++;
			}
		}

		if(work > 0)
			rankLoad.push_back(work);
	}

	inline void computeBBoxes(std::vector<BBox> &cellBBox)
	{
		for(int hz=0; hz<nZ; hz++)
		{
			for(int hy=0; hy<nY; hy++)
			{
				for(int hx=0; hx<nX; hx++)
				{
					double ax = bbox.xmin + hx*(bbox.xmax-bbox.xmin)/nX;
					double bx = bbox.xmin + (hx+1)*(bbox.xmax-bbox.xmin)/nX;
					double ay = bbox.ymin + hy*(bbox.ymax-bbox.ymin)/nY;
					double by = bbox.ymin + (hy+1)*(bbox.ymax-bbox.ymin)/nY;
					double az = bbox.zmin + hz*(bbox.zmax-bbox.zmin)/nZ;
					double bz = bbox.zmin + (hz+1)*(bbox.zmax-bbox.zmin)/nZ;

					unsigned int i = hz*nX*nY+hy*nX+hx;

					cellBBox[i] = BBox(ax, bx, ay, by, az, bz);
				}
			}
		}
	}

	inline void computeDiscardList(const std::vector<std::vector<int>> &cellList, const std::vector<int> &assignedRanks, std::vector<int> &discardList)
	{
		for(int i=0; i<ncells; i++)
		{
			if(assignedRanks[i] != comm_rank)
			{
				for(unsigned j=0; j<cellList[i].size(); j++)
					discardList.push_back(cellList[i][j]);
			}
		}
	}

	inline void exchangeParticles(int count, const std::vector<std::vector<int>> &cellList, const std::vector<int> &globalCellCount, const std::vector<int> &assignedRanks)
	{
		for(int i=0; i<ncells; i++)
		{
			if(assignedRanks[i] != comm_rank && cellList[i].size() > 0)
				dataset.send(cellList[i], assignedRanks[i], i*100);
			else if(assignedRanks[i] == comm_rank)
			{
				int needed = globalCellCount[i] - cellList[i].size();
				if(needed > 0)
				{
					dataset.recv(count, needed, MPI_ANY_SOURCE, i*100);
					count += needed;
				}
			}

			//MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	inline void tagGhostCells(const std::vector<int> &assignedRanks, const std::vector<BBox> &cellBBox, const std::vector<int> &globalCellCount, std::vector<int> &localWanted, std::vector<int> &globalWanted)
	{
		for(int i=0; i<ncells; i++)
			globalWanted[i] = 0;

		for(int i=0; i<ncells; i++)
		{
			if(assignedRanks[i] != comm_rank)
			{
				// maxH is an overrapproximation
				if(	overlap(cellBBox[i].xmin, cellBBox[i].xmax, dataset.bbox.xmin-maxH, dataset.bbox.xmax+maxH) &&
					overlap(cellBBox[i].ymin, cellBBox[i].ymax, dataset.bbox.ymin-maxH, dataset.bbox.ymax+maxH) &&
					overlap(cellBBox[i].zmin, cellBBox[i].zmax, dataset.bbox.zmin-maxH, dataset.bbox.zmax+maxH))
				{
					if(globalCellCount[i] > 0)
					{
						globalWanted[i] = 1;
						localWanted[i] = 1;
					}
				}
			}
		}

		MPI_Allreduce(&localWanted[0], &globalWanted[0], ncells, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	}

	inline void exchangeGhosts(const std::vector<int> &assignedRanks, const std::vector<std::vector<int>> &cellList, const std::vector<int> &globalCellCount, const std::vector<int> &localWanted, const std::vector<int> &globalWanted)
	{
		for(int i=0; i<ncells; i++)
		{
			if(assignedRanks[i] == comm_rank && globalWanted[i] > 0)
			{
				std::vector<int> ranks(globalWanted[i]);
				for(int j=0; j<globalWanted[i]; j++)
				{
					MPI_Status status;
					int rank;
					MPI_Recv(&rank, 1, MPI_INT, MPI_ANY_SOURCE, i*100, MPI_COMM_WORLD, &status);
					dataset.send(cellList[i], rank, i*100);
				}
			}
			else if(assignedRanks[i] != comm_rank && localWanted[i] > 0)
			{
				MPI_Send(&comm_rank, 1, MPI_INT, assignedRanks[i], i*100, MPI_COMM_WORLD);
				int count = dataset.getCount();
				dataset.resize(count+globalCellCount[i]);
				dataset.recv(count, globalCellCount[i], assignedRanks[i], i*100);
			}

			//MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	inline bool overlap(double leftA, double rightA, double leftB, double rightB)
	{
		return leftA < rightB && rightA > leftB;
	}

 	void build(const BBox &newBbox, std::vector<int> &computeList)
	{	
	    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	    MPI_Get_processor_name(processor_name, &name_len);

	    bbox = newBbox;

	   	maxH = computeGlobalMaxH(computeList);

	   	//if(comm_rank == 0) printf("(%d/%d,%s) Global max H = %f\n", comm_rank, comm_size, processor_name, maxH);
		
		nX = std::max((bbox.xmax-bbox.xmin) / maxH, 2.0);
		nY = std::max((bbox.ymax-bbox.ymin) / maxH, 2.0);
		nZ = std::max((bbox.zmax-bbox.zmin) / maxH, 2.0);
		ncells = nX*nY*nZ;

		if(comm_rank == 0) printf("(%d/%d,%s) Distributing particles...\n", comm_rank, comm_size, processor_name);

		std::vector<int> globalCellCount;
		std::vector<std::vector<int>> cellList;
		distributeParticles(computeList, cellList, globalCellCount);

		if(comm_rank == 0) printf("(%d/%d,%s) Assigning work load...\n", comm_rank, comm_size, processor_name);

		std::vector<int> rankLoad;
		std::vector<int> assignedRanks;
		assignRanks(globalCellCount, assignedRanks, rankLoad);

		// if(comm_rank == 0) 
		// {
		// 	int totalLoad = 0;
		// 	for(int i=0; i<comm_size; i++)
		// 	{
		// 		totalLoad += rankLoad[i];
		// 		printf("\t(%d) has load: %d\n", i, rankLoad[i]);
		// 	}
		// 	printf("\tTotal load: %d\n", totalLoad);
		// }

		//if(comm_rank == 0) printf("(%d/%d,%s) Computing discard list...\n", comm_rank, comm_size, processor_name);

		std::vector<int> discardList;
		computeDiscardList(cellList, assignedRanks, discardList);

		int count = dataset.getCount();
		dataset.resize(rankLoad[comm_rank]+discardList.size());

		//printf("\t(%d) Has: %zu Missing: %zu\n", comm_rank, rankLoad[comm_rank]-discardList.size(), discardList.size());

		if(comm_rank == 0) printf("(%d/%d,%s) Exchanging particles...\n", comm_rank, comm_size, processor_name);

		exchangeParticles(count, cellList, globalCellCount, assignedRanks);

		MPI_Barrier(MPI_COMM_WORLD);

		// Discard extra
		//printf("\t(%d) Count: %d, Discarding %zu and resizing to %d\n", comm_rank, dataset.getCount(), discardList.size(), rankLoad[comm_rank]);

		dataset.discard(discardList);
		dataset.resize(rankLoad[comm_rank]);

		//if(comm_rank == 0) printf("(%d/%d,%s) Computing cells bboxes...\n", comm_rank, comm_size, processor_name);

		// Compute cell boxes using globalBBox
		std::vector<BBox> cellBBox(ncells);
		computeBBoxes(cellBBox);

		computeList.resize(dataset.getCount());
		for(unsigned int i=0; i<computeList.size(); i++)
			computeList[i] = i;

		if(comm_rank == 0) printf("(%d/%d,%s) Redistributing particles...\n", comm_rank, comm_size, processor_name);

		cellList.clear();
		globalCellCount.clear();
		distributeParticles(computeList, cellList, globalCellCount);

		// Only now we are allowed to recompute the dataset BBox
		dataset.computeBoundingBox();

		//if(comm_rank == 0) printf("(%d/%d,%s) Finding ghost cells...\n", comm_rank, comm_size, processor_name);

		std::vector<int> localWanted(ncells), globalWanted(ncells);
		tagGhostCells(assignedRanks, cellBBox, globalCellCount, localWanted, globalWanted);

		if(comm_rank == 0) printf("(%d/%d,%s) Exchanging ghost cells...\n", comm_rank, comm_size, processor_name);

		exchangeGhosts(assignedRanks, cellList, globalCellCount, localWanted, globalWanted);

		MPI_Barrier(MPI_COMM_WORLD);

		// dataset.computeBoundingBox();

		// if(comm_rank == 3)
		// {
		// 	printf("(%d/%d,%s) Domain x[%f %f]\n", comm_rank, comm_size, processor_name, dataset.bbox.xmin, dataset.bbox.xmax);
		// 	printf("(%d/%d,%s) Domain y[%f %f]\n", comm_rank, comm_size, processor_name, dataset.bbox.ymin, dataset.bbox.ymax);
		// 	printf("(%d/%d,%s) Domain z[%f %f]\n", comm_rank, comm_size, processor_name, dataset.bbox.zmin, dataset.bbox.zmax);
		// }
	}
};

template<class DatasetAdaptor>
class HTree
{
public:
	DatasetAdaptor &dataset;

	BBox bbox;

	Cell<DatasetAdaptor> root;

	HTree() = delete;
	
	~HTree() = default;

	HTree(DatasetAdaptor &dataset) : 
		dataset(dataset), root(dataset) {}

	void build(std::vector<int> &computeList /*const HTreeParams params = HTreeParams()*/)
 	{
		bbox = dataset.getBBox();

		MPI_Allreduce(MPI_IN_PLACE, &bbox.xmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &bbox.ymin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &bbox.zmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &bbox.xmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &bbox.ymax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &bbox.zmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		root.build(bbox, computeList);
 	}
};

}

