#pragma once

#include <cmath>

#include "BBox.hpp"

// TODO
// Make assign rank recursive to assign load more finely:
// If a cell cannot be assigned, call build recursively (else stop)
// Make discardList, exchange, and others recursive as well (global cell id needed)
// Finish building

namespace sphexa
{

template<class DatasetAdaptor>
class HTreeScheduler
{
public:
	MPI_Comm comm;

	DatasetAdaptor &dataset;

	BBox localBBox, globalBBox;

	double localMaxH, globalMaxH;

	int ncells;
	int nX, nY, nZ;

	int comm_size, comm_rank, name_len;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	HTreeScheduler() = delete;
	
	~HTreeScheduler() = default;

	HTreeScheduler(const MPI_Comm &comm, DatasetAdaptor &dataset) : 
		comm(comm), dataset(dataset) {}

	inline double normalize(double d, double min, double max)
	{
		return (d-min)/(max-min);
	}

	inline BBox computeBoundingBox(const std::vector<int> &computeList)
	{
		double xmin = INFINITY;
		double xmax = -INFINITY;
		double ymin = INFINITY;
		double ymax = -INFINITY;
		double zmin = INFINITY;
		double zmax = -INFINITY;

		for(unsigned int i=0; i<computeList.size(); i++)
		{
			double x = dataset.getX(computeList[i]);
			double y = dataset.getY(computeList[i]);
			double z = dataset.getZ(computeList[i]);

			if(x < xmin) xmin = x;
			if(x > xmax) xmax = x;
			if(y < ymin) ymin = y;
			if(y > ymax) ymax = y;
			if(z < zmin) zmin = z;
			if(z > zmax) zmax = z;
		}

		return BBox(xmin, xmax, ymin, ymax, zmin, zmax);
	}

	inline BBox computeGlobalBoundingBox(const std::vector<int> &computeList)
	{
		BBox bbox = computeBoundingBox(computeList);

		MPI_Allreduce(MPI_IN_PLACE, &bbox.xmin, 1, MPI_DOUBLE, MPI_MIN, comm);
		MPI_Allreduce(MPI_IN_PLACE, &bbox.ymin, 1, MPI_DOUBLE, MPI_MIN, comm);
		MPI_Allreduce(MPI_IN_PLACE, &bbox.zmin, 1, MPI_DOUBLE, MPI_MIN, comm);
		MPI_Allreduce(MPI_IN_PLACE, &bbox.xmax, 1, MPI_DOUBLE, MPI_MAX, comm);
		MPI_Allreduce(MPI_IN_PLACE, &bbox.ymax, 1, MPI_DOUBLE, MPI_MAX, comm);
		MPI_Allreduce(MPI_IN_PLACE, &bbox.zmax, 1, MPI_DOUBLE, MPI_MAX, comm);

		return bbox;
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

	   	MPI_Allreduce(MPI_IN_PLACE, &hmax, 1, MPI_DOUBLE, MPI_MAX, comm);

	   	return hmax;
	}

	inline void distributeParticles(const std::vector<int> &list, const BBox &globalBBox, std::vector<std::vector<int>> &cellList, std::vector<int> &globalCellCount)
	{
		cellList.resize(ncells);
		globalCellCount.resize(ncells);

		std::vector<int> localCount(ncells);

		for(unsigned int i=0; i<list.size(); i++)
		{
			double xx = std::max(std::min(dataset.getX(list[i]),globalBBox.xmax),globalBBox.xmin);
			double yy = std::max(std::min(dataset.getY(list[i]),globalBBox.ymax),globalBBox.ymin);
			double zz = std::max(std::min(dataset.getZ(list[i]),globalBBox.zmax),globalBBox.zmin);

			double posx = normalize(xx, globalBBox.xmin, globalBBox.xmax);
			double posy = normalize(yy, globalBBox.ymin, globalBBox.ymax);
			double posz = normalize(zz, globalBBox.zmin, globalBBox.zmax);

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

		MPI_Allreduce(&localCount[0], &globalCellCount[0], ncells, MPI_INT, MPI_SUM, comm);
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

	inline void computeBBoxes(const BBox &globalBBox, std::vector<BBox> &cellBBox)
	{
		for(int hz=0; hz<nZ; hz++)
		{
			for(int hy=0; hy<nY; hy++)
			{
				for(int hx=0; hx<nX; hx++)
				{
					double ax = globalBBox.xmin + hx*(globalBBox.xmax-globalBBox.xmin)/nX;
					double bx = globalBBox.xmin + (hx+1)*(globalBBox.xmax-globalBBox.xmin)/nX;
					double ay = globalBBox.ymin + hy*(globalBBox.ymax-globalBBox.ymin)/nY;
					double by = globalBBox.ymin + (hy+1)*(globalBBox.ymax-globalBBox.ymin)/nY;
					double az = globalBBox.zmin + hz*(globalBBox.zmax-globalBBox.zmin)/nZ;
					double bz = globalBBox.zmin + (hz+1)*(globalBBox.zmax-globalBBox.zmin)/nZ;

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

	inline void exchangeParticles(int end, const std::vector<std::vector<int>> &cellList, const std::vector<int> &globalCellCount, const std::vector<int> &assignedRanks)
	{
		for(int i=0; i<ncells; i++)
		{
			if(assignedRanks[i] != comm_rank && cellList[i].size() > 0)
			{
				int count = cellList[i].size();
				MPI_Send(&comm_rank, 1, MPI_INT, assignedRanks[i], i*100, comm);
				MPI_Send(&count, 1, MPI_INT, assignedRanks[i], i*100, comm);
				dataset.send(cellList[i], assignedRanks[i], i*100);
			}
			else if(assignedRanks[i] == comm_rank)
			{
				int needed = globalCellCount[i] - cellList[i].size();
				while(needed > 0)
				{
					MPI_Status status;
					int rank, count;
					MPI_Recv(&rank, 1, MPI_INT, MPI_ANY_SOURCE, i*100, comm, &status);
					MPI_Recv(&count, 1, MPI_INT, rank, i*100, comm, &status);
					dataset.recv(end, count, rank, i*100);
					end += count;
					needed -= count;
				}
			}

			//MPI_Barrier(comm);
		}
	}

	inline void tagGhostCells(const std::vector<int> &assignedRanks, const std::vector<BBox> &cellBBox, const BBox &localBBox, const std::vector<int> &globalCellCount, std::vector<int> &localWanted, std::vector<int> &globalWanted)
	{
		for(int i=0; i<ncells; i++)
			globalWanted[i] = 0;

		for(int i=0; i<ncells; i++)
		{
			if(assignedRanks[i] != comm_rank)
			{
				// maxH is an overrapproximation
				if(	overlap(cellBBox[i].xmin, cellBBox[i].xmax, localBBox.xmin-2*localMaxH, localBBox.xmax+2*localMaxH) &&
					overlap(cellBBox[i].ymin, cellBBox[i].ymax, localBBox.ymin-2*localMaxH, localBBox.ymax+2*localMaxH) &&
					overlap(cellBBox[i].zmin, cellBBox[i].zmax, localBBox.zmin-2*localMaxH, localBBox.zmax+2*localMaxH))
				{
					if(globalCellCount[i] > 0)
					{
						globalWanted[i] = 1;
						localWanted[i] = 1;
					}
				}
			}
		}

		MPI_Allreduce(&localWanted[0], &globalWanted[0], ncells, MPI_INT, MPI_SUM, comm);
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
					MPI_Recv(&rank, 1, MPI_INT, MPI_ANY_SOURCE, i*100, comm, &status);
					dataset.send(cellList[i], rank, i*100);
				}
			}
			else if(assignedRanks[i] != comm_rank && localWanted[i] > 0)
			{
				MPI_Send(&comm_rank, 1, MPI_INT, assignedRanks[i], i*100, comm);
				int count = dataset.getCount();
				dataset.resize(count+globalCellCount[i]);
				dataset.recv(count, globalCellCount[i], assignedRanks[i], i*100);
			}

			//MPI_Barrier(comm);
		}
	}

	inline bool overlap(double leftA, double rightA, double leftB, double rightB)
	{
		return leftA < rightB && rightA > leftB;
	}

 	void balance(std::vector<int> &computeList)
	{	
	    MPI_Comm_size(comm, &comm_size);
	    MPI_Comm_rank(comm, &comm_rank);
	    MPI_Get_processor_name(processor_name, &name_len);

	    globalBBox = computeGlobalBoundingBox(computeList);
	   	globalMaxH = computeGlobalMaxH(computeList);

	   	//if(comm_rank == 0) printf("(%d/%d,%s) Global max H = %f\n", comm_rank, comm_size, processor_name, maxH);
		
		nX = std::max((globalBBox.xmax-globalBBox.xmin) / globalMaxH, 2.0);
		nY = std::max((globalBBox.ymax-globalBBox.ymin) / globalMaxH, 2.0);
		nZ = std::max((globalBBox.zmax-globalBBox.zmin) / globalMaxH, 2.0);
		ncells = nX*nY*nZ;

		//if(comm_rank == 0) printf("(%d/%d,%s) Distributing particles...\n", comm_rank, comm_size, processor_name);

		std::vector<int> globalCellCount;
		std::vector<std::vector<int>> cellList;
		distributeParticles(computeList, globalBBox, cellList, globalCellCount);

		//if(comm_rank == 0) printf("(%d/%d,%s) Assigning work load...\n", comm_rank, comm_size, processor_name);

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

		//if(comm_rank == 0) printf("(%d/%d,%s) Exchanging particles...\n", comm_rank, comm_size, processor_name);

		exchangeParticles(count, cellList, globalCellCount, assignedRanks);
		
		// Discard extra
		//printf("\t(%d) Count: %d, Discarding %zu and resizing to %d\n", comm_rank, dataset.getCount(), discardList.size(), rankLoad[comm_rank]);

		dataset.discard(discardList);
		dataset.resize(rankLoad[comm_rank]);

		//if(comm_rank == 0) printf("(%d/%d,%s) Computing cells bboxes...\n", comm_rank, comm_size, processor_name);

		// Compute cell boxes using globalBBox
		std::vector<BBox> cellBBox(ncells);
		computeBBoxes(globalBBox, cellBBox);

		computeList.resize(dataset.getCount());
		for(unsigned int i=0; i<computeList.size(); i++)
			computeList[i] = i;

		//if(comm_rank == 0) printf("(%d/%d,%s) Redistributing particles...\n", comm_rank, comm_size, processor_name);

		cellList.clear();
		globalCellCount.clear();
		distributeParticles(computeList, globalBBox, cellList, globalCellCount);

		// Only now we are allowed to recompute the dataset BBox
		localBBox = computeBoundingBox(computeList);
		localMaxH = computeMaxH(computeList);

		std::vector<int> localWanted(ncells), globalWanted(ncells);
		tagGhostCells(assignedRanks, cellBBox, localBBox, globalCellCount, localWanted, globalWanted);

		//if(comm_rank == 0) printf("(%d/%d,%s) Exchanging ghost cells...\n", comm_rank, comm_size, processor_name);

		exchangeGhosts(assignedRanks, cellList, globalCellCount, localWanted, globalWanted);
	}
};

}

