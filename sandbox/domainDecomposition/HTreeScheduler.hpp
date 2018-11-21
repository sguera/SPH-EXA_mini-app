#pragma once

#include <vector>
#include <cmath>
#include <map>

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

	std::vector<int> assignedRanks;

	BBox localBBox, globalBBox;

	std::map<int,std::vector<int>> sendGhostList;
	std::map<int,int> recvGhostList;
	int ghostCount;

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

	inline bool overlap(double leftA, double rightA, double leftB, double rightB)
	{
		return leftA < rightB && rightA > leftB;
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

	inline void assignRanks(const std::vector<int> &globalCellCount, std::vector<int> &rankLoad)
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

	inline void computeDiscardList(const std::vector<std::vector<int>> &cellList, std::vector<int> &discardList)
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

	inline void exchangeParticles(const std::vector<std::vector<int>> &cellList, const std::vector<int> &globalCellCount)
	{
		std::vector<int> toSend[comm_size];
		std::vector<std::vector<double>> xbuff, ybuff, zbuff, hbuff;

		int needed = 0;

		for(int i=0; i<ncells; i++)
		{
			int rank = assignedRanks[i];
			if(rank != comm_rank && cellList[i].size() > 0)
				toSend[rank].insert(toSend[rank].end(), cellList[i].begin(), cellList[i].end());
			else if(rank == comm_rank)
				needed += globalCellCount[i] - cellList[i].size();
		}

		std::vector<MPI_Request> requests;
		for(int rank=0; rank<comm_size; rank++)
		{
			if(toSend[rank].size() > 0)
			{
				int rcount = requests.size();
				int bcount = xbuff.size();
				int count = toSend[rank].size();

				requests.resize(rcount+6);
				xbuff.resize(bcount+1);
				ybuff.resize(bcount+1);
				zbuff.resize(bcount+1);
				hbuff.resize(bcount+1);

				xbuff[bcount].resize(count);
				ybuff[bcount].resize(count);
				zbuff[bcount].resize(count);
				hbuff[bcount].resize(count);

				for(int j=0; j<count; j++)
				{
					xbuff[bcount][j] = dataset.x[toSend[rank][j]];
					ybuff[bcount][j] = dataset.y[toSend[rank][j]];
					zbuff[bcount][j] = dataset.z[toSend[rank][j]];
					hbuff[bcount][j] = dataset.h[toSend[rank][j]];
				}

				MPI_Isend(&comm_rank, 1, MPI_INT, rank, 0, comm, &requests[rcount]);
				MPI_Isend(&count, 1, MPI_INT, rank, 1, comm, &requests[rcount+1]);
				MPI_Isend(&xbuff[bcount][0], count, MPI_DOUBLE, rank, 2, comm, &requests[rcount+2]);
				MPI_Isend(&ybuff[bcount][0], count, MPI_DOUBLE, rank, 3, comm, &requests[rcount+3]);
				MPI_Isend(&zbuff[bcount][0], count, MPI_DOUBLE, rank, 4, comm, &requests[rcount+4]);
				MPI_Isend(&hbuff[bcount][0], count, MPI_DOUBLE, rank, 5, comm, &requests[rcount+5]);
			}
		}

		int end = dataset.getCount();
		dataset.resize(end+needed);

		while(needed > 0)
		{
			MPI_Status status[6];

			int rank, count;
			MPI_Recv(&rank, 1, MPI_INT, MPI_ANY_SOURCE, 0, comm, &status[0]);
			MPI_Recv(&count, 1, MPI_INT, rank, 1, comm, &status[1]);
			MPI_Recv(&dataset.x[end], count, MPI_DOUBLE, rank, 2, comm, &status[2]);
			MPI_Recv(&dataset.y[end], count, MPI_DOUBLE, rank, 3, comm, &status[3]);
			MPI_Recv(&dataset.z[end], count, MPI_DOUBLE, rank, 4, comm, &status[4]);
			MPI_Recv(&dataset.h[end], count, MPI_DOUBLE, rank, 5, comm, &status[5]);

			end += count;
			needed -= count;
		}

		MPI_Status status[requests.size()];
		MPI_Waitall(requests.size(), &requests[0], status);
	}

	inline void computeGhostList(const std::vector<BBox> &cellBBox, const std::vector<std::vector<int>> &cellList, const BBox &localBBox, const std::vector<int> &globalCellCount)
	{
		std::vector<int> localWanted(ncells), globalWanted(ncells);

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
						localWanted[i] = 1;
				}
			}
		}

		MPI_Allreduce(&localWanted[0], &globalWanted[0], ncells, MPI_INT, MPI_SUM, comm);

		ghostCount = 0;
		std::vector<MPI_Request> requests;
		for(int i=0; i<ncells; i++)
		{
			if(assignedRanks[i] != comm_rank && localWanted[i] > 0)
			{
				int rcount = requests.size();
				requests.resize(rcount+1);
				ghostCount += globalCellCount[i];
				recvGhostList[assignedRanks[i]] += globalCellCount[i];
				MPI_Isend(&comm_rank, 1, MPI_INT, assignedRanks[i], i, comm, &requests[rcount]);
			}
		}

		for(int i=0; i<ncells; i++)
		{
			if(assignedRanks[i] == comm_rank && globalWanted[i] > 0)
			{
				std::vector<int> ranks(globalWanted[i]);
				for(int j=0; j<globalWanted[i]; j++)
				{
					int rank;
					MPI_Status status;
					MPI_Recv(&rank, 1, MPI_INT, MPI_ANY_SOURCE, i, comm, &status);
					sendGhostList[rank].insert(sendGhostList[rank].end(), cellList[i].begin(), cellList[i].end());
				}
			}
		}

		MPI_Status status[requests.size()];
		MPI_Waitall(requests.size(), &requests[0], status);
	}

	inline void exchangeGhosts()
	{
		std::vector<std::vector<double>> xbuff, ybuff, zbuff, hbuff;
		std::vector<MPI_Request> requests;

		for(auto it=sendGhostList.begin(); it!=sendGhostList.end(); ++it)
		{
			int rank = it->first;
			const std::vector<int> &cellList = it->second;

			int rcount = requests.size();
			int bcount = xbuff.size();
			int count = cellList.size();

			requests.resize(rcount+4);
			xbuff.resize(bcount+1);
			ybuff.resize(bcount+1);
			zbuff.resize(bcount+1);
			hbuff.resize(bcount+1);

			xbuff[bcount].resize(count);
			ybuff[bcount].resize(count);
			zbuff[bcount].resize(count);
			hbuff[bcount].resize(count);

			for(int j=0; j<count; j++)
			{
				xbuff[bcount][j] = dataset.x[cellList[j]];
				ybuff[bcount][j] = dataset.y[cellList[j]];
				zbuff[bcount][j] = dataset.z[cellList[j]];
				hbuff[bcount][j] = dataset.h[cellList[j]];
			}

			MPI_Isend(&xbuff[bcount][0], count, MPI_DOUBLE, rank, 0, comm, &requests[rcount]);
			MPI_Isend(&ybuff[bcount][0], count, MPI_DOUBLE, rank, 1, comm, &requests[rcount+1]);
			MPI_Isend(&zbuff[bcount][0], count, MPI_DOUBLE, rank, 2, comm, &requests[rcount+2]);
			MPI_Isend(&hbuff[bcount][0], count, MPI_DOUBLE, rank, 3, comm, &requests[rcount+3]);
		}

		int end = dataset.getCount();
		dataset.resize(end+ghostCount);
		for(auto it=recvGhostList.begin(); it!=recvGhostList.end(); ++it)
		{
			int rank = it->first;
			int count = it->second;
			int rcount = requests.size();

			requests.resize(rcount+4);

			MPI_Irecv(&dataset.x[end], count, MPI_DOUBLE, rank, 0, comm, &requests[rcount]);
			MPI_Irecv(&dataset.y[end], count, MPI_DOUBLE, rank, 1, comm, &requests[rcount+1]);
			MPI_Irecv(&dataset.z[end], count, MPI_DOUBLE, rank, 2, comm, &requests[rcount+2]);
			MPI_Irecv(&dataset.h[end], count, MPI_DOUBLE, rank, 3, comm, &requests[rcount+3]);

			end += count;
		}

		MPI_Status status[requests.size()];
		MPI_Waitall(requests.size(), &requests[0], status);
	}

 	void balance(std::vector<int> &computeList)
	{	
	    MPI_Comm_size(comm, &comm_size);
	    MPI_Comm_rank(comm, &comm_rank);
	    MPI_Get_processor_name(processor_name, &name_len);

	    globalBBox = computeGlobalBoundingBox(computeList);
	   	globalMaxH = computeGlobalMaxH(computeList);

		nX = std::max((globalBBox.xmax-globalBBox.xmin) / globalMaxH, 2.0);
		nY = std::max((globalBBox.ymax-globalBBox.ymin) / globalMaxH, 2.0);
		nZ = std::max((globalBBox.zmax-globalBBox.zmin) / globalMaxH, 2.0);
		ncells = nX*nY*nZ;

		std::vector<int> globalCellCount;
		std::vector<std::vector<int>> cellList;
		distributeParticles(computeList, globalBBox, cellList, globalCellCount);

		std::vector<int> rankLoad;
		assignRanks(globalCellCount, rankLoad);

		exchangeParticles(cellList, globalCellCount);

		std::vector<int> discardList;
		computeDiscardList(cellList, discardList);
		dataset.discard(discardList);

		computeList.resize(dataset.getCount());
		for(unsigned int i=0; i<computeList.size(); i++)
			computeList[i] = i;
	}

	void findGhosts(const std::vector<int> &computeList)
	{
		std::vector<int> globalCellCount;
		std::vector<std::vector<int>> cellList;
		distributeParticles(computeList, globalBBox, cellList, globalCellCount);

		// Only now we are allowed to recompute the dataset BBox
		localBBox = computeBoundingBox(computeList);
		localMaxH = computeMaxH(computeList);

		// Compute cell boxes using globalBBox
		std::vector<BBox> cellBBox(ncells);
		computeBBoxes(globalBBox, cellBBox);

		// Use cell boxes and the localbbox to identify ghost cells
		computeGhostList(cellBBox, cellList, localBBox, globalCellCount);
	}

	void updateGhosts()
	{
		exchangeGhosts();
	}
};

}

