#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "../BBox.hpp"

namespace sphexa
{
template<typename T = double, class ArrayT = std::vector<T>>
class Octree
{
public:
	int cellCount() const
	{
		int c = 1;
		for(int i=0; i<ncells; i++)
			if(cells[i] != nullptr) c += cells[i]->cellCount();
		return c;
	}

	int bucketCount() const
	{
		int c = ncells;
		for(int i=0; i<ncells; i++)
			if(cells[i] != nullptr) c += cells[i]->bucketCount();
		return c;
	}

	inline T normalize(T d, T min, T max) const
	{
		return (d-min)/(max-min);
	}

	inline T distancesq(const T x1, const T y1, const T z1, const T x2, const T y2, const T z2) const
	{
		T xx = x1 - x2;
		T yy = y1 - y2;
		T zz = z1 - z2;

		return xx*xx + yy*yy + zz*zz;
	}

	inline void check_add_start(const int start, const int count, const T xi, const T yi, const T zi, const T r, const unsigned int ngmax, std::vector<int> &neighbors) const
	{
		T dists[count];
		for(int i=0; i<count; i++)
		{
			int id = start+i;
			T xx = (*x)[id];
			T yy = (*y)[id];
			T zz = (*z)[id];

			dists[i] = distancesq(xi, yi, zi, xx, yy, zz);
		}

		for(int i=0; i<count; i++)
		{
			if(neighbors.size() < ngmax && dists[i] < r*r)//distancesq(xi, yi, zi, xx, yy, zz) < r2)
				neighbors.push_back((*ordering)[start+i]);
		}
	}

	// inline T computeMaxH()
	// {
	// 	T hmax = 0.0;
	// 	for(unsigned int i=0; i<ax.size(); i++)
	// 	{
	// 		T h = ah.getH(i);
	// 		if(h > hmax)
	// 			hmax = h;
	// 	}
	// 	return hmax;
	// }

	inline void distributeParticles(const std::vector<int> &list, const ArrayT &ax, const ArrayT &ay, const ArrayT &az, std::vector<std::vector<int>> &cellList)
	{
		for(unsigned int i=0; i<list.size(); i++)
		{
			T xx = ax[list[i]];
			T yy = ay[list[i]];
			T zz = az[list[i]];

			T hx = std::min(std::max((int)(normalize(xx, bbox.xmin, bbox.xmax)*nX),0),nX-1);
			T hy = std::min(std::max((int)(normalize(yy, bbox.ymin, bbox.ymax)*nY),0),nY-1);
			T hz = std::min(std::max((int)(normalize(zz, bbox.zmin, bbox.zmax)*nZ),0),nZ-1);

			unsigned int l = hz*nX*nY+hy*nX+hx;

			cellList[l].push_back(list[i]);
		}
	}

	inline void computeBBoxes(std::vector<BBox<T>> &cellBBox)
	{
		for(int hz=0; hz<nZ; hz++)
		{
			for(int hy=0; hy<nY; hy++)
			{
				for(int hx=0; hx<nX; hx++)
				{
					T ax = bbox.xmin + hx*(bbox.xmax-bbox.xmin)/nX;
					T bx = bbox.xmin + (hx+1)*(bbox.xmax-bbox.xmin)/nX;
					T ay = bbox.ymin + hy*(bbox.ymax-bbox.ymin)/nY;
					T by = bbox.ymin + (hy+1)*(bbox.ymax-bbox.ymin)/nY;
					T az = bbox.zmin + hz*(bbox.zmax-bbox.zmin)/nZ;
					T bz = bbox.zmin + (hz+1)*(bbox.zmax-bbox.zmin)/nZ;

					unsigned int i = hz*nX*nY+hy*nX+hx;

					cellBBox[i] = BBox<T>(ax, bx, ay, by, az, bz);
				}
			}
		}
	}

	inline void computePadding(const std::vector<std::vector<int>> &cellList, std::vector<int> &padding)
	{
		int pad = 0;
		for(int hz=0; hz<nZ; hz++)
		{
			for(int hy=0; hy<nY; hy++)
			{
				for(int hx=0; hx<nX; hx++)
				{
					unsigned int l = hz*nX*nY+hy*nX+hx;
					padding[l] = pad;
					pad += cellList[l].size();
				}
			}
		}
	}

 	void buildRec(const std::vector<int> &list, const BBox<T> &bbox, const ArrayT &ax, const ArrayT &ay, const ArrayT &az, const ArrayT &ah, const unsigned int bucketSize, int ptr)
	{	
		this->bbox = bbox;
	   	//maxH = computeMaxH();

		nX = 2;//std::max((bbox.xmax-bbox.xmin) / maxH, 2.0);
		nY = 2;//std::max((bbox.ymax-bbox.ymin) / maxH, 2.0);
		nZ = 2;//std::max((bbox.zmax-bbox.zmin) / maxH, 2.0);
		ncells = nX*nY*nZ;

		std::vector<std::vector<int>> cellList(ncells);
		distributeParticles(list, ax, ay, az, cellList);

		std::vector<BBox<T>> cellBBox(ncells);
		computeBBoxes(cellBBox);

		std::vector<int> padding(ncells);
		computePadding(cellList, padding);

		cells.resize(ncells);
		start.resize(ncells);
		count.resize(ncells);

		for(int i=0; i<ncells; i++)
		{
			start[i] = ptr+padding[i];
			count[i] = cellList[i].size();

			if(cellList[i].size() > bucketSize)// && bx-ax > PLANCK && by-ay > PLANCK && bz-az > PLANCK)
			{
				cells[i] = std::make_shared<Octree>();
				cells[i]->x = x;
				cells[i]->y = y;
				cells[i]->z = z;
				cells[i]->ordering = ordering;
				cells[i]->buildRec(cellList[i], cellBBox[i], ax, ay, az, ah, bucketSize, ptr+padding[i]);
			}
			else
			{
				start[i] = ptr+padding[i];
				count[i] = cellList[i].size();

				for(int j=0; j<count[i]; j++)
				{
					int id = cellList[i][j];
					(*ordering)[ptr+padding[i]+j] = id;
					(*x)[ptr+padding[i]+j] = ax[id];
					(*y)[ptr+padding[i]+j] = ay[id];
					(*z)[ptr+padding[i]+j] = az[id];
				}
			}
		}
	}

	void build(const BBox<T> &bbox, const ArrayT &ax, const ArrayT &ay, const ArrayT &az, const ArrayT &ah, const unsigned int bucketSize = 128)
	{
		int count = ax.size();

		x = std::make_shared<std::vector<T>>(count);
		y = std::make_shared<std::vector<T>>(count);
		z = std::make_shared<std::vector<T>>(count);

		ordering = std::make_shared<std::vector<int>>(count);

		std::vector<int> list(count);

		for(int i=0; i<count; i++)
			list[i] = i;

		buildRec(list, bbox, ax, ay, az, ah, bucketSize, 0);
	}

	void findNeighborsRec(const T xi, const T yi, const T zi, const T ri, const int ngmax, std::vector<int> &neighbors) const
	{
		int mix = std::max((int)(normalize(xi-ri, bbox.xmin, bbox.xmax)*nX),0);
		int miy = std::max((int)(normalize(yi-ri, bbox.ymin, bbox.ymax)*nY),0);
		int miz = std::max((int)(normalize(zi-ri, bbox.zmin, bbox.zmax)*nZ),0);
		int max = std::min((int)(normalize(xi+ri, bbox.xmin, bbox.xmax)*nX),nX-1);
		int may = std::min((int)(normalize(yi+ri, bbox.ymin, bbox.ymax)*nY),nY-1);
		int maz = std::min((int)(normalize(zi+ri, bbox.zmin, bbox.zmax)*nZ),nZ-1);

		for(int hz=miz; hz<=maz; hz++)
		{
			for(int hy=miy; hy<=may; hy++)
			{
				for(int hx=mix; hx<=max; hx++)
				{
					unsigned int l = hz*nX*nY+hy*nX+hx;
					
					if(cells[l] != nullptr)
		 				cells[l]->findNeighborsRec(xi, yi, zi, ri, ngmax, neighbors);
		 			else
		 				check_add_start(start[l], count[l], xi, yi, zi, ri, ngmax, neighbors);
				}
			}
		}
	}

	void findNeighbors(const T xi, const T yi, const T zi, const T ri, const int ngmax, std::vector<int> &neighbors,
		const bool PBCx = false, const bool PBCy = false, const bool PBCz = false) const
	{
		if((PBCx && (xi-ri < bbox.xmin || xi+ri > bbox.xmax)) || (PBCy && (yi-ri < bbox.ymin || yi+ri > bbox.ymax)) || (PBCz && (zi-ri < bbox.zmin || zi+ri > bbox.zmax)))
		{
			int mix = (int)floor(normalize(xi-ri, bbox.xmin, bbox.xmax)*nX);
			int miy = (int)floor(normalize(yi-ri, bbox.ymin, bbox.ymax)*nY);
			int miz = (int)floor(normalize(zi-ri, bbox.zmin, bbox.zmax)*nZ);
			int max = (int)floor(normalize(xi+ri, bbox.xmin, bbox.xmax)*nX);
			int may = (int)floor(normalize(yi+ri, bbox.ymin, bbox.ymax)*nY);
			int maz = (int)floor(normalize(zi+ri, bbox.zmin, bbox.zmax)*nZ);

			for(int hz=miz; hz<=maz; hz++)
			{
				for(int hy=miy; hy<=may; hy++)
				{
					for(int hx=mix; hx<=max; hx++)
					{
						T displz = PBCz? ((hz < 0) - (hz >= nZ)) * (bbox.zmax-bbox.zmin) : 0;
			 			T disply = PBCy? ((hy < 0) - (hy >= nY)) * (bbox.ymax-bbox.ymin) : 0;
			 			T displx = PBCx? ((hx < 0) - (hx >= nX)) * (bbox.xmax-bbox.xmin) : 0;

						int hzz = (hz + nZ) % nZ;
			 			int hyy = (hy + nY) % nY;
						int hxx = (hx + nX) % nX;

						unsigned int l = hzz*nY*nX+hyy*nX+hxx;

						if(cells[l] != nullptr)
			 				cells[l]->findNeighborsRec(xi+displx, yi+disply, zi+displz, ri, ngmax, neighbors);
			 			else
			 				check_add_start(start[l], count[l], xi+displx, yi+disply, zi+displz, ri, ngmax, neighbors);
			 		}
				}
			}
		}
		else
			findNeighborsRec(xi, yi, zi, ri, ngmax, neighbors);
	}

public:
	std::shared_ptr<std::vector<int>> ordering;

private:
	int ncells;
	int nX, nY, nZ;

	BBox<T> bbox;

	std::vector<std::shared_ptr<Octree>> cells;

	std::vector<int> start, count;
	std::shared_ptr<std::vector<T>> x, y, z;
};

}

