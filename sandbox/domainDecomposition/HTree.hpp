#pragma once

#include <cmath>
#include <memory>

#include "BBox.hpp"
// TODO
// Make assign rank recursive to assign load more finely:
// If a cell cannot be assigned, call build recursively (else stop)
// Make discardList, exchange, and others recursive as well (global cell id needed)
// Finish building

namespace sphexa
{

struct HTreeParams
{
	HTreeParams(unsigned int bucketSize = 128) :
		bucketSize(bucketSize) {}
	unsigned int bucketSize;
};

template<class DatasetAdaptor>
class HTree
{
public:
	DatasetAdaptor &dataset;

	BBox bbox;

	double maxH;

	int ncells;
	int nX, nY, nZ;

	std::vector<std::shared_ptr<HTree>> cells;

	std::vector<int> start, count;
	std::shared_ptr<std::vector<int>> ordering;
	std::shared_ptr<std::vector<double>> x, y, z;

	HTree() = delete;
	
	~HTree() = default;

	HTree(DatasetAdaptor &dataset) : 
		dataset(dataset) {}

	int cellCount() const
	{
		int c = 1;//  C*C*C;
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

	inline double normalize(double d, double min, double max) const
	{
		return (d-min)/(max-min);
	}

	inline double distancesq(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2) const
	{
		double xx = x1 - x2;
		double yy = y1 - y2;
		double zz = z1 - z2;

		return xx*xx + yy*yy + zz*zz;
	}

	inline void check_add_start(const int start, const int count, const double xi, const double yi, const double zi, const double r, const int ngmax, int *ng, int &nvi) const
	{
		double dists[count];
		for(int i=0; i<count; i++)
		{
			int id = start+i;
			double xx = (*x)[id];
			double yy = (*y)[id];
			double zz = (*z)[id];

			dists[i] = distancesq(xi, yi, zi, xx, yy, zz);
		}

		for(int i=0; i<count; i++)
		{
			if(nvi < ngmax && dists[i] < r*r)//distancesq(xi, yi, zi, xx, yy, zz) < r2)
				ng[nvi++] = (*ordering)[i];
		}

	}

	inline double computeMaxH()
	{
		double hmax = 0.0;
		for(unsigned int i=0; i<dataset.getCount(); i++)
		{
			double h = dataset.getH(i);
			if(h > hmax)
				hmax = h;
		}
		return hmax;
	}

	inline void distributeParticles(const std::vector<int> &list, std::vector<std::vector<int>> &cellList)
	{
		for(unsigned int i=0; i<list.size(); i++)
		{
			double xx = std::max(std::min(dataset.getX(list[i]),bbox.xmax),bbox.xmin);
			double yy = std::max(std::min(dataset.getY(list[i]),bbox.ymax),bbox.ymin);
			double zz = std::max(std::min(dataset.getZ(list[i]),bbox.zmax),bbox.zmin);

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
		}
	}

	inline void computeBBoxes(const BBox &bbox, std::vector<BBox> &cellBBox)
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

 	void buildRec(const std::vector<int> &list, const BBox &localBBox, const HTreeParams &params, int ptr)
	{	
		bbox = localBBox;
	   	maxH = computeMaxH();

		nX = std::max((bbox.xmax-bbox.xmin) / maxH, 2.0);
		nY = std::max((bbox.ymax-bbox.ymin) / maxH, 2.0);
		nZ = std::max((bbox.zmax-bbox.zmin) / maxH, 2.0);
		ncells = nX*nY*nZ;

		std::vector<std::vector<int>> cellList(ncells);
		distributeParticles(list, cellList);

		std::vector<BBox> cellBBox(ncells);
		computeBBoxes(bbox, cellBBox);

		std::vector<int> padding(ncells);
		computePadding(cellList, padding);

		cells.resize(ncells);
		start.resize(ncells);
		count.resize(ncells);
		for(int i=0; i<ncells; i++)
		{
			if(cellList[i].size() > params.bucketSize)// && bx-ax > PLANCK && by-ay > PLANCK && bz-az > PLANCK)
			{
				cells[i] = std::make_shared<HTree>(dataset);
				cells[i]->x = x;
				cells[i]->y = y;
				cells[i]->z = z;
				cells[i]->ordering = ordering;
				cells[i]->buildRec(cellList[i], cellBBox[i], params, ptr+padding[i]);
			}
			else
			{
				start[i] = ptr+padding[i];
				count[i] = cellList[i].size();

				for(int j=0; j<count[i]; j++)
				{
					int id = cellList[i][j];
					(*ordering)[ptr+padding[i]+j] = id;
					(*x)[ptr+padding[i]+j] = dataset.getX(id);
					(*y)[ptr+padding[i]+j] = dataset.getY(id);
					(*z)[ptr+padding[i]+j] = dataset.getZ(id);
				}
			}
		}
	}

	void build(const BBox &localBBox, const HTreeParams &params = HTreeParams())
	{
		int count = dataset.getCount();

		x = std::make_shared<std::vector<double>>(count);
		y = std::make_shared<std::vector<double>>(count);
		z = std::make_shared<std::vector<double>>(count);

		ordering = std::make_shared<std::vector<int>>(count);

		std::vector<int> list(count);

		for(int i=0; i<count; i++)
			list[i] = i;

		buildRec(list, localBBox, params, 0);
	}

	void findNeighborsRec(const double xi, const double yi, const double zi, const double ri, const int ngmax, int *ng, int &nvi) const
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
		 				cells[l]->findNeighborsRec(xi, yi, zi, ri, ngmax, ng, nvi);
		 			else
		 				check_add_start(start[l], count[l], xi, yi, zi, ri, ngmax, ng, nvi);
				}
			}
		}
	}

	void findNeighbors(const double xi, const double yi, const double zi, const double ri, const int ngmax, int *ng, int &nvi,
		const bool PBCx = false, const bool PBCy = false, const bool PBCz = false) const
	{
		if((PBCx && (xi-ri < bbox.xmin || xi+ri > bbox.xmax)) || (PBCy && (yi-ri < bbox.ymin || yi+ri > bbox.ymax)) || (PBCz && (zi-ri < bbox.zmin || zi+ri > bbox.zmax)))
		{
			int mix = (int)floor(normalize(xi-ri, bbox.xmin, bbox.xmax)*nX) % nX;
			int miy = (int)floor(normalize(yi-ri, bbox.ymin, bbox.ymax)*nY) % nY;
			int miz = (int)floor(normalize(zi-ri, bbox.zmin, bbox.zmax)*nZ) % nZ;
			int max = (int)floor(normalize(xi+ri, bbox.xmin, bbox.xmax)*nX) % nX;
			int may = (int)floor(normalize(yi+ri, bbox.ymin, bbox.ymax)*nY) % nY;
			int maz = (int)floor(normalize(zi+ri, bbox.zmin, bbox.zmax)*nZ) % nZ;

			for(int hz=miz; hz<=maz; hz++)
			{
				for(int hy=miy; hy<=may; hy++)
				{
					for(int hx=mix; hx<=max; hx++)
					{
						double displz = PBCz? ((hz < 0) - (hz >= nZ)) * (bbox.zmax-bbox.zmin) : 0;
			 			double disply = PBCy? ((hy < 0) - (hy >= nY)) * (bbox.ymax-bbox.ymin) : 0;
			 			double displx = PBCx? ((hx < 0) - (hx >= nX)) * (bbox.xmax-bbox.xmin) : 0;

						int hzz = (hz + nZ) % nZ;
			 			int hyy = (hy + nY) % nY;
						int hxx = (hx + nX) % nX;

						unsigned int l = hzz*nY*nX+hyy*nX+hxx;

						if(cells[l] != nullptr)
			 				cells[l]->findNeighborsRec(xi+displx, yi+disply, zi+displz, ri, ngmax, ng, nvi);
			 			else
			 				check_add_start(start[l], count[l], xi+displx, yi+disply, zi+displz, ri, ngmax, ng, nvi);
			 		}
				}
			}
		}
		else
			findNeighborsRec(xi, yi, zi, ri, ngmax, ng, nvi);
	}
};

}

