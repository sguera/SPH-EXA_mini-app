#ifndef SPHEXA_BROADTREE_HPP
#define SPHEXA_BROADTREE_HPP

#include <vector>
#include <cmath>
#include <mutex>
#include <stdio.h>
#include <unistd.h>

#include <iostream>
using namespace std;

namespace sphexa
{
constexpr unsigned int const BUCKETSIZE = 128;
//constexpr double const RATIO = 0.5;
//constexpr int const TREE = 1;
constexpr int const BLOCK_SIZE = 32;
constexpr double const PLANCK = 1e-15;

class BroadTree
{
public:
	BroadTree();
	BroadTree(double *x, double *y, double *z, int *ordering);

	~BroadTree();

	void clean();

	int cellCount() const;
	int bucketCount() const;

	void setBox(const double minx, const double maxx, const double miny, const double maxy, const double minz, const double maxz);

	void build(const int n, const double *x, const double *y, const double *z, const double *h, int **ordering = 0);
	
	void findNeighbors(const double xi, const double yi, const double zi, const double ri, const int ngmax, int *ng, int &nvi, 
		const bool PBCx = false, const bool PBCy = false, const bool PBCz = false) const;
 
private:
	double _minx, _maxx;
	double _miny, _maxy;
	double _minz, _maxz;

	BroadTree **_p;
	int B;//, C;
	int ncells, nX, nY, nZ;

	int *_start;
	int *_count;

	double *_x, *_y, *_z;
	int *_ordering;

	void cleanRec(bool zero = true);

	void buildSortRec(const std::vector<int> &list, const double *x, const double *y, const double *z, const double *h, int it);
	
	void findNeighborsRec(const double xi, const double yi, const double zi, const double ri, const int ngmax, int *ng, int &nvi) const;

	static inline double normalize(double d, double min, double max);

	static inline double distancesq(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2);

	static inline void check_add_start(const int start, const int count, const int *ordering, const double *x, const double *y, const double *z, const double xi, const double yi, const double zi, const double r, const int ngmax, int *ng, int &nvi);
};

inline double BroadTree::normalize(double d, double min, double max)
{
	return (d-min)/(max-min);
}

inline double BroadTree::distancesq(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2)
{
	double xx = x1 - x2;
	double yy = y1 - y2;
	double zz = z1 - z2;

	return xx*xx + yy*yy + zz*zz;
}

inline void BroadTree::check_add_start(const int start, const int count, const int *ordering, const double *x, const double *y, const double *z, const double xi, const double yi, const double zi, const double r, const int ngmax, int *ng, int &nvi)
{
	double dists[count];
	for(int i=0; i<count; i++)
	{
		int id = start+i;
		double xx = x[id];
		double yy = y[id];
		double zz = z[id];

		dists[i] = distancesq(xi, yi, zi, xx, yy, zz);
	}

	for(int i=0; i<count; i++)
	{
		if(nvi < ngmax && dists[i] < r*r)//distancesq(xi, yi, zi, xx, yy, zz) < r2)
			ng[nvi++] = ordering[i];
	}
}

BroadTree::BroadTree()
{
	_p = 0;
	_start = 0;
	_count = 0;
	_x = 0;
	_y = 0;
	_z = 0;
	_ordering = 0;
	setBox(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

BroadTree::BroadTree(double *x, double *y, double *z, int *ordering)
{
	_p = 0;
	_start = 0;
	_count = 0;
	_x = x;
	_y = y;
	_z = z;
	_ordering = ordering;
	setBox(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

BroadTree::~BroadTree()
{
	clean();
}

void BroadTree::clean()
{
	if(_x) delete[] _x;
	if(_y) delete[] _y;
	if(_z) delete[] _z;
	if(_ordering) delete[] _ordering;
	_x = _y = _z = 0;
	_ordering = 0;
	cleanRec();
}

void BroadTree::cleanRec(bool zero)
{
	if(zero)
	{
		_x = _y = _z = 0;
		_ordering = 0;
	}
	if(_p)
	{
		for(int i=0; i<ncells; i++)
		{
			if(_p[i])
				_p[i]->cleanRec();
			delete _p[i];
			_p[i] = 0;
		}
		delete[] _p;
		_p = 0;
	}
	if(_start)
	{
		delete[] _start;
		delete[] _count;
		_start = 0;
		_count = 0;
	}
}

void BroadTree::setBox(const double minx, const double maxx, const double miny, const double maxy, const double minz, const double maxz)
{
	_minx = minx;
	_maxx = maxx;
	_miny = miny;
	_maxy = maxy;
	_minz = minz;
	_maxz = maxz;
}

int BroadTree::cellCount() const
{
	int cells = 1;//  C*C*C;
	for(int i=0; i<ncells; i++)
		if(_p[i]) cells += _p[i]->cellCount();
	return cells;
}

int BroadTree::bucketCount() const
{
	int cells = ncells;
	for(int i=0; i<ncells; i++)
		if(_p[i]) cells += _p[i]->bucketCount();
	return cells;
}

void BroadTree::build(const int n, const double *x, const double *y, const double *z, const double *h, int **ordering)
{
	clean();

	std::vector<int> list(n);

	_x = new double[n];
	_y = new double[n];
	_z = new double[n];

	_ordering = new int[n];

	if(ordering)
		*ordering = _ordering;

	//#pragma omp parallel for
	for(int i=0; i<n; i++)
		list[i] = i;

	//#pragma omp parallel
	//#pragma omp single
	buildSortRec(list, x, y, z, h, 0);
}

inline double findMaxH(const std::vector<int> &list, const double *h)
{
	double hmax = 0.0;
	for(unsigned int i=0; i<list.size(); i++)
	{
		if(h[list[i]] > hmax)
			hmax = h[list[i]];
	}
	return hmax;
}

void BroadTree::buildSortRec(const std::vector<int> &list, const double *x, const double *y, const double *z, const double *h, int it)
{
	// Find maximum h
	double hmax = findMaxH(list, h);

	// Find how many cuts are needed in each dimension of the box
	nX = std::max((_maxx-_minx) / hmax, 2.0);
	nY = std::max((_maxy-_miny) / hmax, 2.0);
	nZ = std::max((_maxz-_minz) / hmax, 2.0);

	// Find total number of cells
	ncells = nX*nY*nZ;

	if(_p == 0)
	{
		_p = new BroadTree*[ncells];
		for(int i=0; i<ncells; i++)
			_p[i] = 0;
	}

	if(_start == 0)
	{
		_start = new int[ncells];
		_count = new int[ncells];
		for(int i=0; i<ncells; i++)
		{
			_start[i] = 0;
			_count[i] = 0;
		}
	}

	int *padding = new int[ncells];
	//std::mutex *mtx = new std::mutex[ncells];
	std::vector<int> *tmp = new std::vector<int>[ncells];

	//#pragma omp parallel for schedule(static)
	for(unsigned int i=0; i<list.size(); i++)
	{
		double xx = std::max(std::min(x[list[i]],_maxx),_minx);
		double yy = std::max(std::min(y[list[i]],_maxy),_miny);
		double zz = std::max(std::min(z[list[i]],_maxz),_minz);

		double posx = normalize(xx, _minx, _maxx);
		double posy = normalize(yy, _miny, _maxy);
		double posz = normalize(zz, _minz, _maxz);

		int hx = posx*nX;
		int hy = posy*nY;
		int hz = posz*nZ;

		hx = std::min(hx,nX-1);
		hy = std::min(hy,nY-1);
		hz = std::min(hz,nZ-1);

		unsigned int l = hz*nX*nY+hy*nX+hx;

		//mtx[l].lock();
		#ifdef _DEBUG
        std::cout << "l=" << l 
                  << " hz=" << hz
                  << " nX=" << nX
                  << " nY=" << nY
                  << " hy=" << hy
                  << " nX=" << nX
                  << " hx=" << hx
                  << " minx=" << _minx
                  << " maxx=" << _maxx
                  << std::endl;
        #endif
		tmp[l].push_back(list[i]);
		//mtx[l].unlock();
	}

	// BLOCK PARTITIONING (SIMILAR TO MM-multiplication)
	int mppc = std::max((int)(list.size()/(ncells)), 1);
	int b3 = BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE;
	B = std::max((int)pow(b3/mppc, 1.0/3.0), 1);

	int pad = 0;
	for(int bi=0; bi<nZ; bi += B)
    {
        for(int bj=0; bj<nY; bj += B)
        {
            for(int bk=0; bk<nX; bk += B)
            {
            	int hzmax = std::min(bi+B,nZ);
                for(int hz=bi; hz<hzmax; ++hz)
                {
                    int hymax = std::min(bj+B,nY);
                    for(int hy=bj; hy<hymax; ++hy)
                    { 
                        int hxmax = std::min(bk+B,nX);
                        for(int hx=bk; hx<hxmax; ++hx)
                        {
							unsigned int l = hz*nX*nY+hy*nX+hx;
							padding[l] = pad;
							pad += tmp[l].size();
						}
					}
				}
			}
		}
	}

	//#pragma omp parallel for collapse(3) schedule(static)
	for(int hz=0; hz<nZ; hz++)
	{
		for(int hy=0; hy<nY; hy++)
		{
			for(int hx=0; hx<nX; hx++)
			{
				double ax = _minx + hx*(_maxx-_minx)/nX;
				double bx = _minx + (hx+1)*(_maxx-_minx)/nX;
				double ay = _miny + hy*(_maxy-_miny)/nY;
				double by = _miny + (hy+1)*(_maxy-_miny)/nY;
				double az = _minz + hz*(_maxz-_minz)/nZ;
				double bz = _minz + (hz+1)*(_maxz-_minz)/nZ;

				unsigned int l = hz*nX*nY+hy*nX+hx;

				// 
				if(tmp[l].size() > BUCKETSIZE && bx-ax > PLANCK && by-ay > PLANCK && bz-az > PLANCK)
				{
					_p[l] = new BroadTree(_x, _y, _z, _ordering);
					_p[l]->setBox(ax, bx, ay, by, az, bz);
					_p[l]->buildSortRec(tmp[l], x, y, z, h, it+padding[l]);
				}
				else
				{	
					_start[l] = it+padding[l];
					_count[l] = tmp[l].size();

					for(int i=0; i<_count[l]; i++)
					{
						int id = tmp[l][i];
						*(_ordering+it+padding[l]+i) = id;
						*(_x+it+padding[l]+i) = x[id];
						*(_y+it+padding[l]+i) = y[id];
						*(_z+it+padding[l]+i) = z[id];
					}
				}
			}
		}
	}

	delete[] padding;
	//delete[] mtx;
	delete[] tmp;
}

void BroadTree::findNeighbors(const double xi, const double yi, const double zi, const double ri, const int ngmax, int *ng, int &nvi,
	const bool PBCx, const bool PBCy, const bool PBCz) const
{
	if((PBCx && (xi-ri < _minx || xi+ri > _maxx)) || (PBCy && (yi-ri < _miny || yi+ri > _maxy)) || (PBCz && (zi-ri < _minz || zi+ri > _maxz)))
	{
		int mix = (int)floor(normalize(xi-ri, _minx, _maxx)*nX) % nX;
		int miy = (int)floor(normalize(yi-ri, _miny, _maxy)*nY) % nY;
		int miz = (int)floor(normalize(zi-ri, _minz, _maxz)*nZ) % nZ;
		int max = (int)floor(normalize(xi+ri, _minx, _maxx)*nX) % nX;
		int may = (int)floor(normalize(yi+ri, _miny, _maxy)*nY) % nY;
		int maz = (int)floor(normalize(zi+ri, _minz, _maxz)*nZ) % nZ;

		for(int hz=miz; hz<=maz; hz++)
		{
			for(int hy=miy; hy<=may; hy++)
			{
				for(int hx=mix; hx<=max; hx++)
				{
					double displz = PBCz? ((hz < 0) - (hz >= nZ)) * (_maxz-_minz) : 0;
		 			double disply = PBCy? ((hy < 0) - (hy >= nY)) * (_maxy-_miny) : 0;
		 			double displx = PBCx? ((hx < 0) - (hx >= nX)) * (_maxx-_minx) : 0;

					int hzz = (hz + nZ) % nZ;
		 			int hyy = (hy + nY) % nY;
					int hxx = (hx + nX) % nX;

					unsigned int l = hzz*nY*nX+hyy*nX+hxx;

					if(_p[l])
		 				_p[l]->findNeighborsRec(xi+displx, yi+disply, zi+displz, ri, ngmax, ng, nvi);
		 			else
		 				check_add_start(_start[l], _count[l], _ordering, _x, _y, _z, xi+displx, yi+disply, zi+displz, ri, ngmax, ng, nvi);
		 		}
			}
		}
	}
	else
		findNeighborsRec(xi, yi, zi, ri, ngmax, ng, nvi);
}

void BroadTree::findNeighborsRec(const double xi, const double yi, const double zi, const double ri, const int ngmax, int *ng, int &nvi) const
{
	int mix = std::max((int)(normalize(xi-ri, _minx, _maxx)*nX),0);
	int miy = std::max((int)(normalize(yi-ri, _miny, _maxy)*nY),0);
	int miz = std::max((int)(normalize(zi-ri, _minz, _maxz)*nZ),0);
	int max = std::min((int)(normalize(xi+ri, _minx, _maxx)*nX),nX-1);
	int may = std::min((int)(normalize(yi+ri, _miny, _maxy)*nY),nY-1);
	int maz = std::min((int)(normalize(zi+ri, _minz, _maxz)*nZ),nZ-1);

	for(int hz=miz; hz<=maz; hz++)
	{
		for(int hy=miy; hy<=may; hy++)
		{
			for(int hx=mix; hx<=max; hx++)
			{
				unsigned int l = hz*nX*nY+hy*nX+hx;

				if(_p[l])
					_p[l]->findNeighborsRec(xi, yi, zi, ri, ngmax, ng, nvi);
				else
					check_add_start(_start[l], _count[l], _ordering, _x, _y, _z, xi, yi, zi, ri, ngmax, ng, nvi);
			}
		}
	}
}

}

#endif // SPHEXA_TREE_HPP
