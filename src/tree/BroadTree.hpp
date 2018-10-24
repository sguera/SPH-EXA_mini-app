#ifndef SPHEXA_BROADTREE_HPP
#define SPHEXA_BROADTREE_HPP

#include <vector>
#include <cmath>
#include <mutex>
#include <stdio.h>
#include <unistd.h>

namespace sphexa
{
constexpr unsigned int const MAXP = 8;
constexpr double const RATIO = 0.5;
constexpr int const TREE = 1;
constexpr int const BLOCK_SIZE = 32;
constexpr int const PLANCK = 1e-15;

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

	void build(const int n, const double *x, const double *y, const double *z, int **ordering = 0);
	
	void findNeighbors(const double xi, const double yi, const double zi, const double ri, const int ngmax, int *ng, int &nvi, 
		const bool PBCx = false, const bool PBCy = false, const bool PBCz = false) const;
 
private:
	double _minx, _maxx;
	double _miny, _maxy;
	double _minz, _maxz;

	BroadTree **_p;
	int B, C;

	int *_start;
	int *_count;

	double *_x, *_y, *_z;
	int *_ordering;

	void cleanRec();

	void buildSortRec(const std::vector<int> &list, const double *x, const double *y, const double *z, int it);
	
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
	cleanRec();
	if(_x) delete[] _x;
	if(_y) delete[] _y;
	if(_z) delete[] _z;
	if(_ordering) delete[] _ordering;
	_x = _y = _z = 0;
	_ordering = 0;
}

void BroadTree::cleanRec()
{
	_x = _y = _z = 0;
	_ordering = 0;
	if(_p)
	{
		for(int i=0; i<C*C*C; i++)
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
	for(int i=0; i<C*C*C; i++)
		if(_p[i]) cells += _p[i]->cellCount();
	return cells;
}

int BroadTree::bucketCount() const
{
	int cells = C*C*C;
	for(int i=0; i<C*C*C; i++)
		if(_p[i]) cells += _p[i]->bucketCount();
	return cells;
}

void BroadTree::build(const int n, const double *x, const double *y, const double *z, int **ordering)
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
	buildSortRec(list, x, y, z, 0);
}

void BroadTree::buildSortRec(const std::vector<int> &list, const double *x, const double *y, const double *z, int it)
{
	//#pragma omp task
	{
		if(TREE)
			C = std::max((int)pow(list.size()/(float)MAXP,1.0/3.0), 2);
		 	//C = max((int)(log(list.size())/log(DIVIDE)), 2);
		else
		 	C = 2;

		int *padding = 0;
		std::mutex *mtx = 0;
		std::vector<int> *tmp = 0;

		double ratio = 1.0;
		while(1)
		{
			if(_p == 0)
			{
				_p = new BroadTree*[C*C*C];
				for(int i=0; i<C*C*C; i++)
					_p[i] = 0;
			}

			if(_start == 0)
			{
				_start = new int[C*C*C];
				_count = new int[C*C*C];
				for(int i=0; i<C*C*C; i++)
				{
					_start[i] = 0;
					_count[i] = 0;
				}
			}

			padding = new int[C*C*C];
			mtx = new std::mutex[C*C*C];
			tmp = new std::vector<int>[C*C*C];

			//#pragma omp parallel for schedule(static)
			for(unsigned int i=0; i<list.size(); i++)
			{
				double xx = std::max(std::min(x[list[i]],_maxx),_minx);
				double yy = std::max(std::min(y[list[i]],_maxy),_miny);
				double zz = std::max(std::min(z[list[i]],_maxz),_minz);

				double posx = normalize(xx, _minx, _maxx);
				double posy = normalize(yy, _miny, _maxy);
				double posz = normalize(zz, _minz, _maxz);

				int hx = posx*C;
				int hy = posy*C;
				int hz = posz*C;

				hx = std::min(hx,C-1);
				hy = std::min(hy,C-1);
				hz = std::min(hz,C-1);

				unsigned int l = hz*C*C+hy*C+hx;

				mtx[l].lock();
				tmp[l].push_back(list[i]);
				mtx[l].unlock();
			}

			int full = 0;
			int empty = 0;

			int ppc = std::max((int)(list.size()/(C*C*C)), 1);
			int b3 = BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE;
			B = std::max((int)pow(b3/ppc, 1.0/3.0), 1);

			int pad = 0;
			for(int bi=0; bi<C; bi += B)
	        {
	            for(int bj=0; bj<C; bj += B)
	            {
	                for(int bk=0; bk<C; bk += B)
	                {
	                	int hzmax = std::min(bi+B,C);
	                    for(int hz=bi; hz<hzmax; ++hz)
	                    {
	                        int hymax = std::min(bj+B,C);
	                        for(int hy=bj; hy<hymax; ++hy)
	                        { 
	                            int hxmax = std::min(bk+B,C);
	                            for(int hx=bk; hx<hxmax; ++hx)
	                            {
									unsigned int l = hz*C*C+hy*C+hx;
									padding[l] = pad;
									pad += tmp[l].size();

									if(tmp[l].size() > 4*MAXP) full++;
									else if(tmp[l].size() < MAXP/4) empty++;
								}
							}
						}
					}
				}
			}

			ratio = empty/(float)(C*C*C);
			if(ratio > RATIO)
			{
				if(C/2 > 2)
				{
					cleanRec();
					delete[] padding;
					delete[] mtx;
					delete[] tmp;
					C = C/2;
					continue;
				}
			}

			//#pragma omp parallel for collapse(3) schedule(static)
			for(int hz=0; hz<C; hz++)
			{
				for(int hy=0; hy<C; hy++)
				{
					for(int hx=0; hx<C; hx++)
					{
						double ax = _minx + hx*(_maxx-_minx)/C;
						double bx = _minx + (hx+1)*(_maxx-_minx)/C;
						double ay = _miny + hy*(_maxy-_miny)/C;
						double by = _miny + (hy+1)*(_maxy-_miny)/C;
						double az = _minz + hz*(_maxz-_minz)/C;
						double bz = _minz + (hz+1)*(_maxz-_minz)/C;

						unsigned int l = hz*C*C+hy*C+hx;

						if(tmp[l].size() > 4*MAXP && bx-ax > PLANCK && by-ay > PLANCK && bz-az > PLANCK)
						{
							_p[l] = new BroadTree(_x, _y, _z, _ordering);
							_p[l]->setBox(ax, bx, ay, by, az, bz);
							_p[l]->buildSortRec(tmp[l], x, y, z, it+padding[l]);
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

			break;
		}

		delete[] padding;
		delete[] mtx;
		delete[] tmp;
	}
}

void BroadTree::findNeighbors(const double xi, const double yi, const double zi, const double ri, const int ngmax, int *ng, int &nvi,
	const bool PBCx, const bool PBCy, const bool PBCz) const
{
	if((PBCx && (xi-ri < _minx || xi+ri > _maxx)) || (PBCy && (yi-ri < _miny || yi+ri > _maxy)) || (PBCz && (zi-ri < _minz || zi+ri > _maxz)))
	{
		int mix = (int)floor(normalize(xi-ri, _minx, _maxx)*C) % C;
		int miy = (int)floor(normalize(yi-ri, _miny, _maxy)*C) % C;
		int miz = (int)floor(normalize(zi-ri, _minz, _maxz)*C) % C;
		int max = (int)floor(normalize(xi+ri, _minx, _maxx)*C) % C;
		int may = (int)floor(normalize(yi+ri, _miny, _maxy)*C) % C;
		int maz = (int)floor(normalize(zi+ri, _minz, _maxz)*C) % C;

		for(int hz=miz; hz<=maz; hz++)
		{
			for(int hy=miy; hy<=may; hy++)
			{
				for(int hx=mix; hx<=max; hx++)
				{
					double displz = PBCz? ((hz < 0) - (hz >= C)) * (_maxz-_minz) : 0;
		 			double disply = PBCy? ((hy < 0) - (hy >= C)) * (_maxy-_miny) : 0;
		 			double displx = PBCx? ((hx < 0) - (hx >= C)) * (_maxx-_minx) : 0;

					int hzz = (hz + C) % C;
		 			int hyy = (hy + C) % C;
					int hxx = (hx + C) % C;

					unsigned int l = hzz*C*C+hyy*C+hxx;

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
	int mix = std::max((int)(normalize(xi-ri, _minx, _maxx)*C),0);
	int miy = std::max((int)(normalize(yi-ri, _miny, _maxy)*C),0);
	int miz = std::max((int)(normalize(zi-ri, _minz, _maxz)*C),0);
	int max = std::min((int)(normalize(xi+ri, _minx, _maxx)*C),C-1);
	int may = std::min((int)(normalize(yi+ri, _miny, _maxy)*C),C-1);
	int maz = std::min((int)(normalize(zi+ri, _minz, _maxz)*C),C-1);

	for(int hz=miz; hz<=maz; hz++)
	{
		for(int hy=miy; hy<=may; hy++)
		{
			for(int hx=mix; hx<=max; hx++)
			{
				unsigned int l = hz*C*C+hy*C+hx;

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
