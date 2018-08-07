#include <cmath>
#include <stack>
#include <stdio.h>
#include <unistd.h>
#include "Tree.hpp"

using namespace std;

inline double normalize(double d, double min, double max)
{
	return (d-min)/(max-min);
}

inline double distance(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2)
{
	double xx = x1 - x2;
	double yy = y1 - y2;
	double zz = z1 - z2;

	return sqrt(xx*xx + yy*yy + zz*zz);
}

inline void check_add_list(const vector<int> &list, const double *x, const double *y, const double *z, const double xi, const double yi, const double zi, const double r, const int ngmax, int *ng, int &nvi)
{
	for(unsigned int i=0; i<list.size(); i++)
	{
		int id = list[i];
		double xx = x[id];
		double yy = y[id];
		double zz = z[id];

		if(nvi < ngmax && distance(xi, yi, zi, xx, yy, zz) < r)
			ng[nvi++] = id;
	}
}

Tree::Tree()
{
	_p = 0;
	_list = 0;
	init(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

Tree::~Tree()
{
	clean();
}

void Tree::clean()
{
	if(_p)
	{
		for(int i=0; i<C*C*C; i++)
		{
			if(_p[i])
				_p[i]->clean();
			delete _p[i];
			_p[i] = 0;
		}
		delete[] _p;
		_p = 0;
	}
	if(_list)
	{
		delete[] _list;
		_list = 0;
	}
}

void Tree::init(const double minx, const double maxx, const double miny, const double maxy, const double minz, const double maxz)
{
	clean();
	_minx = minx;
	_maxx = maxx;
	_miny = miny;
	_maxy = maxy;
	_minz = minz;
	_maxz = maxz;
}

int Tree::cellCount()
{
	int cells = C*C*C;
	for(int i=0; i<C*C*C; i++)
		if(_p[i]) cells += _p[i]->cellCount();
	return cells;
}

void Tree::build(const int n, const double *x, const double *y, const double *z)
{
	vector<int> list(n);
	for(int i=0; i<n; i++)
		list[i] = i;
	clean();

	#pragma omp parallel
	{
		#pragma omp single
		{
			buildRec(list, x, y, z);
		}
	}
}

void Tree::buildRec(const vector<int> &list, const double *x, const double *y, const double *z)
{
	C = max((int)log(list.size()), 2);

	if(_list == 0)
		_list = new vector<int>[C*C*C];
	if(_p == 0)
	{
		_p = new Tree*[C*C*C];
		for(int i=0; i<C*C*C; i++)
			_p[i] = 0;
	}

	for(unsigned int i=0; i<list.size(); i++)
	{
		double xx = max(min(x[list[i]],_maxx),_minx);
		double yy = max(min(y[list[i]],_maxy),_miny);
		double zz = max(min(z[list[i]],_maxz),_minz);

		double posx = normalize(xx, _minx, _maxx);
		double posy = normalize(yy, _miny, _maxy);
		double posz = normalize(zz, _minz, _maxz);

		int hx = posx*C;
		int hy = posy*C;
		int hz = posz*C;

		hx = min(hx,C-1);
		hy = min(hy,C-1);
		hz = min(hz,C-1);

		_list[hz*C*C+hy*C+hx].push_back(list[i]);
	}

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
				if(_list[l].size() > MAXP && bx-ax > PLANCK && by-ay > PLANCK && bz-az > PLANCK)
				{
					#pragma omp task
					{
						_p[l] = new Tree();
						_p[l]->init(ax, bx, ay, by, az, bz);
						_p[l]->buildRec(_list[l], x, y, z);

						// Remove intermediate level lists to save space
						_list[l].clear();
						vector<int>().swap(_list[l]);
					}
				}
			}
		}
	}
}

void Tree::findNeighbors(const int i, const double *x, const double *y, const double *z, const double r, const int ngmax, int *ng, int &nvi,
	const bool PBCx, const bool PBCy, const bool PBCz)
{
	double xi = x[i];
	double yi = y[i];
	double zi = z[i];

	int mix = normalize(xi-r, _minx, _maxx)*C;
	int miy = normalize(yi-r, _miny, _maxy)*C;
	int miz = normalize(zi-r, _minz, _maxz)*C;
	int max = normalize(xi+r, _minx, _maxx)*C;
	int may = normalize(yi+r, _miny, _maxy)*C;
	int maz = normalize(zi+r, _minz, _maxz)*C;

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
	 				_p[l]->findNeighborsRec(x, y, z, xi+displx, yi+disply, zi+displz, r, ngmax, ng, nvi);
	 			else
	 				check_add_list(_list[l], x, y, z, xi+displx, yi+disply, zi+displz, r, ngmax, ng, nvi);
			}
		}
	}
}

void Tree::findNeighborsRec(const double *x, const double *y, const double *z, const double xi, const double yi, const double zi, const double r, const int ngmax, int *ng, int &nvi)
{
	int mix = max((int)(normalize(xi-r, _minx, _maxx)*C),0);
	int miy = max((int)(normalize(yi-r, _miny, _maxy)*C),0);
	int miz = max((int)(normalize(zi-r, _minz, _maxz)*C),0);
	int max = min((int)(normalize(xi+r, _minx, _maxx)*C),C-1);
	int may = min((int)(normalize(yi+r, _miny, _maxy)*C),C-1);
	int maz = min((int)(normalize(zi+r, _minz, _maxz)*C),C-1);

	for(int hz=miz; hz<=maz; hz++)
	{
		for(int hy=miy; hy<=may; hy++)
		{
			for(int hx=mix; hx<=max; hx++)
			{
				unsigned int l = hz*C*C+hy*C+hx;

				if(_p[l])
					_p[l]->findNeighborsRec(x, y, z, xi, yi, zi, r, ngmax, ng, nvi);
				else
					check_add_list(_list[l], x, y, z, xi, yi, zi, r, ngmax, ng, nvi);
			}
		}
	}
}
