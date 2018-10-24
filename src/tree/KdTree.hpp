#ifndef SPHEXA_KDTREE_HPP
#define SPHEXA_KDTREE_HPP

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

#define BUCKET 64
#define _GLIBCXX_PARALLEL

namespace sphexa
{

class KdTree
{
	inline double distancesq(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2) const
	{
		double xx = x1 - x2;
		double yy = y1 - y2;
		double zz = z1 - z2;

		return xx*xx + yy*yy + zz*zz;
	}

	inline double distbnd(double x, double amin, double amax) const
	{
		if (x > amax)
			return (x-amax); 
		else if(x < amin)
			return (amin-x);
		else
			return 0.0;
	}

	inline double sq(double a) const
	{
		return a*a;
	}
public:
	KdTree() : _left(NULL), _right(NULL) , _x(NULL), _y(NULL), _z(NULL) {}

	void clean()
	{
		cleanRec();
		if(_x) delete[] _x;
		if(_y) delete[] _y;
		if(_z) delete[] _z;
		if(_ordering) delete[] _ordering;
		_x = _y = _z = NULL;
		_left = _right = NULL;
	}

	void cleanRec()
	{
		if(_left)
		{
			_left->cleanRec();
			delete _left;
		}
		if(_right)
		{
			_right->cleanRec();
			delete _right;
		}
	}

	void setBox(const double minx, const double maxx, const double miny, const double maxy, const double minz, const double maxz)
	{
		_minx = minx;
		_maxx = maxx;
		_miny = miny;
		_maxy = maxy;
		_minz = minz;
		_maxz = maxz;
	}

	inline void computeBox()
	{
		// Set directly
		_minx = 1000;
		_maxx = -1000;
		_miny = 1000;
		_maxy = -1000;
		_minz = 1000;
		_maxz = -1000;
		for(int i=0; i<_count; i++)
		{
			if(_x[i] < _minx) _minx = _x[i];
			if(_x[i] > _maxx) _maxx = _x[i];
			if(_y[i] < _miny) _miny = _y[i];
			if(_y[i] > _maxy) _maxy = _y[i];
			if(_z[i] < _minz) _minz = _z[i];
			if(_z[i] > _maxz) _maxz = _z[i];
		}
	}

	inline void buildRec(const int n, double *x, double *y, double *z, int *ordering, int depth = 0)
	{
		_k = 3;
		_depth = depth;
		_count = n;
		_half = _count/2;

		_x = x;
		_y = y;
		_z = z;
		_ordering = ordering;

		computeBox();

		_axis = 0;
		double largestAx = _maxx-_minx;
		if(_maxy-_miny > largestAx)
		{
			largestAx = _maxy-_miny;
			_axis = 1;
		}
		if(_maxz-_minz > largestAx)
		{
			largestAx = _maxz-_minz;
			_axis = 2;
		}

		const double *axisList[_k] = {_x, _y, _z};
		const double *data = axisList[_axis];

		vector<int> idx(n);
		
		iota(idx.begin(), idx.end(), 0);

		std::vector<int>::iterator first = idx.begin();
		std::vector<int>::iterator last = idx.end();
		std::vector<int>::iterator middle = first+_half;
		std::nth_element(first, first+_half, last, [&](size_t a, size_t b) {return data[a] < data[b];}); // can specify comparator as optional 4th arg

		_median = data[*middle];

		double *xt = new double[n], *yt = new double[n], *zt = new double[n];
		int *ot = new int[n];

		for(int i=0; i<_count; i++)
		{
			xt[i] = x[idx[i]];
			yt[i] = y[idx[i]];
			zt[i] = z[idx[i]];
			ot[i] = ordering[idx[i]];
		}

		for(int i=0; i<_count; i++)
		{
			_x[i] = xt[i];
			_y[i] = yt[i];
			_z[i] = zt[i];
			_ordering[i] = ot[i];
		}

		delete[] xt;
		delete[] yt;
		delete[] zt;
		delete[] ot;

		const double boxMin[_k] = {_minx, _miny, _minz};
		const double boxMax[_k] = {_maxx, _maxy, _maxz};

		_lower = boxMin[_axis];
		_upper = boxMax[_axis];

	    if(_count > BUCKET)
	    {
	    	_left = new KdTree();
	    	_right = new KdTree();

	    	_left->buildRec(_half, _x, _y, _z, _ordering, depth+1);
	    	_right->buildRec(n-_half, _x+_half, _y+_half, _z+_half, _ordering+_half, depth+1);
	    }
	}

	inline void build(const int n, const double *x, const double *y, const double *z)
	{
		_x = new double[n];
		_y = new double[n];
		_z = new double[n];
		_ordering = new int[n];

		for(int i=0; i<n; i++)
		{
			_x[i] = x[i];
			_y[i] = y[i];
			_z[i] = z[i];
			_ordering[i] = i;
		}

		buildRec(n, _x, _y, _z, _ordering);

		printf("%f %f %f %f %f %f\n", _minx, _maxx, _miny, _maxy, _minz, _maxz);
	}

	//int cc = 0;
	inline void check_add(const double xi, const double yi, const double zi, const double r2, const int ngmax, int *ng, int &nvi) const
	{
		for(int i=0; i<_count; i++)
		{
			double xx = _x[i];
			double yy = _y[i];
			double zz = _z[i];

			if(nvi < ngmax && distancesq(xi, yi, zi, xx, yy, zz) < r2)
				ng[nvi++] = _ordering[i];
		}
		//cc += count;
	}

	inline double inrange(const double xi, const double yi, const double zi, const double r2) const
	{
		double d = 0;
		printf("HERE:\n");
		printf("%f\n", _median);
		printf("%f %f %f %f %f %f\n", _minx, _maxx, _miny, _maxy, _minz, _maxz);
		d += sq(distbnd(xi, _minx, _maxx));
		d += sq(distbnd(yi, _miny, _maxy));
		d += sq(distbnd(zi, _minz, _maxz));
		return d < r2;
	}

	inline bool check(const double *particle, const double xi, const double yi, const double zi, const double r, const double r2, const int ngmax, int *ng, int &nvi) const
	{
		if(inrange(xi, yi, zi, r2))
		{
			if(_count <= BUCKET)
				check_add(xi, yi, zi, r2, ngmax, ng, nvi);
			else
				findNeighborsRec(particle, xi, yi, zi, r, r2, ngmax, ng, nvi);
			return true;
		}
		else
			return false;
	}

	inline void findNeighborsRec(const double *particle, const double xi, const double yi, const double zi, const double r, const double r2, const int ngmax, int *ng, int &nvi) const
	{
		_left->check(particle, xi, yi, zi, r, r2, ngmax, ng, nvi);
		_right->check(particle, xi, yi, zi, r, r2, ngmax, ng, nvi);
	}

	inline void findNeighbors(const double xi, const double yi, const double zi, const double r, const int ngmax, int *ng, int &nvi, 
		const bool /*PBCx*/ = false, const bool /*PBCy*/ = false, const bool /*PBCz*/ = false) const
	{
		const double particle[_k] = {xi, yi, zi};
		if(_left && _right)
			findNeighborsRec(particle, xi, yi, zi, r, r*r, ngmax, ng, nvi);
		else
			check(particle, xi, yi, zi, r, r*r, ngmax, ng, nvi);
	}
 
public:

	KdTree *_left, *_right;
	
	double *_x, *_y, *_z;

	double _minx, _maxx;
	double _miny, _maxy;
	double _minz, _maxz;

	double _lower, _upper;

	double _median;

	int *_ordering;

	int _k, _depth, _axis, _count, _half;
};

}

#endif // SPHEXA_TREE_HPP
