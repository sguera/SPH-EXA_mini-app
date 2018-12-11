#ifndef SPHEXA_NNFTREE_HPP
#define SPHEXA_NNFTREE_HPP

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include "nanoflann.hpp"

using namespace std;

#define BUCKET 64
#define _GLIBCXX_PARALLEL

namespace sphexa
{

template <typename T>
struct PointCloud
{
	struct Point { T  x,y,z; };

	std::vector<Point>  pts;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline T kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		if (dim == 0) return pts[idx].x;
		else if (dim == 1) return pts[idx].y;
		else return pts[idx].z;
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};

class SearchResultRadiusSq
{
public:
  /// @brief Deleted default constructor
	SearchResultRadiusSq() = delete;

  /// @param[in] radius: range of the search
	SearchResultRadiusSq(const double radius, const int ngmax, int *ng, int &nvi) : 
		_r2(radius*radius), _ngmax(ngmax), ng(ng), nvi(nvi)
	{
	}

  /// @brief Default destructor
	~SearchResultRadiusSq() = default;

  /// @brief Function called during search to add an element matching the criteria
  /// @param[in] distance2: square of distance between the origin of the search and the point
  /// @param[in] index: index of the point
  /// @return true if the search should be continued, false if it sufficient
	bool addPoint(const double distance2, const int index)
	{
		if (distance2 < _r2)
			ng[nvi++] = index;
		return true;
	}

  /// @brief Return worst distance
	inline double worstDist() const {return _r2;}

  /// @brief Get the size of results
	inline int size() const {return nvi;}

  /// @brief Test if this is full
	inline bool full() const {return nvi == _ngmax;}

	const double _r2;
	const int _ngmax;
	int *ng;
	int &nvi;
};

class NNFTree
{
public:
	typedef nanoflann::KDTreeSingleIndexAdaptor<
	nanoflann::L2_Simple_Adaptor<double, PointCloud<double> > ,
	PointCloud<double>,
		3 /* dim */
	> my_kd_tree_t;

	NNFTree() : _index(NULL) {};

	void clean()
	{
		delete _index;
	}

	void setBox(const double /*minx*/, const double /*maxx*/, const double /*miny*/, const double /*maxy*/, const double /*minz*/, const double /*maxz*/)
	{ }

	void build(const int n, const double *x, const double *y, const double *z)
	{
		_points.pts.resize(n);
		for(int i = 0; i < n; i++)
		{
			_points.pts[i].x = x[i];
			_points.pts[i].y = y[i];
			_points.pts[i].z = z[i];
		}

		_index = new my_kd_tree_t(3 /*dim*/, _points, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );

		_index->buildIndex();
	}

	inline void findNeighbors(const double xi, const double yi, const double zi, const double r, const int ngmax, int *ng, int &nvi, 
		const bool /*PBCx*/ = false, const bool /*PBCy*/ = false, const bool /*PBCz*/ = false) const
	{
		const double query_pt[3] = {xi, yi, zi};

		SearchResultRadiusSq resultSet(r, ngmax, ng, nvi);

		_index->findNeighbors(resultSet, query_pt, nanoflann::SearchParams());
	}

public:

	PointCloud<double> _points;
	my_kd_tree_t *_index;
};

}

#endif // SPHEXA_TREE_HPP
