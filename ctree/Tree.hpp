#ifndef SPHEXA_TREE_HPP
#define SPHEXA_TREE_HPP

#include <vector>


namespace sphexa
{
constexpr unsigned int MAXP = 8;
constexpr double RATIO = 0.5;
constexpr int TREE = 1;
constexpr int BLOCK_SIZE = 32;
constexpr int PLANCK = 1e-15;

class Tree
{
public:
	Tree();
	~Tree();

	void clean();

	int cellCount();
	int bucketCount();

	void setBox(const double minx, const double maxx, const double miny, const double maxy, const double minz, const double maxz);

	void buildSort(const int n, const double *x, const double *y, const double *z, int **ordering = 0);
	
	void findNeighbors(const double xi, const double yi, const double zi, const double ri, const int ngmax, int *ng, int &nvi, 
		const bool PBCx = false, const bool PBCy = false, const bool PBCz = false);
 
private:
	double _minx, _maxx;
	double _miny, _maxy;
	double _minz, _maxz;

	Tree **_p;
	int B, C;

	int *_start;
	int *_count;

	static double *_x, *_y, *_z;
	static int *_ordering;

	void cleanRec();

	void buildSortRec(const std::vector<int> &list, const double *x, const double *y, const double *z, int it);
	
	void findNeighborsRec(const double xi, const double yi, const double zi, const double ri, const int ngmax, int *ng, int &nvi);

	static inline double normalize(double d, double min, double max);

	static inline double distance(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2);

	static inline void check_add_start(const int start, const int count, const int *ordering, const double *x, const double *y, const double *z, const double xi, const double yi, const double zi, const double r, const int ngmax, int *ng, int &nvi);
};

}

#endif // SPHEXA_TREE_HPP
