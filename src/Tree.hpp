#ifndef TREE_HPP
#define TREE_HPP

#include <vector>

extern unsigned int MAXP;
extern double RATIO;
extern int TREE;
extern int BLOCK_SIZE;
extern int PLANCK;

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
};

#endif
