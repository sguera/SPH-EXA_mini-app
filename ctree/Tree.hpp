#ifndef TREE_HPP
#define TREE_HPP

#include <vector>

#define MAXP 64
#define PLANCK 1e-15

class Tree
{
public:
	Tree();
	~Tree();

	void clean();

	int cellCount();
	
	void init(const double minx, const double maxx, const double miny, const double maxy, const double minz, const double maxz);

	void build(const int n, const double *x, const double *y, const double *z);
	
	void findNeighbors(const int i, const double *x, const double *y, const double *z, const double r, const int ngmax, int *ng, int &nvi, 
		const bool PBCx = false, const bool PBCy = false, const bool PBCz = false);

private:
	double _minx, _maxx;
	double _miny, _maxy;
	double _minz, _maxz;

	int C;
	Tree **_p;
	std::vector<int> *_list;

	void cleanRec();

	void buildRec(const std::vector<int> &list, const double *x, const double *y, const double *z);

	void findNeighborsRec(const double *x, const double *y, const double *z, 
		const double xi, const double yi, const double zi, const double r, const int ngmax, int *ng, int &nvi);
};

#endif