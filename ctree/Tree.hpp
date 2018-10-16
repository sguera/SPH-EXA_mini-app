#ifndef SPHEXA_TREE_HPP
#define SPHEXA_TREE_HPP

#include <vector>


namespace SPHExa
{

constexpr int const MAXP = 64;
constexpr double const PLANCK = 1e-15;

class Tree
{
public:
	Tree();
	~Tree();

	void clean();

	int cellCount() const;
	
	void init(const double minx, const double maxx, const double miny, const double maxy, const double minz, const double maxz);

	void build(const int n, const double *x, const double *y, const double *z);
	
	void findNeighbors(const int i, const double *x, const double *y, const double *z, const double r, const int ngmax, int *ng, int &nvi, 
		const bool PBCx = false, const bool PBCy = false, const bool PBCz = false) const;
 
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
		const double xi, const double yi, const double zi, const double r, const int ngmax, int *ng, int &nvi) const;

	static inline double normalize(double d, double min, double max);

	static inline double distance(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2);

	static inline void check_add_list(const std::vector<int> &list, const double *x, const double *y, const double *z, const double xi, const double yi, const double zi, const double r, const int ngmax, int *ng, int &nvi);
};

}

#endif // SPHEXA_TREE_HPP
