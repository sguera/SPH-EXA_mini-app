#ifndef UTILS_HPP
#define UTILS_HPP

inline double distance(const double x1, const double y1, const double z1, const double x2, const double y2, const double z2)
{
	double xx = x1 - x2;
	double yy = y1 - y2;
	double zz = z1 - z2;

	return sqrt(xx*xx + yy*yy + zz*zz);
}

#endif