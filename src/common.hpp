#pragma once

namespace sphexa
{

#define PI 3.141592653589793
	
template<typename T>
inline T distance(const T x1, const T y1, const T z1, const T x2, const T y2, const T z2)
{
	T xx = x1 - x2;
	T yy = y1 - y2;
	T zz = z1 - z2;

	return sqrt(xx*xx + yy*yy + zz*zz);
}

}

