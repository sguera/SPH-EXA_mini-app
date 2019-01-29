#pragma once

namespace sphexa
{

template<typename T = double>
class BBox
{
public:
	BBox(T xmin = -1, T xmax = 1, T ymin = -1, T ymax = 1, T zmin = -1, T zmax = 1, bool PBCx = false, bool PBCy = false, bool PBCz = false) : 
		xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax), PBCx(PBCx), PBCy(PBCy), PBCz(PBCz) {}

	T xmin, xmax, ymin, ymax, zmin, zmax;
	bool PBCx, PBCy, PBCz;
};

}
