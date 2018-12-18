#pragma once

namespace sphexa
{

class BBox
{
public:
	BBox(double xmin = -1, double xmax = 1, double ymin = -1, double ymax = 1, double zmin = -1, double zmax = 1, bool PBCx = false, bool PBCy = false, bool PBCz = false) : 
		xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax), PBCx(PBCx), PBCy(PBCy), PBCz(PBCz) {}

	double xmin, xmax, ymin, ymax, zmin, zmax;
	bool PBCx, PBCy, PBCz;
};

}

