#pragma once

namespace sphexa
{

class BBox
{
public:
	BBox(double xmin = -1, double xmax = 1, double ymin = -1, double ymax = 1, double zmin = -1, double zmax = 1) : 
		xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax) {}

	double xmin, xmax, ymin, ymax, zmin, zmax;
};

}

