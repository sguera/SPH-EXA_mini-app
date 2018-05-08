#include "point3d.h"

#include <cmath>
#include <iostream>

bool Point3d::isNaN() const {
  return std::isnan(x) && std::isnan(y) && std::isnan(z);
}

bool Point3d::operator==(const Point3d& other) const {
  // TODO - give some leeway cuz floating point
  return (x == other.x && y == other.y && z == other.z) || 
         (isNaN() || other.isNaN());
}

bool Point3d::operator!=(const Point3d& other) const {
  return !operator==(other);
}

std::ostream& operator<<(std::ostream& out, const Point3d& rhs) {
	out << "{" << rhs.x << ", " << rhs.y << ", " << rhs.z << "}";
	return out;
}