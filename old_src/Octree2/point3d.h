#ifndef POINT3D_H_
#define POINT3D_H_

#include <iostream>

struct Point3d {
  double x, y, z;

  bool isNaN() const;

  bool operator==(const Point3d& other) const;
  bool operator!=(const Point3d& other) const;
};

std::ostream& operator<<(std::ostream& out, const Point3d& rhs);

#endif // defined POINT3D_H_