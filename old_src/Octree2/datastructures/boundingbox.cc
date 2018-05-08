#include "boundingbox.h"

#include <limits>
#include <cmath>
#include <iostream>
#include <array>

using std::array;

bool BoundingBox::contains(const BoundingBox& other) const {
  return mins_.x <= other.mins_.x && maxes_.x >= other.maxes_.x &&
         mins_.y <= other.mins_.y && maxes_.y >= other.maxes_.y &&
         mins_.z <= other.mins_.z && maxes_.z >= other.maxes_.z;
}

bool BoundingBox::contains(const Point3d& point) const {
  return mins_.x <= point.x && point.x <= maxes_.x &&
         mins_.y <= point.y && point.y <= maxes_.y &&
         mins_.z <= point.z && point.z <= maxes_.z;
}

BoundingBox BoundingBox::overlap(const BoundingBox& other) const {
  // trivial cases
  if (contains(other)) {
    return other;
  } else if (other.contains(*this)) {
    return *this;
  } 

  // Check if there is no intersection
  if (maxes_.x < other.mins_.x || mins_.x > other.maxes_.x ||
      maxes_.y < other.mins_.y || mins_.y > other.maxes_.y ||
      maxes_.z < other.mins_.z || mins_.z > other.maxes_.z) {
    return invalidBox;
  }

  // Actually calculate the bounds
  double upperX = std::min(maxes_.x, other.maxes_.x);
  double upperY = std::min(maxes_.y, other.maxes_.y);
  double upperZ = std::min(maxes_.z, other.maxes_.z);

  double lowerX = std::max(mins_.x, other.mins_.x);
  double lowerY = std::max(mins_.y, other.mins_.y);
  double lowerZ = std::max(mins_.z, other.mins_.z);

  return BoundingBox{
    { lowerX, lowerY, lowerZ },
    { upperX, upperY, upperZ }
  };
}

array<BoundingBox, 8> BoundingBox::partition() const {
  const double xmid = (maxes_.x - mins_.x) / 2.;
  const double ymid = (maxes_.y - mins_.y) / 2.;
  const double zmid = (maxes_.z - mins_.z) / 2.;

  std::array<BoundingBox, 8> ret{{
    BoundingBox{{mins_.x, mins_.y, mins_.z}, {xmid, ymid, zmid}},   // bottom left front
    BoundingBox{{xmid, mins_.y, mins_.z}, {maxes_.x, ymid, zmid}},  // bottom right front
    BoundingBox{{mins_.x, ymid, mins_.z}, {xmid, maxes_.y, zmid}},  // bottom left back
    BoundingBox{{xmid, ymid, mins_.z}, {maxes_.x, maxes_.y, zmid}}, // bottom right back
    BoundingBox{{mins_.x, mins_.y, zmid}, {xmid, ymid, maxes_.z}},  // top left front
    BoundingBox{{xmid, mins_.y, zmid}, {maxes_.x, ymid, maxes_.z}}, // top right front
    BoundingBox{{mins_.x, ymid, zmid}, {xmid, maxes_.y, maxes_.z}}, // top left back
    BoundingBox{{xmid, ymid, zmid}, {maxes_.x, maxes_.y, maxes_.z}} // top right back
  }};
  return ret;
}

bool BoundingBox::operator==(const BoundingBox& rhs) const {
  bool allEqual = rhs.mins_ == mins_ && rhs.maxes_ == maxes_;
  if (allEqual) return true;

  bool allNaN = rhs.maxes_.isNaN() && rhs.mins_.isNaN() && 
                maxes_.isNaN() && mins_.isNaN();
  if (allNaN) return true;

  return false;
}

bool BoundingBox::operator!=(const BoundingBox& rhs) const {
  return !operator==(rhs);
}

std::ostream& operator<<(std::ostream& out, const BoundingBox& rhs) {
  out << "{ " << rhs.mins_ << ", " << rhs.maxes_ << " }";
  return out;
}


std::size_t BoundingBox::getChildPartitionIndex(const Point3d& p) const {
  // children are ordered left to right, front to back, bottom to top.

  double xmid = (maxes_.x - mins_.x) / 2.;
  double ymid = (maxes_.y - mins_.y) / 2.;
  double zmid = (maxes_.z - mins_.z) / 2.;
  bool left = p.x < xmid && p.x >= mins_.x;
  bool front = p.y < ymid && p.y >= mins_.y;
  bool bottom = p.z < zmid && p.z >= mins_.z;

  return (!bottom << 2) | (!left << 1) | !front;
}