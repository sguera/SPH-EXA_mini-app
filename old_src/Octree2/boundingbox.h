#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include <array>
#include <cstddef>
#include <limits>
#include <cmath>
#include <iostream>

#include "point3d.h"

using limits = std::numeric_limits<double>;

struct BoundingBox {
  bool contains(const BoundingBox& other) const;
  bool contains(const Point3d& p) const;

  BoundingBox overlap(const BoundingBox& other) const;
  std::array<BoundingBox, 8> partition() const;

  bool operator==(const BoundingBox& rhs) const;
  bool operator!=(const BoundingBox& rhs) const;

  std::size_t getChildPartitionIndex(const Point3d& p) const;

  Point3d mins_, maxes_;
};

static const BoundingBox initialBox = {
  { limits::max(), limits::max(), limits::max() },
  { limits::min(), limits::min(), limits::min() }
};

static const BoundingBox invalidBox = {
  { limits::quiet_NaN(), limits::quiet_NaN(), limits::quiet_NaN() },
  { limits::quiet_NaN(), limits::quiet_NaN(), limits::quiet_NaN() }
};

template <typename InputIterator>
BoundingBox makeBoundingBox(InputIterator begin, InputIterator end) {
  BoundingBox returnBox = initialBox;

  for (auto it = begin; it != end; ++it) {
    returnBox.mins_.x = std::min(it->x, returnBox.mins_.x);
    returnBox.mins_.y = std::min(it->y, returnBox.mins_.y);
    returnBox.mins_.z = std::min(it->z, returnBox.mins_.z);
    returnBox.maxes_.x = std::max(it->x, returnBox.maxes_.x);
    returnBox.maxes_.y = std::max(it->y, returnBox.maxes_.y);
    returnBox.maxes_.z = std::max(it->z, returnBox.maxes_.z);
  }

  return returnBox;
}

#endif // defined BOUNDINGBOX_H