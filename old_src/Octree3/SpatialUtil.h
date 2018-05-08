#ifndef SPATIAL_UTILS_H_
#define SPATIAL_UTILS_H_

namespace sphexa {
template <unsigned int D> struct bbox;
    
///
/// @brief Contains the minimum and maximum extents of a hypercube in @p D
/// dimensional space
///
/// @tparam D the number of spatial dimensions
///
template <unsigned int D> struct bbox {
  typedef Vector<double, D> double_d;

  ///
  /// @brief minimum point in the box (i.e. lower left corner for D=2)
  ///
  double_d bmin;

  ///
  /// @brief maximum point in the box (i.e. upper right corner for D=2)
  ///
  double_d bmax;

  inline bbox()
      : bmin(double_d::Constant(detail::get_max<double>())),
        bmax(double_d::Constant(-detail::get_max<double>())) {}

  inline bbox(const double_d &p) : bmin(p), bmax(p) {}

  inline bbox(const double_d &min, const double_d &max)
      : bmin(min), bmax(max) {}

  ///
  /// @return the bounding box covering both input boxes
  ///
  inline bbox operator+(const bbox &arg) {
    bbox bounds;
    for (size_t i = 0; i < D; ++i) {
      bounds.bmin[i] = std::min(bmin[i], arg.bmin[i]);
      bounds.bmax[i] = std::max(bmax[i], arg.bmax[i]);
    }
    return bounds;
  }

  ///
  /// @return true if lhs box is within rhs box
  ///
  inline bool operator<(const bbox &arg) {
    bbox bounds;
    bool within = true;
    for (size_t i = 0; i < D; ++i) {
      within |= bmin[i] >= arg.bmin[i];
      within |= bmax[i] < arg.bmax[i];
    }
    return within;
  }

  ///
  /// @return true if lhs box is the same or within rhs box
  ///
  inline bool operator<=(const bbox &arg) {
    bbox bounds;
    bool within = true;
    for (size_t i = 0; i < D; ++i) {
      within |= bmin[i] >= arg.bmin[i];
      within |= bmax[i] <= arg.bmax[i];
    }
    return within;
  }

  ///
  /// @return true if box has no volume
  ///
  inline bool is_empty() {
    for (size_t i = 0; i < D; ++i) {
      if (bmax[i] < bmin[i])
        return true;
    }
    return false;
  }
};

///
/// @brief print bbox to a stream
///
/// @tparam D the number of spatial dimensions
/// @param out the stream
/// @param b the box to print
///
template <unsigned int D>
std::ostream &operator<<(std::ostream &out, const bbox<D> &b) {
  return out << "bbox(" << b.bmin << "<->" << b.bmax << ")";
}

}

#endif
