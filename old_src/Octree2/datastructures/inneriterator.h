#ifndef INNER_ITERATOR_H_DEFINED
#define INNER_ITERATOR_H_DEFINED

#include <utility>
#include <vector>

#include "point3d.h"

template <typename InputIterator>
struct InnerIterator {
  using wrapped_type = typename std::vector<std::pair<InputIterator, Point3d>>::const_iterator;
  wrapped_type it_;

  InnerIterator(wrapped_type it) : it_(it) {}

  const Point3d& operator*() const {
    return std::get<1>(*it_);
  }

  const Point3d* operator->() const {
    return &std::get<1>(*it_);
  } 

  InnerIterator<InputIterator>& operator++() {
    ++it_;
    return *this;
  }

  InnerIterator<InputIterator> operator++(int) {
    InnerIterator<InputIterator> other = *this;
    ++it_;
    return other;
  }

  bool operator==(const InnerIterator<InputIterator>& rhs) const {
    return it_ == rhs.it_;
  }

  bool operator!=(const InnerIterator<InputIterator>& rhs) const {
    return !operator==(rhs);
  }
};

#endif // defined INNER_ITERATOR_H_DEFINED