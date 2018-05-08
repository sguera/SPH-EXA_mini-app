/*
    file - octree.h

    Templated implementation of an octree for a cpu

 */


#ifndef OCTREE_CPU_H
#define OCTREE_CPU_H

#include "point3d.h"
#include "boundingbox.h"
#include "inneriterator.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <utility>
#include <type_traits>


template <typename InputIterator, class PointExtractor, 
          size_t max_per_node = 16, size_t max_depth = 100>
class Octree {
 public:
  using tree_type = Octree<InputIterator, PointExtractor, max_per_node, max_depth>;

  Octree();

  Octree(InputIterator begin, InputIterator end);

  Octree(InputIterator begin, InputIterator end, PointExtractor f);

  Octree(const tree_type& rhs);

  template <size_t max_per_node_>
  Octree(const Octree<InputIterator, PointExtractor, max_per_node_, max_depth>& rhs);
  
  template <size_t max_depth_>
  Octree(const Octree<InputIterator, PointExtractor, max_per_node, max_depth_>& rhs);
  
  template <size_t max_per_node_, size_t max_depth_>
  Octree(const Octree<InputIterator, PointExtractor, max_per_node_, max_depth_>& rhs);
  
  Octree(tree_type&& rhs);

  void swap(tree_type& rhs);

  template <typename OutputIterator>
  bool search(const BoundingBox& box, OutputIterator& it) const;

  tree_type& operator=(tree_type rhs);

  tree_type& operator=(tree_type&& rhs);

  ~Octree();

  size_t size() const;
 
 private:  
  class Node;

  struct LeafNodeValues {
    std::array<std::pair<InputIterator, Point3d>, max_per_node> values_;
    size_t size_;
  };

  using childNodeArray = std::array<Node*, 8>;
  using maxItemNode = std::vector<std::pair<InputIterator, Point3d>>;

  union NodeValues {
    NodeValues() : internalValue_() {}
    NodeValues(const LeafNodeValues& v) : leafValue_(v) {}
    NodeValues(const childNodeArray& v) : internalValue_(v) {}
    NodeValues(const maxItemNode& v) : maxDepthLeafValue_(v) {}
    NodeValues(const NodeValues& v) : NodeValues() {
      memcpy(this, &v, sizeof(NodeValues));
    }
    NodeValues& operator=(const NodeValues& rhs) {
      memcpy(this, &rhs, sizeof(NodeValues));
      return *this;
    }
    ~NodeValues() {}

    LeafNodeValues leafValue_;
    childNodeArray internalValue_;
    maxItemNode maxDepthLeafValue_;
  };

  enum class NodeContents : char {
    LEAF = 1,
    MAX_DEPTH_LEAF = 2,
    INTERNAL = 4
  };

  class Node {
   public:    
    Node(const std::vector<std::pair<InputIterator, Point3d>>& input_values);

    Node(const std::vector<std::pair<InputIterator, Point3d>>& input_values, 
         const BoundingBox& box,
         size_t current_depth);

    ~Node();

    template <typename OutputIterator>
    bool search(const BoundingBox& box, OutputIterator& it) const;

   private:
    NodeValues value_;
    BoundingBox extrema_;
    NodeContents tag_;

    void init_max_depth_leaf(const std::vector<std::pair<InputIterator, Point3d>>& input_values);

    void init_leaf(const std::vector<std::pair<InputIterator, Point3d>>& input_values);
    
    void init_internal(
        const std::vector<std::pair<InputIterator, Point3d>>& input_values,
        size_t current_depth);

  };

  PointExtractor functor_;
  Node* head_;
  size_t size_;
};

// convenience macros to avoid typing so much
#define OCTREE Octree<InputIterator, PointExtractor, max_per_node, max_depth>
#define OCTREE_TEMPLATE typename InputIterator, class PointExtractor, size_t max_per_node, size_t max_depth

template <OCTREE_TEMPLATE>
OCTREE::Octree(): functor_(PointExtractor()), head_(nullptr), size_(0) {}

template <OCTREE_TEMPLATE>
OCTREE::Octree(InputIterator begin, InputIterator end)
  : Octree(begin, end, PointExtractor()) { }

template <OCTREE_TEMPLATE>
OCTREE::Octree(InputIterator begin, InputIterator end, PointExtractor f)
    : functor_(f), head_(nullptr), size_(0) {

  std::vector<std::pair<InputIterator, Point3d>> v;
  v.reserve(std::distance(begin, end));

  for (auto it = begin; it != end; ++it) {
    v.push_back(std::pair<InputIterator, Point3d>(it, functor_(*it)));
  }
  
  size_ = v.size();
  head_ = new Node(v);
}

template <OCTREE_TEMPLATE>
OCTREE::Octree(OCTREE::tree_type&& rhs) 
  : functor_(rhs.functor), head_(rhs.head_), size_(rhs.size_) { }

template <OCTREE_TEMPLATE>
void OCTREE::swap(OCTREE::tree_type& rhs) {
  std::swap(head_, rhs.head_);
  std::swap(functor_, rhs.functor_);
  std::swap(size_, rhs.size_);
}

template <OCTREE_TEMPLATE>
template <typename OutputIterator>
bool OCTREE::search(const BoundingBox& box, OutputIterator& it) const {
  return head_->search(box, it);
}

template <OCTREE_TEMPLATE>
typename OCTREE::tree_type& OCTREE::operator=(typename OCTREE::tree_type rhs) {
  swap(rhs);
  return *this;
}

template <OCTREE_TEMPLATE>
typename OCTREE::tree_type& OCTREE::operator=(typename OCTREE::tree_type&& rhs) {
  swap(rhs);
  return *this;
}

template <OCTREE_TEMPLATE>
OCTREE::~Octree() {
  delete head_;
}

template <OCTREE_TEMPLATE>
size_t OCTREE::size() const {
  return size_;
}

template <OCTREE_TEMPLATE>
OCTREE::Node::Node(const std::vector<std::pair<InputIterator, Point3d>>& input_values)
  : Node(input_values, 
         makeBoundingBox(
            InnerIterator<InputIterator>(input_values.begin()), 
            InnerIterator<InputIterator>(input_values.end())),
         0) { }

template <OCTREE_TEMPLATE>
OCTREE::Node::Node(
    const std::vector<std::pair<InputIterator, Point3d>>& input_values, 
    const BoundingBox& box,
    size_t current_depth) : extrema_(box)  {
  if (current_depth > max_depth) {
    init_max_depth_leaf(input_values);
  } else if (input_values.size() <= max_per_node) {
    init_leaf(input_values);
  } else {
    init_internal(input_values, current_depth);
  }
}

template <OCTREE_TEMPLATE>
OCTREE::Node::~Node() {
  if (tag_ == NodeContents::INTERNAL) {
    for (auto childPointer : value_.internalValue_) {
      delete childPointer;
    }
    value_.internalValue_.~childNodeArray();
  } else if (tag_ == NodeContents::LEAF) {
    value_.leafValue_.~LeafNodeValues();
  } else if (tag_ == NodeContents::MAX_DEPTH_LEAF) {
    value_.maxDepthLeafValue_.~maxItemNode();
  }
}

template <OCTREE_TEMPLATE>
template <typename OutputIterator>
bool OCTREE::Node::search(const BoundingBox& p, OutputIterator& it) const {
  bool success = false;
  if (tag_ == NodeContents::INTERNAL) {
    for (auto child : value_.internalValue_) {
      if (child) {
        success |= child->search(p, it);
      }
    }
  } else if (tag_ == NodeContents::LEAF) {
    const LeafNodeValues& children = value_.leafValue_;
    for (size_t i = 0; i < children.size_; ++i) {
      const Point3d& point = std::get<1>(children.values_[i]);
      if (p.contains(point)) {
        *it = std::get<0>(children.values_[i]);
        ++it;
        success = true;
      }
    }
  } else if (tag_ == NodeContents::MAX_DEPTH_LEAF) {
    for (auto child : value_.maxDepthLeafValue_) {
      Point3d& point = std::get<1>(child);
      if (p.contains(point)) {
        *it = std::get<0>(child);
        ++it;
        success = true;
      }
    }
  }
  return success;
}

template <OCTREE_TEMPLATE>
void OCTREE::Node::init_max_depth_leaf(
    const std::vector<std::pair<InputIterator, Point3d>>& input_values) {  
  value_ = input_values;
  tag_ = NodeContents::MAX_DEPTH_LEAF;
}

template <OCTREE_TEMPLATE>
void OCTREE::Node::init_leaf(
    const std::vector<std::pair<InputIterator, Point3d>>& input_values)  {
  std::array<std::pair<InputIterator, Point3d>, max_per_node> a;
  std::copy(input_values.begin(), input_values.end(), a.begin());
  value_ = LeafNodeValues{a, input_values.size()};
  tag_ = NodeContents::LEAF;
}

template <OCTREE_TEMPLATE>
void OCTREE::Node::init_internal(
    const std::vector<std::pair<InputIterator, Point3d>>& input_values,
    size_t current_depth)  {
  std::array<std::vector<std::pair<InputIterator, Point3d>>, 8> childVectors;
  std::array<BoundingBox, 8> boxes = extrema_.partition();
  std::array<Node*, 8> children;

  for (unsigned child = 0; child < 8; ++child) {
    std::vector<std::pair<InputIterator, Point3d>>& childVector = childVectors[child];
    childVector.reserve(input_values.size() / 8);

    std::copy_if(
      input_values.begin(), 
      input_values.end(), 
      std::back_inserter(childVector),
      [&boxes, child](const std::pair<InputIterator, Point3d>& element) -> bool {
        Point3d p = std::get<1>(element);
        return boxes[child].contains(p);
      }
    );

    children[child] = childVector.empty()
        ? nullptr
        : new Node(childVector, boxes[child], ++current_depth);
  }

  value_ = children;
  tag_ = NodeContents::INTERNAL;
}

#endif // DEFINED OCTREE_CPU_H