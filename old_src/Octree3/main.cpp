#ifndef SPATIAL_DATA_STRUCTURES_H_
#define SPATIAL_DATA_STRUCTURES_H_

#include <chrono>
typedef std::chrono::system_clock Clock;

using namespace sphexa;


int main () {
N = 100;
    typedef Particles<std::tuple<>, 2> Particles_t;
    typedef typename Particles_t::position position;
    Particles_t particles(N);

    std::uniform_real_distribution<double> uniform(0, 1);
    for (size_t i = 0; i < N; ++i) {
      auto &gen = get<generator>(particles)[i];
      get<position>(particles)[i] = vdouble2(uniform(gen), uniform(gen));
    }

    particles.init_neighbour_search(vdouble2(0, 0), vdouble2(1, 1),
                                    vdouble2(false, false));


typedef Particles_t::query_type Query_t;
    Query_t query = particles.get_query();


for (auto i = query.get_children(); i != false; ++i) {
      std::cout << query.get_bounds(i) << std::endl;
    }

typedef Particles<std::tuple<>, 2, std::vector, HyperOctree>
        ParticlesOcttree_t;
    ParticlesOcttree_t particles_octtree(N);

    for (size_t i = 0; i < N; ++i) {
      auto &gen = get<generator>(particles_octtree)[i];
      get<position>(particles_octtree)[i] =
          vdouble2(uniform(gen), uniform(gen));
    }

    particles_octtree.init_neighbour_search(vdouble2(0, 0), vdouble2(1, 1),
                                            vdouble2(false, false));

    auto query_octtree = particles_octtree.get_query();


    std::cout << "recursive depth-first" << std::endl;
    for (auto i = query_octtree.get_children(); i != false; ++i) {
      std::function<void(decltype(i) &)> depth_first;
      depth_first = [&](const auto &parent) {
        std::cout << query_octtree.get_bounds(parent) << std::endl;
        for (auto i = query_octtree.get_children(parent); i != false; ++i) {
          depth_first(i);
        }
      };
      depth_first(i);
    }

    /*`
    This construction might be a bit clumsy to use in practice however, so
    Aboria provides a special depth-first iterator
    [classref Aboria::NeighbourQueryBase::all_iterator] to allow you to write a
    loop equivalent to the recursive depth-first code given above.
    The [memberref Aboria::NeighbourQueryBase::get_subtree] function returns a
    [classref Aboria::NeighbourQueryBase::all_iterator] that performs a
    depth-first iteration over the tree. Note that you can also pass in a
    child_iterator to [memberref Aboria::NeighbourQueryBase::get_subtree] to
    iterate over the sub-tree below a particular node of the tree.
    */

    std::cout << "subtree depth-first" << std::endl;
    for (auto i = query_octtree.get_subtree(); i != false; ++i) {
      std::cout << query_octtree.get_bounds(i) << std::endl;
    }

    /*`
    You might also want to distinguish between leaf nodes (nodes with no
    children) and non-leaf nodes. You can do this with the
    [memberref Aboria::NeighbourQueryBase::is_leaf_node] function, which takes a
    reference to a node (rather than an iterator), and can be used like so
    */

    std::cout << "subtree depth-first showing leaf nodes" << std::endl;
    for (auto i = query_octtree.get_subtree(); i != false; ++i) {
      if (query_octtree.is_leaf_node(*i)) {
        std::cout << "leaf node with bounds = " << query_octtree.get_bounds(i)
                  << std::endl;
      } else {
        std::cout << "non-leaf node with bounds = "
                  << query_octtree.get_bounds(i) << std::endl;
      }
    }


    std::cout << "subtree depth-first showing leaf nodes and particles"
              << std::endl;
    for (auto i = query_octtree.get_subtree(); i != false; ++i) {
      if (query_octtree.is_leaf_node(*i)) {
        std::cout << "leaf node with bounds = " << query_octtree.get_bounds(i)
                  << std::endl;
        for (auto j = query_octtree.get_bucket_particles(*i); j != false; ++j) {
          std::cout << "\t has particle with position" << get<position>(*j)
                    << std::endl;
        }
      } else {
        std::cout << "non-leaf node with bounds = "
                  << query_octtree.get_bounds(i) << std::endl;
      }
    }

}
