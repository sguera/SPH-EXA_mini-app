#pragma once

#include "sphexa.hpp"

namespace gravity
{

template <class I, class T>
void gravityTreeWalk(std::vector<sphexa::Task>& taskList, const std::vector<I> &tree, cstone::Octree<I, cstone::GlobalTree> globalTree,
                            cstone::Octree<I, cstone::LocalTree> localTree, const std::vector<T> &x, const std::vector<T> &y,
                            const std::vector<T> &z, const std::vector<T> &m, const std::vector<I> &codes, const cstone::Box<T> &box,
                            bool withGravitySync = false)
{
}

} // namespace gravity
