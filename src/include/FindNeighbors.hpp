#pragma once

#include <vector>

#include "TaskLoop.hpp"

namespace sphexa
{

template<class Tree>
class FindNeighbors : public TaskLoop
{
public:
	
	FindNeighbors() = delete;
	
	~FindNeighbors() = default;

	FindNeighbors(const Tree &tree, std::vector<std::vector<int>> &neighbors) : 
		TaskLoop(neighbors.size()), tree(tree), neighbors(neighbors) {}

	virtual void compute(int i) override
	{
		neighbors[i].resize(0);
        tree.findNeighbors(i, neighbors[i]);
	}

private:
	const Tree &tree;
	std::vector<std::vector<int>> &neighbors;
};



}