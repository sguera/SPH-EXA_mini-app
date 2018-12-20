#pragma once

#include "Task.hpp"

namespace sphexa
{

template<class Tree, typename T = double>
class BuildTree : public Task
{
public:
	
	BuildTree() = delete;
	
	~BuildTree() = default;

	BuildTree(const BBox<T> &bbox, Tree &tree) : 
		bbox(bbox), tree(tree) {}

	virtual void compute() override
	{
		tree.build(bbox);
	}

private:
	const BBox<T> &bbox;
	Tree &tree;
};

}