#pragma once

namespace sphexa
{

template<class Tree, typename T = double>
class BuildTree
{
public:

	BuildTree(const BBox<T> &bbox, Tree &tree) : bbox(bbox), tree(tree) {}

	void compute()
	{
		tree.build(bbox);
	}

private:
	const BBox<T> &bbox;
	Tree &tree;
};

}