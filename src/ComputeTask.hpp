#pragma once

#include "Task.hpp"

class ComputeTask : public Task
{
public:
	
	virtual void compute() = 0;

	void exec()
	{
		compute();
	}
};

