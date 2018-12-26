#pragma once

#include "Task.hpp"

namespace sphexa
{

class TaskLoop : public Task
{
public:

	TaskLoop() = delete;
	
	TaskLoop(int count) : count(count) {}
	
	virtual void preProcess() {}

	virtual void compute(int i) = 0;

	virtual void postProcess() {}

	virtual void compute()
	{
		preProcess();

		#pragma omp parallel for
		for(int i = 0; i<count; ++i)
			compute(i);

		postProcess();
	}

private:
	int count;
};

}

