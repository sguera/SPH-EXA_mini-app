#pragma once

#include <vector>
#include "Task.hpp"

namespace sphexa
{

class TaskScheduler
{
public:
	void add(Task *t)
	{
		tasks.push_back(t);
	}

	void exec()
	{
		for(auto t : tasks)
		{
			t->exec();
		}
	}

private:
	std::vector<Task*> tasks;
	std::vector<double> times;
};

}