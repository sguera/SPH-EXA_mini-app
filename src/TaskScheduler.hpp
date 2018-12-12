#pragma once

#include <vector>
#include <chrono>
#include <string>

#include "Task.hpp"

using namespace std::chrono;

namespace sphexa
{

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> TimePoint;
typedef std::chrono::duration<float> Time;

class TaskScheduler
{
public:
	struct Params
	{
		Params(int verbose = 0, const std::string name = "") :
			verbose(verbose), name(name) {}
		int verbose;
		const std::string name;
	};

public:
	void add(Task *t, Params p = Params())
	{
		tasks.push_back(t);
		params.push_back(p);
	}

	void exec()
	{
		float totalTime = 0;

		for(unsigned int i=0; i<tasks.size(); i++)
		{
			TimePoint start = Clock::now();

			tasks[i]->compute();

			TimePoint stop = Clock::now();

			float ms = duration_cast<duration<float>>(stop-start).count();

			totalTime += ms;

			if(params[i].verbose)
				std::cout << "# " << params[i].name << ": total time " << ms << "s" << std::endl;
		}

		std::cout << "=== Total time for iteration " << totalTime << "s" <<std::endl;
	}

private:
	std::vector<Task*> tasks;
	std::vector<Params> params;
};

}

