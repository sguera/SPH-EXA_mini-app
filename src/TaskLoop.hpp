#pragma once

namespace sphexa
{

class TaskLoop : public Task
{
public:

	TaskLoop() = delete;
	
	TaskLoop(int count) : count(count) {}
	
	virtual void preprocess() {}

	virtual void compute(int i) = 0;

	virtual void postprocess() {}

	virtual void compute()
	{
		preprocess();

		#pragma omp parallel for
		for(int i = 0; i<count; ++i)
			compute(i);

		postprocess();
	}

private:
	int count;
};

}

