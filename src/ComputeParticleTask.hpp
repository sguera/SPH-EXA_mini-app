#pragma once

namespace sphexa
{

class ComputeParticleTask : public Task
{
public:

	ComputeParticleTask(int count) : count(count) {}
	
	virtual void preprocess() {}

	virtual void compute(int particle_id) = 0;

	virtual void postprocess() {}

	void exec()
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