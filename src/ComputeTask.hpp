#pragma once

class ComputeTask : public Task
{

public:
	
	virtual compute() = 0;

	void exec(){

		compute();
	}
}