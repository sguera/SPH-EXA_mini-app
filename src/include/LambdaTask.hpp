#pragma once

#include <functional>

#include "Task.hpp"

namespace sphexa
{

class LambdaTask : public Task
{
public:
	LambdaTask() = delete;
	
	LambdaTask(const std::function<void()> func) : func(func) {}

	virtual void compute() override { func(); }

private:
	const std::function<void()> func; 
};

}