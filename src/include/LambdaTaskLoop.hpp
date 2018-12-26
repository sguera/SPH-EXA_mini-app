#pragma once

#include <functional>

#include "TaskLoop.hpp"

namespace sphexa
{

class LambdaTaskLoop : public TaskLoop
{
public:
	LambdaTaskLoop() = delete;

    LambdaTaskLoop(int n, const std::function<void(int i)> func) : TaskLoop(n), func(func) {}

    virtual void compute(int i) override { func(i); }

private:
    const std::function<void(int i)> func; 
};

}