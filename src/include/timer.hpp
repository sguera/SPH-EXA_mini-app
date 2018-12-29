#pragma once

#include <chrono>
#include <iostream>
#include <functional>

namespace sphexa
{
	namespace timer
	{
		typedef std::chrono::high_resolution_clock Clock;
		typedef std::chrono::time_point<Clock> TimePoint;
		typedef std::chrono::duration<float> Time;

		inline float duration(const TimePoint start, const TimePoint stop)
		{
			return std::chrono::duration_cast<Time>(stop-start).count();
		}

		void report_time(std::function<void()> call, const std::string &name)
		{
			TimePoint start = Clock::now();

			call();
			
			TimePoint stop = Clock:: now();

			std::cout << "# " << name << ": total time " << duration(start, stop) << "s" << std::endl;
		}
	}
}
