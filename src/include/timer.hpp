#pragma once

#include <chrono>
#include <iostream>
#include <functional>

namespace sphexa
{

class Timer
{
public:
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::time_point<Clock> TimePoint;
    typedef std::chrono::duration<float> Time;

    float duration() { return std::chrono::duration_cast<Time>(tstop - tstart).count(); }

    void start() { tstart = tstop = tlast = Clock::now(); }

    void stop() { tstop = Clock::now(); }

    void step(const std::string &name)
    {
        stop();
        std::cout << "# " << name << ": " << std::chrono::duration_cast<Time>(tstop - tlast).count() << "s" << std::endl;
        tlast = tstop;
    }

private:
    TimePoint tstart, tstop, tlast;
};

} // namespace sphexa
