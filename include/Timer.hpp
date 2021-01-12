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

    Timer(std::ostream& out) : out(out) {}

    float duration() { return std::chrono::duration_cast<Time>(tstop - tstart).count(); }

    void start() { tstart = tstop = tlast = Clock::now(); }

    void stop() { tstop = Clock::now(); }

    void step(const std::string &name)
    {
        stop();
        out << "# " << name << ": " << std::chrono::duration_cast<Time>(tstop - tlast).count() << "s" << std::endl;
        tlast = tstop;
    }

private:
    std::ostream& out;
    TimePoint tstart, tstop, tlast;
};

class MasterProcessTimer : public Timer
{
public:
    MasterProcessTimer(std::ostream& out, int rank)
        : Timer(out), rank(rank)
    {
    }

    float duration() { return rank == 0 ? Timer::duration() : 0.0f; }

    void start()
    {
        if (rank == 0) Timer::start();
    }
    void stop()
    {
        if (rank == 0) Timer::stop();
    }
    void step(const std::string &name)
    {
        if (rank == 0) Timer::step(name);
    }

private:
    int rank;
};

} // namespace sphexa
