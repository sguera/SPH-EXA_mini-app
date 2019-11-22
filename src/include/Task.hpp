#pragma once

#include <vector>
#include <atomic>
#include <mutex>

namespace sphexa
{
struct Task
{
    Task(size_t size = 0)
        : clist(size)
        , neighbors(size * ngmax)
        , neighborsCount(size)
    {
    }

    void resize(size_t size)
    {
        clist.resize(size);
        neighbors.resize(size * ngmax);
        neighborsCount.resize(size);
    }

    std::vector<int> clist;
    std::vector<int> neighbors;
    std::vector<int> neighborsCount;

    constexpr static size_t ngmax = 650;
};

struct TaskQueue
{
    TaskQueue(std::vector<Task> &taskList)
        : taskList(taskList)
    {
    }
    bool areAllProcessed() { return lastProcessedTask == taskList.size(); }
    size_t size() { return taskList.size(); }
    Task &pop() {
        const std::lock_guard<std::mutex> lock(mtx);
        printf("Queue: Returning task %lu\n", lastProcessedTask.load());

        return taskList[lastProcessedTask++];
    }
    std::mutex mtx;
    std::vector<Task> &taskList;
    std::atomic<size_t> lastProcessedTask{0};
};

} // namespace sphexa
