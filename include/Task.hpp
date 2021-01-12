#pragma once

#include <vector>

namespace sphexa
{

struct Task
{
    Task(const size_t ngmax, const size_t ng0)
        : ngmax(ngmax)
        , ng0(ng0)
    {
    }

    void resize(const size_t size)
    {
        clist.resize(size);
        neighbors.resize(size * ngmax);
        neighborsCount.resize(size);
    }

    const size_t ngmax;
    const size_t ng0;

    std::vector<int> clist;
    std::vector<int> neighbors;
    std::vector<int> neighborsCount;
};

class TaskList
{
public:
    TaskList(const std::vector<int> &clist, const size_t nTasks, const size_t ngmax, const size_t ng0)
        : ngmax(ngmax)
        , ng0(ng0)
        , nTasks(nTasks)
        , tasks(create(clist, nTasks, ngmax, ng0))
    {
    }

    void update(const std::vector<int> &clist)
    {
        const int partitionSize = clist.size() / nTasks;
        const int lastPartitionOffset = clist.size() - nTasks * partitionSize;

        initTasks(clist, partitionSize, lastPartitionOffset, tasks);
    }

    const size_t ngmax;
    const size_t ng0;
    const size_t nTasks;
    std::vector<Task> tasks;

private:
    std::vector<Task> create(const std::vector<int> &clist, const size_t nTasks, const size_t ngmax, const size_t ng0)
    {
        const int partitionSize = clist.size() / nTasks;
        const int lastPartitionOffset = clist.size() - nTasks * partitionSize;

        std::vector<Task> taskList(nTasks, Task(ngmax, ng0));

        initTasks(clist, partitionSize, lastPartitionOffset, taskList);

        return taskList;
    }

    void initTasks(const std::vector<int> clist, const size_t partitionSize, const size_t lastPartitionOffset,
                   std::vector<Task> &tasksToInit)
    {
#pragma omp parallel for
        for (size_t i = 0; i < nTasks; ++i)
        {
            const size_t begin = i * partitionSize;
            const size_t end = (i + 1) * partitionSize + (i == nTasks - 1 ? lastPartitionOffset : 0);
            const size_t size = end - begin;

            tasksToInit[i].resize(size);
            for (size_t j = 0; j < size; j++)
                tasksToInit[i].clist[j] = clist[j + begin];
        }
    }
};
} // namespace sphexa
