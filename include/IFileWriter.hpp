#pragma once

#include <string>
#include <vector>

namespace sphexa
{
template <typename Dataset>
struct IFileWriter
{
    virtual void dumpParticleDataToBinFile(const Dataset& d, const std::string &path) const = 0;
    virtual void dumpParticleDataToAsciiFile(const Dataset& d, const std::vector<int> &clist, const std::string &path) const = 0;
    virtual void dumpCheckpointDataToBinFile(const Dataset& d, const std::string &path) const = 0;

    virtual ~IFileWriter() = default;
};
} // namespace sphexa
