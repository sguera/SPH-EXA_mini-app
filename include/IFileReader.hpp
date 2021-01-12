#pragma once

#include <string>
#include <vector>

namespace sphexa
{
template <typename Dataset>
struct IFileReader
{
    virtual Dataset readParticleDataFromBinFile(const std::string &path, const size_t noParticles) const = 0;
    virtual Dataset readParticleDataFromCheckpointBinFile(const std::string &path) const = 0;

    virtual ~IFileReader() = default;
};
} // namespace sphexa
