#pragma once

#include <exception>

namespace sphexa
{
struct FileNotOpenedException : std::runtime_error
{
    using std::runtime_error::runtime_error;
};
struct MPIFileNotOpenedException : std::runtime_error
{
    MPIFileNotOpenedException(const std::string &what, const int mpierr) : std::runtime_error(what), mpierr(mpierr) {}

    const int mpierr;
};
} // namespace sphexa
