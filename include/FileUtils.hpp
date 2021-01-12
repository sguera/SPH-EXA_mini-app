#pragma once

#include <fstream>
#include <vector>

#ifdef USE_MPI
#include "MPIFileUtils.hpp"
#endif
#include "Exceptions.hpp"

namespace sphexa
{
namespace fileutils
{
namespace details
{
void writeParticleDataToBinFile(std::ofstream &) {}

template <typename Arg, typename... Args>
void writeParticleDataToBinFile(std::ofstream &file, const Arg &first, Args &&... args)
{
    file.write((char *)&first[0], first.size() * sizeof(first[0]));

    writeParticleDataToBinFile(file, args...);
}

void writeParticleDataToAsciiFile(std::ostream &, const int, const char) {}

template <typename Arg, typename... Args>
void writeParticleDataToAsciiFile(std::ostream &file, const int idx, const char separator, const Arg &first, Args &&... data)
{
    file << first[idx] << separator;

    writeParticleDataToAsciiFile(file, idx, separator, data...);
}

void readParticleDataFromBinFile(std::ifstream &) {}

template <typename Arg, typename... Args>
void readParticleDataFromBinFile(std::ifstream &file, Arg &first, Args &&... args)
{
    file.read(reinterpret_cast<char *>(&first[0]), first.size() * sizeof(first[0]));

    readParticleDataFromBinFile(file, args...);
}

} // namespace details

template <typename Dataset, typename... Args>
void writeParticleCheckpointDataToBinFile(const Dataset &d, const std::string &path, Args &&... data)
{
    std::ofstream checkpoint;
    checkpoint.open(path, std::ofstream::out | std::ofstream::binary);

    if (checkpoint.is_open())
    {
        checkpoint.write((char *)&d.n, sizeof(d.n));
        checkpoint.write((char *)&d.ttot, sizeof(d.ttot));
        checkpoint.write((char *)&d.minDt, sizeof(d.minDt));

        details::writeParticleDataToBinFile(checkpoint, data...);

        checkpoint.close();
    }
    else
    {
        throw FileNotOpenedException("Can't open file to write Checkpoint at path: " + path);
    }
}

template <typename... Args>
void writeParticleDataToBinFile(const std::string &path, Args &&... data)
{
    std::ofstream checkpoint;
    checkpoint.open(path, std::ofstream::out | std::ofstream::binary);

    if (checkpoint.is_open())
    {
        details::writeParticleDataToBinFile(checkpoint, data...);

        checkpoint.close();
    }
    else
    {
        throw FileNotOpenedException("Can't open file at path: " + path);
    }
}

template <typename... Args>
void writeParticleDataToAsciiFile(const std::vector<int> &clist, const std::string &path, const bool append, const char separator, Args &&... data)
{
    std::ios_base::openmode mode;
    if (append)
        mode = std::ofstream::app;
    else
        mode = std::ofstream::out;

    std::ofstream dump(path, mode);

    if (dump.is_open())
    {
        for (size_t pi = 0; pi < clist.size(); ++pi)
        {
            const int i = clist[pi];
            details::writeParticleDataToAsciiFile(dump, i, separator, data...);
            dump << std::endl;
        }
    }
    else
    {
        throw FileNotOpenedException("Can't open file at path: " + path);
    }

    dump.close();
}

template <typename... Args>
void writeParticleDataToAsciiFile(const std::vector<int> &clist, const std::string &path, const char separator, Args &&... data)
{
    writeParticleDataToAsciiFile(clist, path, false, separator, data...);
}

template <typename... Args>
void readParticleDataFromBinFile(const std::string &path, Args &&... data)
{
    std::ifstream inputfile(path, std::ios::binary);

    if (inputfile.is_open())
    {
        details::readParticleDataFromBinFile(inputfile, data...);
        inputfile.close();
    }
    else
    {
        throw FileNotOpenedException("Can't open file at path: " + path);
    }
}

} // namespace fileutils
} // namespace sphexa
