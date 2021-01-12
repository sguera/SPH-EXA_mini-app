#pragma once

#include <mpi.h>

#include "Exceptions.hpp"

namespace sphexa
{
namespace fileutils
{
namespace details
{
void readFileMPI(const MPI_File &, const size_t, const MPI_Offset &, const MPI_Offset &, const int) {}

template <typename Arg, typename... Args>
void readFileMPI(const MPI_File &file, const size_t count, const MPI_Offset &offset, const MPI_Offset &col, const int blockNo, Arg &first,
                 Args &&... args)
{
    MPI_Status status;

    MPI_File_set_view(file, blockNo * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_read(file, first.data(), count, MPI_DOUBLE, &status);

    readFileMPI(file, count, offset, col, blockNo + 1, args...);
}

void writeParticleDataToBinFileWithMPI(const MPI_File &, const size_t, const MPI_Offset &, const MPI_Offset &, const int) {}

template <typename Arg, typename... Args>
void writeParticleDataToBinFileWithMPI(const MPI_File &file, const size_t count, const MPI_Offset &offset, const MPI_Offset &col,
                                       const int blockNo, Arg &first, Args &&... args)
{
    MPI_Status status;

    MPI_File_set_view(file, blockNo * col + offset, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_write(file, first.data(), count, MPI_DOUBLE, &status);

    writeParticleDataToBinFileWithMPI(file, count, offset, col, blockNo + 1, args...);
}
} // namespace details

template <typename Dataset, typename... Args>
void writeParticleCheckpointDataToBinWithMPI(const Dataset &d, const std::string &path, Args &&... data)
{
    MPI_File file;
    MPI_Status status;

    const int err = MPI_File_open(d.comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);
    if (err != MPI_SUCCESS) { throw MPIFileNotOpenedException("Can't open MPI file at path: " + path, err); }

    const size_t split = d.n / d.nrank;
    const size_t remaining = d.n - d.nrank * split;

    const MPI_Offset col = d.n * sizeof(double);
    const MPI_Offset headerOffset = 2 * sizeof(double) + sizeof(size_t);
    MPI_Offset offset = headerOffset + d.rank * split * sizeof(double);

    if (d.rank > d.nrank - 1) offset += remaining * sizeof(double);
    if (d.rank == 0)
    {
        MPI_File_write(file, &d.n, 1, MPI_UNSIGNED_LONG, &status);
        MPI_File_write(file, &d.ttot, 1, MPI_DOUBLE, &status);
        MPI_File_write(file, &d.minDt, 1, MPI_DOUBLE, &status);
    }
    MPI_Barrier(d.comm);

    details::writeParticleDataToBinFileWithMPI(file, d.count, offset, col, 0, data...);

    MPI_File_close(&file);
}

template <typename Dataset, typename... Args>
void writeParticleDataToBinFileWithMPI(const Dataset &d, const std::string &path, Args &&... data)
{
    MPI_File file;

    const int err = MPI_File_open(d.comm, path.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &file);
    if (err != MPI_SUCCESS) { throw MPIFileNotOpenedException("Can't open MPI file at path: " + path, err); }

    const size_t split = d.n / d.nrank;
    const size_t remaining = d.n - d.nrank * split;

    const MPI_Offset col = d.n * sizeof(double);
    MPI_Offset offset = d.rank * split * sizeof(double);
    if (d.rank > d.nrank - 1) offset += remaining * sizeof(double);

    details::writeParticleDataToBinFileWithMPI(file, d.count, offset, col, 0, data...);

    MPI_File_close(&file);
}
template <typename Dataset, typename... Args>
void readParticleDataFromBinFileWithMPI(const std::string &path, Dataset &pd, Args &&... data)
{
    const size_t split = pd.n / pd.nrank;
    const size_t remaining = pd.n - pd.nrank * split;

    pd.count = pd.rank != pd.nrank - 1 ? split : split + remaining;
    pd.resize(pd.count);

    MPI_File fh;

    const MPI_Offset col = pd.n * sizeof(double);

    MPI_Offset offset = pd.rank * split * sizeof(double);
    if (pd.rank > pd.nrank - 1) offset += remaining * sizeof(double);

    const int err = MPI_File_open(pd.comm, path.c_str(), MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) { throw MPIFileNotOpenedException("Can't open MPI file at path: " + path, err); }

    details::readFileMPI(fh, pd.count, offset, col, 0, data...);

    MPI_File_close(&fh);
}

template <typename Dataset, typename... Args>
void readParticleCheckpointDataFromBinFileWithMPI(const std::string &path, Dataset &pd, Args &&... data)
{
    MPI_File fh;
    MPI_Status status;

    const int err = MPI_File_open(pd.comm, path.c_str(), MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS)
    {
        throw MPIFileNotOpenedException("Can't open MPI file at path: " + path, err);
    }

    MPI_File_read(fh, &pd.n, 1, MPI_UNSIGNED_LONG, &status);

    const MPI_Offset headerOffset = 2 * sizeof(double) + sizeof(size_t);

    const size_t split = pd.n / pd.nrank;
    const size_t remaining = pd.n - pd.nrank * split;
    const MPI_Offset col = pd.n * sizeof(double);

    MPI_Offset offset = headerOffset + pd.rank * split * sizeof(double);
    if (pd.rank > pd.nrank - 1) offset += remaining * sizeof(double);

    pd.count = pd.rank != pd.nrank - 1 ? split : split + remaining;
    pd.resize(pd.count);

    MPI_File_read(fh, &pd.ttot, 1, MPI_DOUBLE, &status);
    MPI_File_read(fh, &pd.minDt, 1, MPI_DOUBLE, &status);

    fileutils::details::readFileMPI(fh, pd.count, offset, col, 0, data...);

    MPI_File_close(&fh);
}

} // namespace fileutils
} // namespace sphexa
