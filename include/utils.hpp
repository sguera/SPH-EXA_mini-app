#pragma once

namespace sphexa
{

int initAndGetRankId()
{
    int rank = 0;
#ifdef USE_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    return rank;
}

int exitSuccess()
{
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return EXIT_SUCCESS;
}

} // namespace sphexa
