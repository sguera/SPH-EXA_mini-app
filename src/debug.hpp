#pragma once

#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

void debug()
{
    #ifdef USE_MPI
    int mpirank, mpisize, mpiversion, mpisubversion;
    int resultlen = -1;
    char mpilibversion[MPI_MAX_LIBRARY_VERSION_STRING];
    mpirank = MPI::COMM_WORLD.Get_rank();
    mpisize = MPI::COMM_WORLD.Get_size();
    if (mpirank == 0) {
        MPI_Get_version( &mpiversion, &mpisubversion );
        MPI_Get_library_version(mpilibversion, &resultlen);
        std::cout << "# MPI: version/" << mpiversion << mpisubversion 
            << " " << mpisize << " rank(s)" << std::endl;
        printf("# MPI: %.*s\n", 42, mpilibversion);
    }
    #endif

    #ifdef _OPENMP
    printf("# OPENMP: version/%u %d thread(s)\n", _OPENMP, omp_get_max_threads());
    #endif

    #ifdef USE_MPI
    if (mpirank == 0) {
    #endif

    // compiler version:
    #ifdef _CRAYC
    //#define CURRENT_PE_ENV "CRAY"
    std::cout << "# COMPILER: CCE/" << _RELEASE << "." << _RELEASE_MINOR << std::endl;
    #endif

    //std::cout << "COMPILER: GNU/" << <<  << std::endl;

    #ifdef __GNUC__
    //#define CURRENT_PE_ENV "GNU"
    std::cout << "# COMPILER: GNU/" << __GNUC__ << "." << __GNUC_MINOR__
        << "." << __GNUC_PATCHLEVEL__
        << std::endl;
    #endif

    #ifdef __INTEL_COMPILER
    //#define CURRENT_PE_ENV "INTEL"
    std::cout << "# COMPILER: INTEL/" << __INTEL_COMPILER << std::endl;
    #endif

    #ifdef __PGI
    //#define CURRENT_PE_ENV "PGI"
    std::cout << "# COMPILER: PGI/" << __PGIC__
         << "." << __PGIC_MINOR__
         << "." << __PGIC_PATCHLEVEL__
         << std::endl;
    #endif

    #ifdef USE_MPI
    }
    #endif
}

