#pragma once

#include <iostream>

void debug()
{
    // compiler version:
    #ifdef _CRAYC
    //#define CURRENT_PE_ENV "CRAY"
    std::cout << "compiler: CCE/" << _RELEASE << "." << _RELEASE_MINOR << std::endl;
    #endif

    //std::cout << "compiler: GNU/" << <<  << std::endl;

    #ifdef __GNUC__
    //#define CURRENT_PE_ENV "GNU"
    std::cout << "compiler: GNU/" << __GNUC__ << "." << __GNUC_MINOR__
        << "." << __GNUC_PATCHLEVEL__
        << std::endl;
    #endif

    #ifdef __INTEL_COMPILER
    //#define CURRENT_PE_ENV "INTEL"
    std::cout << "compiler: INTEL/" << __INTEL_COMPILER << std::endl;
    #endif

    #ifdef __PGI
    //#define CURRENT_PE_ENV "PGI"
    std::cout << "compiler: PGI/" << __PGIC__
         << "." << __PGIC_MINOR__
         << "." << __PGIC_PATCHLEVEL__
         << std::endl;
    #endif
}

