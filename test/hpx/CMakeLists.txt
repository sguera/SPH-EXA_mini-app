cmake_minimum_required(VERSION 3.14)

project(sphexa CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -mtune=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")
# need to hack arch flag into CUDA flags, because CMake doesn't propagate
# any flags to the device link step, but arch flags are crucial for device linking
# see issue https://gitlab.kitware.com/cmake/cmake/issues/18265
list(APPEND CMAKE_CUDA_FLAGS "-arch=sm_60")

find_package(HPX REQUIRED)
link_directories(${HPX_LIBRARY_DIR})


add_executable(stub stub.cpp)
hpx_setup_target(stub COMPONENT_DEPENDENCIES iostreams)


include_directories(${PROJECT_SOURCE_DIR}/../../src ${PROJECT_SOURCE_DIR}/../../src/include
                    ${PROJECT_SOURCE_DIR}/../../src/include/sph)



set(CUDASRC ../../src/include/sph/cuda)
add_library(numeric_gpu STATIC ${CUDASRC}/cudaIAD.cu ${CUDASRC}/cudaMomentumAndEnergyIAD.cu ${CUDASRC}/cudaDensity.cu ${CUDASRC}/cudaLookupTables.cu)

target_compile_definitions(numeric_gpu PUBLIC USE_CUDA)
target_compile_options(numeric_gpu PRIVATE --expt-relaxed-constexpr)
target_compile_features(numeric_gpu PUBLIC cxx_std_14)
set_target_properties(numeric_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# *** Install
#install(TARGETS numeric_gpu EXPORT DMRGTargets COMPONENT libraries DESTINATION lib)
#export(TARGETS numeric_gpu APPEND FILE "${PROJECT_BINARY_DIR}/DMRGTargets.cmake")


#add_executable(main main.cpp)

#target_compile_definitions(main PUBLIC USE_CUDA)
#set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(main PUBLIC numeric_gpu)
#hpx_setup_target(main COMPONENT_DEPENDENCIES iostreams)

add_executable(sqpatch ${PROJECT_SOURCE_DIR}/../../src/sqpatch.cpp)

target_compile_definitions(sqpatch PUBLIC USE_CUDA)
set_target_properties(sqpatch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(sqpatch PUBLIC numeric_gpu)


#if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
#  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -mtune=native -DNDEBUG")
#endif()
#message(STATUS "build type: " ${CMAKE_BUILD_TYPE})
