cmake_minimum_required(VERSION 3.14)

project(sphexa CXX)

find_package(HPX REQUIRED)
link_directories(${HPX_LIBRARY_DIR})

set(CMAKE_CXX_STANDARD 14)

add_executable(sqpatch src/sqpatch.cpp)
hpx_setup_target(sqpatch COMPONENT_DEPENDENCIES iostreams)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    target_link_libraries(sqpatch ${OpenMP_CXX_LIBRARIES})
endif()

include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/include
                    ${PROJECT_SOURCE_DIR}/src/include/sph)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -mtune=native -DNDEBUG")
endif()
message(STATUS "build type: " ${CMAKE_BUILD_TYPE})
