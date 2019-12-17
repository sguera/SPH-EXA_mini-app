cmake_minimum_required(VERSION 3.14)

project(sphexa CXX)

find_package(HPX REQUIRED)
link_directories(${HPX_LIBRARY_DIR})

set(CMAKE_CXX_STANDARD 14)

add_executable(stub stub.cpp)
hpx_setup_target(stub COMPONENT_DEPENDENCIES iostreams)


include_directories(${PROJECT_SOURCE_DIR}/../../src ${PROJECT_SOURCE_DIR}/../../src/include
                    ${PROJECT_SOURCE_DIR}/../../src/include/sph)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -mtune=native -DNDEBUG")
endif()
message(STATUS "build type: " ${CMAKE_BUILD_TYPE})

