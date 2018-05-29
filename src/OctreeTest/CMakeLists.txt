project(octree)
cmake_minimum_required(VERSION 2.8)

# Release by default
# Turn on Debug with "-DCMAKE_BUILD_TYPE=Debug"
if( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release )
endif()

find_package(Boost)

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  ADD_EXECUTABLE(main Octree.cpp)
endif()