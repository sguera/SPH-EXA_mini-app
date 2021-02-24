

if(CMAKE_CUDA_COMPILER)
    add_executable(cudaNeighborsTest $<TARGET_OBJECTS:cuda_find_neighbors_obj> neighbor_driver.cpp)
    target_include_directories(cudaNeighborsTest PRIVATE ../../include)
    target_include_directories(cudaNeighborsTest PRIVATE ../)

    target_link_libraries(cudaNeighborsTest PUBLIC CUDA::cudart OpenMP::OpenMP_CXX)
endif()

add_executable(octree_perf octree.cpp)

target_include_directories(octree_perf PRIVATE ../../include)
target_include_directories(octree_perf PRIVATE ../)
target_link_libraries(octree_perf PUBLIC OpenMP::OpenMP_CXX)


add_executable(scan_perf scan.cpp)

target_include_directories(scan_perf PRIVATE ../../include)
target_include_directories(scan_perf PRIVATE ../)
target_link_libraries(scan_perf PUBLIC OpenMP::OpenMP_CXX)