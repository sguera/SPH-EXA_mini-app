
set(UNIT_TESTS
        boxoverlap.cpp
        btree.cpp
        btreetraversal.cpp
        collisions_cpu.cpp
        clz.cpp
        domaindecomp.cpp
        disjointoctree.cpp
        findneighbors.cpp
        halodiscovery.cpp
        haloneighbors.cpp
        histogram.cpp
        layout.cpp
        morton.cpp
        octree.cpp
        octree_internal.cpp
        octree_util.cpp
        zorder.cpp
        test_main.cpp)

add_executable(component_units ${UNIT_TESTS})

target_include_directories(component_units PRIVATE ../../include)
target_include_directories(component_units PRIVATE ../)
target_link_libraries(component_units gtest_main)
add_test(NAME ComponentUnits COMMAND component_units)

add_executable(component_units_omp ${UNIT_TESTS})

target_include_directories(component_units_omp PRIVATE ../../include)
target_include_directories(component_units_omp PRIVATE ../)
target_link_libraries(component_units_omp PRIVATE gtest_main)
target_link_libraries(component_units_omp PUBLIC OpenMP::OpenMP_CXX)
add_test(NAME ComponentUnitsOmp COMMAND component_units_omp)
