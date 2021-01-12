
cmake_minimum_required(VERSION 2.8.11)
project(SphUnitTests CXX)

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

add_executable(run_mpi_tests test_main.cpp)
target_include_directories(run_mpi_tests PRIVATE ../../src)
target_include_directories(run_mpi_tests PRIVATE ../../src/include)
target_link_libraries(run_mpi_tests gtest_main)
add_test(NAME run_mpi_tests COMMAND run_mpi_tests.exe)

add_executable(run_sync_test test_sync.cpp)
target_include_directories(run_sync_test PRIVATE ../../src)
target_include_directories(run_sync_test PRIVATE ../../src/include)
target_link_libraries(run_sync_test gtest_main)
add_test(NAME run_sync_test COMMAND run_mpi_tests.exe)

