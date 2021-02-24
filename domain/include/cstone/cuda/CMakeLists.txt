
if(CMAKE_CUDA_COMPILER)
    add_library(cuda_find_neighbors_obj OBJECT findneighbors.cu)
    target_include_directories(cuda_find_neighbors_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)

    add_library(gather_obj OBJECT gather.cu)
endif()
