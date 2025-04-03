cmake_minimum_required(VERSION 3.10)
project(cuda)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)

include_directories(${SRC_DIR})


aux_source_directory(${SRC_DIR} SRCS)

add_executable(main ${SRCS})