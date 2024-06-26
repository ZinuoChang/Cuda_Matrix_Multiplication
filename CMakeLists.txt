cmake_minimum_required(VERSION 3.10)

# Set the policy CMP0104 to NEW
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# Set the CUDA architectures you want to target
set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80")

project(MatrixVectorMul LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++14)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_definitions(-DEIGEN_DONT_VECTORIZE)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler -Wno-deprecated-declarations)

add_executable(MatrixVectorMul src/main.cpp src/matrix.cu)
# target_link_libraries(MatrixVectorMul ${CUDA_LIBRARIES})