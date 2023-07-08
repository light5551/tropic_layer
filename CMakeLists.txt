cmake_minimum_required(VERSION 3.21)
project(tropic_layer CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(tropic_layer src/main.cu)

set_target_properties(tropic_layer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
