cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(tropics LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable(${PROJECT_NAME} src/main.cu)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)