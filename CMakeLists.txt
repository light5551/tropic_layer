#cmake_minimum_required(VERSION 3.9)
#project(tropic_layer LANGUAGES CXX CUDA)
#
#set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CXX_STANDARD 17)
#
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#set(Torch_DIR ~/libs/libtorch/share/cmake/Torch)
#list(APPEND CMAKE_MODULE_PATH ${Torch_DIR})
#
#find_package(Torch REQUIRED)
#
#add_executable(tropic_layer src/main.cu)
#
#set_target_properties(tropic_layer PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)

cmake_minimum_required(VERSION 3.9)
set(${PROJECT_NAME} tropics)
project(${PROJECT_NAME}  LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CXX_COMPILER /usr/bin/g++11)


#list(APPEND CMAKE_PREFIX_PATH /home/light5551/.local/lib/python3.10/site-packages/torch/share/cmake)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#target_link_libraries(${PROJECT_NAME} PRIVATE Torch)
add_executable(${PROJECT_NAME} src/main.cu)
#add_executable(${PROJECT_NAME} src/torch.cpp)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA 17)

#target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
