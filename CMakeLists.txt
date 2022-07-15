cmake_minimum_required(VERSION 3.22)
project(cuda_hashing CUDA)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY /home/shford/CLionProjects/cuda_hashing/cmake-build-debug/)
set(CUDAToolkit_ROOT /usr/lib/nvidia-cuda-toolkit/)
set(CMAKE_CUDA_STANDARD 14)

#add_executable(cuda_hashing MRC.cu)
add_executable(cuda_hashing main.cu tasks.cu subroutines.cu WjCryptLib_Md5.cu
        tasks.cuh subroutines.cuh WjCryptLib_Md5.cuh cuda_consts.cuh)

set_target_properties(cuda_hashing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
