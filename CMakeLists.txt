cmake_minimum_required(VERSION 3.22)

set(CUDAToolkit_ROOT /usr/local/cuda/)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(FindCUDAToolkit REQUIRED)
#set(FindCUDAToolkit REQUIRED)

#set(CMAKE_CUDA_COMPILER nvcc)

project(cuda_hashing CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 86)


add_executable(cuda_hashing MRC.cu cuda_consts.cuh)
#add_executable(cuda_hashing main.cu tasks.cu subroutines.cu WjCryptLib_Md5.cu
#        tasks.cuh subroutines.cuh WjCryptLib_Md5.cuh cuda_consts.cuh)

set_target_properties(cuda_hashing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
