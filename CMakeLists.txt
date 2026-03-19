cmake_minimum_required(VERSION 3.20)
project(gr-raytracer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GRRT_ENABLE_CUDA "Enable CUDA GPU backend" OFF)

if(GRRT_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # RTX 2080 = Turing = SM 7.5
    find_package(CUDAToolkit REQUIRED)
endif()

# Shared library
add_library(grrt SHARED
    src/api.cpp
    src/schwarzschild.cpp
    src/rk4.cpp
    src/geodesic_tracer.cpp
    src/camera.cpp
    src/renderer.cpp
    src/spectrum.cpp
    src/tonemapper.cpp
    src/accretion_disk.cpp
    src/celestial_sphere.cpp
    src/kerr.cpp
)
target_include_directories(grrt PUBLIC include/ ${CMAKE_BINARY_DIR}/include PRIVATE third_party/)
target_compile_definitions(grrt PRIVATE GRRT_BUILDING_DLL)

# Generate export header
include(GenerateExportHeader)
generate_export_header(grrt
    EXPORT_FILE_NAME include/grrt_export.h
    EXPORT_MACRO_NAME GRRT_EXPORT
)

# OpenMP (optional)
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(grrt PRIVATE OpenMP::OpenMP_CXX)
endif()

if(GRRT_ENABLE_CUDA)
    target_sources(grrt PRIVATE
        cuda/cuda_backend.cu
        cuda/cuda_render.cu
    )
    target_include_directories(grrt PRIVATE cuda/)
    target_link_libraries(grrt PRIVATE CUDA::cudart)
    target_compile_definitions(grrt PRIVATE GRRT_HAS_CUDA)
    target_compile_options(grrt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v -lineinfo>)
    set_property(TARGET grrt PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# CLI tool
add_executable(grrt-cli cli/main.cpp)
target_link_libraries(grrt-cli PRIVATE grrt)
target_include_directories(grrt-cli PRIVATE third_party/)

if(GRRT_ENABLE_CUDA)
    add_executable(grrt-cuda-test cuda/tests/test_cuda.cu)
    target_include_directories(grrt-cuda-test PRIVATE include/ cuda/ ${CMAKE_BINARY_DIR}/include)
    target_link_libraries(grrt-cuda-test PRIVATE CUDA::cudart)
    set_target_properties(grrt-cuda-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
