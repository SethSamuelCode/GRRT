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
    src/metric.cpp
    src/opacity.cpp
    src/noise.cpp
    src/volumetric_disk.cpp
    src/fits_writer.cpp
)
target_include_directories(grrt PUBLIC include/ ${CMAKE_BINARY_DIR}/include PRIVATE third_party/)
target_compile_definitions(grrt PRIVATE GRRT_BUILDING_DLL)

# Generate export header
include(GenerateExportHeader)
generate_export_header(grrt
    EXPORT_FILE_NAME include/grrt_export.h
    EXPORT_MACRO_NAME GRRT_EXPORT
)

# Aggressive optimisation (C++ only, not CUDA)
if(MSVC)
    target_compile_options(grrt PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
        /arch:AVX2 /fp:fast   # SIMD: AVX2 + FMA + relaxed float
        /Oi                    # Replace calls with CPU intrinsics
        /Ot                    # Favour speed over code size
        /Ob3                   # Aggressive inlining (VS 2022+)
        /GL                    # Whole-program optimisation (emit IR)
    >)
    target_link_options(grrt PRIVATE $<$<CONFIG:Release>:/LTCG>)   # Link-time code generation
else()
    target_compile_options(grrt PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
        -mavx2 -mfma -ffast-math        # SIMD + relaxed float
        -march=native                    # Tune for this exact CPU
        -flto                            # Link-time optimisation
        -funroll-loops                   # Unroll small loops
        -fno-semantic-interposition      # Allow inlining/devirt across DSO TUs
        -fno-plt                         # Eliminate PLT trampolines for external calls
        -falign-functions=32             # Align hot functions to 32-byte boundaries
        -falign-loops=32                 # Align hot loops to 32-byte boundaries
        -fprefetch-loop-arrays           # Software prefetch for array traversals
        -fdata-sections                  # Dead-code stripping: one section per datum
        -ffunction-sections              # Dead-code stripping: one section per function
    >)
    target_link_options(grrt PRIVATE
        $<$<CONFIG:Release>:-flto>
        -Wl,--gc-sections                # Strip unreferenced sections
    )
endif()

# OpenMP (optional)
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(grrt PRIVATE OpenMP::OpenMP_CXX)
endif()

if(GRRT_ENABLE_CUDA)
    target_sources(grrt PRIVATE
        cuda/cuda_backend.cu
        cuda/cuda_render.cu
        cuda/cuda_vol_host_data.cpp
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
if(MSVC)
    target_compile_options(grrt-cli PRIVATE /GL)
    target_link_options(grrt-cli PRIVATE $<$<CONFIG:Release>:/LTCG>)
else()
    target_compile_options(grrt-cli PRIVATE
        -mavx2 -mfma -ffast-math
        -march=native
        -flto
        -funroll-loops
        -fno-semantic-interposition
        -fno-plt
        -falign-functions=32
        -falign-loops=32
        -fprefetch-loop-arrays
        -fdata-sections
        -ffunction-sections
    )
    target_link_options(grrt-cli PRIVATE
        $<$<CONFIG:Release>:-flto>
        -Wl,--gc-sections
    )
endif()

# Test executables
add_executable(test-opacity tests/test_opacity.cpp)
target_link_libraries(test-opacity PRIVATE grrt)

add_executable(test-volumetric tests/test_volumetric.cpp)
target_link_libraries(test-volumetric PRIVATE grrt)

add_executable(test-spectral tests/test_spectral.cpp)
target_link_libraries(test-spectral PRIVATE grrt)

if(GRRT_ENABLE_CUDA)
    add_executable(grrt-cuda-test cuda/tests/test_cuda.cu)
    target_include_directories(grrt-cuda-test PRIVATE include/ cuda/ ${CMAKE_BINARY_DIR}/include)
    target_link_libraries(grrt-cuda-test PRIVATE CUDA::cudart)
    set_target_properties(grrt-cuda-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
