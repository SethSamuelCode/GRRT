cmake_minimum_required(VERSION 3.20)
project(gr-raytracer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(grrt PRIVATE OpenMP::OpenMP_CXX)
endif()

# CLI tool
add_executable(grrt-cli cli/main.cpp)
target_link_libraries(grrt-cli PRIVATE grrt)
target_include_directories(grrt-cli PRIVATE third_party/)
