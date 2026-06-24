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
    src/disk_column_bvp.cpp
    src/slim_disk_radial.cpp
    src/fits_writer.cpp
    src/romberg_step.cpp
    src/sobol_sampler.cpp
    src/disk_step_entry.cpp
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
        /arch:AVX2 /fp:precise # SIMD: AVX2; precise (deterministic) FP for the correctness phase — restore /fp:fast once the coupling lands
        /Oi                    # Replace calls with CPU intrinsics
        /Ot                    # Favour speed over code size
        /Ob3                   # Aggressive inlining (VS 2022+)
        /GL                    # Whole-program optimisation (emit IR)
    >)
    target_link_options(grrt PRIVATE $<$<CONFIG:Release>:/LTCG>)   # Link-time code generation
else()
    target_compile_options(grrt PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
        -mavx2 -mfma -fno-fast-math -ffp-contract=off  # SIMD; precise (deterministic) FP for the correctness phase — restore -ffast-math once the coupling lands
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
    # GCC-only flags (Clang does not support these)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(grrt PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -fipa-pta                        # Interprocedural pointer analysis
            -fdevirtualize-at-ltrans         # Extra devirtualization pass at LTO link time
            -fvect-cost-model=unlimited      # Force vectorization regardless of cost model
        >)
    endif()
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
    # GCC-only flags (Clang does not support these)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(grrt-cli PRIVATE
            -fipa-pta
            -fdevirtualize-at-ltrans
            -fvect-cost-model=unlimited
        )
    endif()
    target_link_options(grrt-cli PRIVATE
        $<$<CONFIG:Release>:-flto>
        -Wl,--gc-sections
    )
endif()

# Test executables
add_executable(test-opacity tests/test_opacity.cpp)
target_link_libraries(test-opacity PRIVATE grrt)

add_executable(test-opacity-rosseland tests/test_opacity_rosseland.cpp)
target_link_libraries(test-opacity-rosseland PRIVATE grrt)

add_executable(test-volumetric tests/test_volumetric.cpp)
target_link_libraries(test-volumetric PRIVATE grrt)

add_executable(test-spectral tests/test_spectral.cpp)
target_link_libraries(test-spectral PRIVATE grrt)

add_executable(test-romberg-step tests/test_romberg_step.cpp)
target_link_libraries(test-romberg-step PRIVATE grrt)

add_executable(test-sobol-sampler tests/test_sobol_sampler.cpp)
target_link_libraries(test-sobol-sampler PRIVATE grrt)

add_executable(test-disk-step-entry tests/test_disk_step_entry.cpp)
target_link_libraries(test-disk-step-entry PRIVATE grrt)

add_executable(test-raymarch-step-control tests/test_raymarch_step_control.cpp)
target_link_libraries(test-raymarch-step-control PRIVATE grrt)

add_executable(test-column-bvp tests/test_column_bvp.cpp)
target_link_libraries(test-column-bvp PRIVATE grrt)

add_executable(test-slim-disk-radial tests/test_slim_disk_radial.cpp)
target_link_libraries(test-slim-disk-radial PRIVATE grrt)

# Analytic-Jacobian vs FD cross-check (the permanent correctness oracle for the
# analytic slim-disk Jacobian).  #includes slim_disk_radial.cpp + opacity.cpp
# directly to reach the anonymous-namespace helpers (slim_numerical_jacobian /
# slim_analytic_jacobian), the same pattern as the slim-* probe tools; does NOT
# link grrt to avoid duplicate symbols.
add_executable(test-slim-jacobian tests/test_slim_jacobian.cpp)
target_include_directories(test-slim-jacobian PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# TEMPORARY DIAGNOSTIC PROBE (slim-disk basin-vs-noise verdict). Includes the
# slim/opacity .cpp directly to reach internal helpers; does NOT link grrt to
# avoid duplicate symbols. Safe to delete along with tools/slim_diag_probe.cpp.
add_executable(slim-diag-probe tools/slim_diag_probe.cpp)
target_include_directories(slim-diag-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# TEMPORARY DIAGNOSTIC PROBE (spin-walk first-rung stall: warm-start re-projection
# vs a>0 inner-solve basin). Same include-the-cpp pattern as slim-diag-probe.
# Safe to delete along with tools/slim_spinwalk_probe.cpp.
add_executable(slim-spinwalk-probe tools/slim_spinwalk_probe.cpp)
target_include_directories(slim-spinwalk-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# TEMPORARY DIAGNOSTIC PROBE (cold-seed spin sweep: at what spin does a fresh
# cold thin-disk seed stop converging?). Same include-the-cpp pattern.
# Safe to delete along with tools/slim_coldseed_sweep.cpp.
add_executable(slim-coldseed-sweep tools/slim_coldseed_sweep.cpp)
target_include_directories(slim-coldseed-sweep PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# TEMPORARY DIAGNOSTIC PROBE (warm-start Mdot-continuation sweep: does carrying U
# warm across a fine f_Edd ladder cross the Eddington regularity wall where the
# cold seed fails?). Same include-the-cpp pattern.
# Safe to delete along with tools/slim_warmstart_sweep.cpp.
add_executable(slim-warmstart-sweep tools/slim_warmstart_sweep.cpp)
target_include_directories(slim-warmstart-sweep PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# Pseudo-arclength continuation probe (Tasks 1-4): traces the slim-disk branch
# THROUGH the f_Edd≈0.11 fold.  Includes the slim/opacity .cpp directly to reach
# internal helpers (slim_arclength_tangent, arclength_corrector); does NOT link grrt.
# Safe to delete along with tools/slim_arclength_probe.cpp.
add_executable(slim-arclength-probe tools/slim_arclength_probe.cpp)
target_include_directories(slim-arclength-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# TEMPORARY slim-branch SEED probe: tests whether a thick/advective/high-Σ upper-
# branch seed converges directly (or by continuation) at high f_Edd, sidestepping
# the f_Edd≈0.11 fold.  Includes the slim/opacity .cpp directly (probe pattern) to
# reach internal helpers.  Safe to delete along with tools/slim_slimseed_probe.cpp.
add_executable(slim-slimseed-probe tools/slim_slimseed_probe.cpp)
target_include_directories(slim-slimseed-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# TEMPORARY slim-branch VALIDATION probe: reproduces + dumps the near-Eddington
# slim solution (soundness), sweeps the slim-seed basin (robustness), and maps the
# upper branch across f_Edd / spin.  Includes the slim/opacity .cpp directly (probe
# pattern).  Safe to delete along with tools/slim_slimseed_validate.cpp.
add_executable(slim-slimseed-validate tools/slim_slimseed_validate.cpp)
target_include_directories(slim-slimseed-validate PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# TEMPORARY benchmark probe: benchmarks the near-Eddington slim solution against
# ground truth (NT reduction at low f_Edd, slim-theory expectations at f_Edd=0.9,
# f_adv sign check, resolution test, thick-inner/thin-outer seed).  Includes the
# slim/opacity .cpp directly (probe pattern).  Safe to delete along with
# tools/slim_benchmark_probe.cpp.
add_executable(slim-benchmark-probe tools/slim_benchmark_probe.cpp)
target_include_directories(slim-benchmark-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# TEMPORARY NT-term probe: evaluates the slim §23 energy terms (Q_vis/Q_rad/Q_adv)
# on the EXACT Novikov-Thorne state (Page-Thorne flux + alpha-relation Sigma + grey
# midplane T_c) to pinpoint which term breaks the slim->NT reduction at large r.
# No solver runs (pure term evaluation).  Includes the slim/opacity .cpp directly
# (probe pattern).  Safe to delete along with tools/slim_nt_term_probe.cpp.
add_executable(slim-nt-term-probe tools/slim_nt_term_probe.cpp)
target_include_directories(slim-nt-term-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# Q_adv entropy-form validation: on a CONVERGED profile, evaluates the §23
# advective-cooling bracket (flag #1) three independent ways — the solver's frozen
# η₃=3/2 bracket vs the raw 2D entropy identity (implementation check) vs the true
# variable η₃(β) (measures refinement #11).  Closes the Q_adv gate-blindness.
# Includes the slim/opacity .cpp directly.  Safe to delete with the .cpp.
add_executable(slim-qadv-entropy-probe tools/slim_qadv_entropy_probe.cpp)
target_include_directories(slim-qadv-entropy-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# TEMPORARY full-target probe (a=0.998, f_Edd≈0.9). Links grrt. Safe to delete
# along with tools/slim_full_target_probe.cpp.
add_executable(slim-full-target-probe tools/slim_full_target_probe.cpp)
target_link_libraries(slim-full-target-probe PRIVATE grrt)

# TEMPORARY safety-budget smoke check. Safe to delete along with
# tools/slim_budget_smoke.cpp.
add_executable(slim-budget-smoke tools/slim_budget_smoke.cpp)
target_link_libraries(slim-budget-smoke PRIVATE grrt)

# Diagnostic tool: dump the volumetric-disk LUTs to CSV (banding investigation).
add_executable(dump-disk-lut tools/dump_disk_lut.cpp)
target_link_libraries(dump-disk-lut PRIVATE grrt)

if(GRRT_ENABLE_CUDA)
    add_executable(grrt-cuda-test cuda/tests/test_cuda.cu)
    target_include_directories(grrt-cuda-test PRIVATE include/ cuda/ ${CMAKE_BINARY_DIR}/include)
    target_link_libraries(grrt-cuda-test PRIVATE CUDA::cudart)
    set_target_properties(grrt-cuda-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# TEMPORARY single-entry Jacobian FD step-scan (classifies the post-Qvis-fix
# marginal FD-gate failure).  Safe to delete along with tools/slim_jacentry_scan.cpp.
add_executable(slim-jacentry-scan tools/slim_jacentry_scan.cpp)
target_include_directories(slim-jacentry-scan PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# TEMPORARY near-Eddington profile dump + mesh check (post-Qvis-fix re-test).
# Safe to delete along with tools/slim_edd09_dump.cpp.
add_executable(slim-edd09-dump tools/slim_edd09_dump.cpp)
target_include_directories(slim-edd09-dump PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# TEMPORARY Sądowski-structure residual probe (f_Edd≈0.9 closure diagnostic, Part 2):
# feed the Sądowski-shape slim seed to OUR residual and read the group magnitudes
# (small => reachable root; large => closure inadequacy).  Includes the slim/opacity
# .cpp directly.  Safe to delete along with tools/slim_sadowski_residual_probe.cpp.
add_executable(slim-sadowski-residual-probe tools/slim_sadowski_residual_probe.cpp)
target_include_directories(slim-sadowski-residual-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)
