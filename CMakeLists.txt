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

# C1 coupled-column re-pose (Σ+T_c-driven). Include-the-.cpp probe: #includes
# opacity.cpp + disk_column_bvp.cpp + disk_column_coupled.cpp in one TU to reach
# the column's TU-local helpers (node_deriv, column_residual, analytic_jacobian);
# does NOT link grrt to avoid duplicate symbols.
add_executable(test-column-coupled tests/test_column_coupled.cpp)
target_include_directories(test-column-coupled PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

add_executable(test-slim-disk-radial tests/test_slim_disk_radial.cpp)
target_link_libraries(test-slim-disk-radial PRIVATE grrt)

# Analytic-Jacobian vs FD cross-check (the permanent correctness oracle for the
# analytic slim-disk Jacobian).  #includes slim_disk_radial.cpp + opacity.cpp
# directly to reach the anonymous-namespace helpers (slim_numerical_jacobian /
# slim_analytic_jacobian), the same pattern as the slim-* probe tools; does NOT
# link grrt to avoid duplicate symbols.
add_executable(test-slim-jacobian tests/test_slim_jacobian.cpp)
target_include_directories(test-slim-jacobian PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# Task 10 GATE: analytic reduced/Schur Jacobian of the COUPLED slim-disk driver
# (slim_coupled_reduced_jacobian) vs the perturb-resolve FD oracle, at a feasible
# synthetic operating point.  Validates the C3 column-mediated Σ,T_c Schur block was
# assembled correctly into the radial Jacobian (per-column scaled 2-norm < 1e-3).
# Include-the-.cpp: opacity + column-bvp + column-coupled + slim-radial + slim-coupled
# in one TU (reaches slim_disk_coupled.cpp's TU-local helpers); does NOT link grrt.
add_executable(test-slim-coupled-jacobian tests/test_slim_coupled_jacobian.cpp)
target_include_directories(test-slim-coupled-jacobian PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

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

# f_adv FREEDOM probe: tests whether the Sigma-folds are artifacts of holding f_adv
# fixed. Solves eval(T_eff,f_adv)=(T_c,scale*Sigma) via the BASE column solver as a
# black box (no new column physics). Safe to delete with tools/slim_fadv_freedom_probe.cpp.
add_executable(slim-fadv-freedom-probe tools/slim_fadv_freedom_probe.cpp)
target_include_directories(slim-fadv-freedom-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# η₄ MOMENT probe: gates column_moments' η₄ = (1/Σ)∫ρz²dz (S11 density 2nd
# moment) against the one-zone reductions (uniform→h²/3, Gaussian→H²) plus an
# inline reference quadrature. Includes the slim/opacity .cpp directly (probe
# pattern). Safe to delete along with tools/slim_eta4_moment_probe.cpp.
add_executable(slim-eta4-moment-probe tools/slim_eta4_moment_probe.cpp)
target_include_directories(slim-eta4-moment-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

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

# C4 coupled-column slim-disk SMOKE probe: runs solve_slim_disk_coupled at a modest
# operating point (a=0.9, f_Edd=0.02, small N) and on a degenerate input (mdot=0).
# Include-the-.cpp consumer pattern: #includes opacity.cpp + disk_column_bvp.cpp +
# disk_column_coupled.cpp + slim_disk_radial.cpp + slim_disk_coupled.cpp in ONE TU
# (slim_disk_coupled.cpp reaches the radial solver's TU-local helpers this way); does
# NOT link grrt to avoid duplicate symbols.  Safe to delete along with
# tools/slim_coupled_smoke_probe.cpp (and src/slim_disk_coupled.cpp is consumer-only,
# never added to the grrt library).
add_executable(slim-coupled-smoke-probe tools/slim_coupled_smoke_probe.cpp)
target_include_directories(slim-coupled-smoke-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# Task 11 coupled-column -> Novikov-Thorne REDUCTION gate: drives solve_column_coupled
# at the EXACT NT state (a=0.9, f_Edd=0.02; Page-Thorne flux + α-relation Σ + grey T_c,
# reusing slim_nt_term_probe's NT-state math) and checks the emergent one-face column
# flux reduces to the one-zone NT diffusion flux 32σT_c⁴/(3κΣ) with the O(1) vertical-
# structure factor f_F≈0.94 (Sądowski 2011 Eq 45), gating only on the radii where the
# column converges and is gas-dominated/optically-thick.  Include-the-.cpp consumer
# pattern: opacity.cpp + disk_column_bvp.cpp + disk_column_coupled.cpp +
# slim_disk_radial.cpp in ONE TU (the coupled column needs the radial solver's TU-local
# machinery); does NOT link grrt to avoid duplicate symbols.  Safe to delete along with
# tools/slim_coupled_nt_probe.cpp.
add_executable(slim-coupled-nt-probe tools/slim_coupled_nt_probe.cpp)
target_include_directories(slim-coupled-nt-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# Task 12 coupled-column f_Edd CONTINUATION WALK probe (THE DECISIVE EXPERIMENT):
# walk the coupled (column-driven) slim disk via f_Edd continuation from a feasible
# low-f_Edd cold-seed base UP toward the POC target f_Edd≈0.9 (a=0.9), warm-starting
# each rung from the previous converged radial state; reports REACHED-0.9 (physical?)
# / STALLED-at-f_Edd_max (column/radial/fold) / BASE-INFEASIBLE — the OUTCOME is the
# deliverable (honest, exploratory).  Include-the-.cpp consumer pattern: opacity.cpp +
# disk_column_bvp.cpp + disk_column_coupled.cpp + slim_disk_radial.cpp +
# slim_disk_coupled.cpp in ONE TU (reaches relax_coupled + the radial solver's TU-local
# machinery); does NOT link grrt to avoid duplicate symbols.  Safe to delete along with
# tools/slim_coupled_walk_probe.cpp.
add_executable(slim-coupled-walk-probe tools/slim_coupled_walk_probe.cpp)
target_include_directories(slim-coupled-walk-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)
# The per-node coupled column solves in slim_disk_coupled.cpp are OpenMP-parallelized;
# this include-the-.cpp probe needs /openmp on ITS OWN target (it does NOT link grrt, so
# it does not inherit the library's OpenMP link).  MSVC: OpenMP::OpenMP_CXX adds /openmp.
if(OpenMP_CXX_FOUND)
    target_link_libraries(slim-coupled-walk-probe PRIVATE OpenMP::OpenMP_CXX)
endif()

# OpenMP parallel==serial CORRECTNESS GATE for the coupled per-node column solves.
# Builds the coupled residual R and reduced Jacobian J at a feasible (manifold-calibrated)
# base point once SERIAL (1 thread) and once PARALLEL (max threads) and asserts they match
# to FP tolerance AND are deterministic (two parallel builds agree); then times a Jacobian
# build serial vs parallel.  Same include-the-.cpp consumer order as the walk probe; needs
# /openmp on THIS target.  Safe to delete along with tools/slim_omp_gate_probe.cpp.
add_executable(slim-omp-gate-probe tools/slim_omp_gate_probe.cpp)
target_include_directories(slim-omp-gate-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)
if(OpenMP_CXX_FOUND)
    target_link_libraries(slim-omp-gate-probe PRIVATE OpenMP::OpenMP_CXX)
endif()

# S-CURVE / MULTI-ROOT probe (DELETABLE, READ-ONLY diagnostic): at a node's exact
# reconstructed ColumnCoupledInputs (mirroring eval_node_coupled) from a live coupled
# checkpoint, scans the base column's Sigma0(T_eff) to expose EVERY distinct converged
# branch at the SAME (r, Sigma), then sweeps the production seed path over Tc_seed to
# see which branch each seed lands in.  Answers "does the column solve admit multiple
# roots at node 9?" (the S-curve hypothesis for the Group-2 residual stall).  Same
# include-the-.cpp consumer order as the walk probe; needs /openmp on THIS target.
# Safe to delete along with tools/slim_multiroot_probe.cpp.
add_executable(slim-multiroot-probe tools/slim_multiroot_probe.cpp)
target_include_directories(slim-multiroot-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)
if(OpenMP_CXX_FOUND)
    target_link_libraries(slim-multiroot-probe PRIVATE OpenMP::OpenMP_CXX)
endif()

# PLUMBING CONSISTENCY AUDIT probe (DELETABLE, audit-only): replicates Path A
# (calibrate_seed_to_manifold's ColumnCoupledInputs + 2-D/1-D seed builder) and Path B
# (eval_node_coupled's ColumnCoupledInputs + solve_column_coupled) at the SAME node and
# prints a field-by-field comparison + the manifold round-trip verdict.  No full radial
# relax (column solves only).  Same include-the-.cpp consumer order as the walk probe;
# does NOT link grrt.  Safe to delete along with tools/slim_plumbing_audit_probe.cpp.
add_executable(slim-plumbing-audit-probe tools/slim_plumbing_audit_probe.cpp)
target_include_directories(slim-plumbing-audit-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# FOCUSED HARD-NODE probe (DELETABLE, diagnostic): isolates the solve_column_coupled
# HIGH-Σ failure at ONE representative node (a=0.9, f_Edd=0.01, Σ~5e4) — builds the
# f_adv=0 manifold column via build_coupled_seed, shows solve_column_coupled's internal
# seed lands on the WRONG branch, and proves (decisive experiment) that seeding the
# augmented Newton from the manifold column converges.  Same include-the-.cpp consumer
# order as the walk probe; does NOT link grrt.  Safe to delete along with
# tools/slim_coupled_hard_probe.cpp.
add_executable(slim-coupled-hard-probe tools/slim_coupled_hard_probe.cpp)
target_include_directories(slim-coupled-hard-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# Σ0 n_z RESOLUTION probe (DELETABLE, diagnostic): decides whether the Σ0≈8.6e3
# saturation ceiling at the high-Σ failing node (a=0.9, f_Edd=0.01, Σ~5e4) is a
# RESOLUTION ARTIFACT of the relax bring-up n_z=24 or a PHYSICAL column-capacity
# limit, by sweeping the base column solver's Σ0 ceiling + the coupled solve over
# n_z ∈ {24,48,96,200,400}.  Same include-the-.cpp consumer order as the hard
# probe; does NOT link grrt.  Safe to delete along with tools/slim_coupled_nz_probe.cpp.
add_executable(slim-coupled-nz-probe tools/slim_coupled_nz_probe.cpp)
target_include_directories(slim-coupled-nz-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# Σ↔V CHECK probe (DELETABLE, diagnostic): the resume-action check of the 2026-06-30
# handoff §0 — is the demanded inner Σ≈5e4 g/cm² PHYSICAL or a thin-disk-seed artifact?
# Per-node mass-conservation inversion (the V required to carry Ṁ through the column's
# Σ0 capacity) + the column ceiling (fresh, n_z=96) + one-zone c_s.  Decides Option A
# (transonic-aware seed) vs Option C (convection #13).  Same include-the-.cpp consumer
# order as the hard probe; does NOT link grrt.  Safe to delete with the .cpp.
add_executable(slim-sigma-v-probe tools/slim_sigma_v_probe.cpp)
target_include_directories(slim-sigma-v-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# GLOBAL-SEED DUMP probe (DELETABLE, diagnostic): Task 1 of the seed-pivot —
# dump the production build_slim_disk_seed(a,f_Edd) RAW (no relax) to judge slim
# vs torus + column-feasibility of its Σ, before wiring it into relax_coupled.
# include-the-.cpp (opacity + slim_disk_radial); does NOT link grrt. Delete with the .cpp.
add_executable(slim-seed-dump-probe tools/slim_seed_dump_probe.cpp)
target_include_directories(slim-seed-dump-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# SONIC-Σ vs CAPACITY probe (DELETABLE, diagnostic): the rigorous convection gate —
# self-consistent sonic-point Σ (column's own c_s) vs the pure-radiative column Σ0
# capacity at f_Edd=0.9. Confirms whether the column genuinely cannot hold the disk
# (→ convection #13). include-the-.cpp; does NOT link grrt. Delete with the .cpp.
add_executable(slim-sonic-sigma-probe tools/slim_sonic_sigma_probe.cpp)
target_include_directories(slim-sonic-sigma-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# CONVECTION VERIFY probe (DELETABLE): confirms convection triggers + flattens ∇ in the
# deep-rad-pressure column (the midplane-cooling mechanism behind the β 7e-6→3e-3 shift).
add_executable(slim-convection-verify-probe tools/slim_convection_verify_probe.cpp)
target_include_directories(slim-convection-verify-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# BASE-INFEASIBLE per-node CLASSIFIER probe (DELETABLE, diagnostic): at f_Edd=0.001 a=0.9,
# classifies EACH failing radial node's column as H1 (seed-placement: Σ above the column's
# Σ0 capacity, no reachable column) vs H2 (solver-robustness: a feasible column exists near
# the target but the direct solve cannot reach it) — via the Σ0-capacity ratio + a fine
# continuation from a known-feasible anchor, plus a convective/chatter landscape.  Same
# include-the-.cpp consumer order as the walk probe; does NOT link grrt.  Safe to delete
# along with tools/slim_base_classify_probe.cpp.
add_executable(slim-base-classify-probe tools/slim_base_classify_probe.cpp)
target_include_directories(slim-base-classify-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# n_z-REFINEMENT + MULTI-START disambiguator probe (DELETABLE, diagnostic): resolves the
# three-way ambiguity (H1 physical Σ-wall / ARTIFACT n_z=24 resolution / H2-basin solver)
# behind the base-infeasible wall at f_Edd=0.001 a=0.9.  Part 1 re-ratios the Σ0 capacity
# at n_z=96; Part 2 walks Σ0_max + direct target-Σ solve across n_z {24,48,96}; Part 3
# multi-starts solve_column_coupled at the pinned demanded Σ.  Same include-the-.cpp order
# as the base-classify probe; does NOT link grrt.  Delete with tools/slim_nz_refine_probe.cpp.
add_executable(slim-nz-refine-probe tools/slim_nz_refine_probe.cpp)
target_include_directories(slim-nz-refine-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# Richardson capacity probe (DELETABLE, diagnostic): Σ0 capacity vs n_z {96,128,192,256}
# for the resistant nodes 3 & 9 -> extrapolate the n_z→∞ limit to decide RESOLUTION
# (curable) vs PHYSICAL WALL (monolithic rebuild).  Delete with tools/slim_richardson_probe.cpp.
add_executable(slim-richardson-probe tools/slim_richardson_probe.cpp)
target_include_directories(slim-richardson-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

# Full-18 feasibility @ n_z=256 (DELETABLE, diagnostic): all 18 radial nodes, does each
# column solve at its demanded Σ at high vertical resolution.  Delete with tools/slim_full256_probe.cpp.
add_executable(slim-full256-probe tools/slim_full256_probe.cpp)
target_include_directories(slim-full256-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)

