# Build options. Defaults suit a developer checkout; CI and pip builds override them.
option(PULSELOB_BUILD_TESTS "Build the C++ unit tests (GoogleTest)" ON)
option(PULSELOB_BUILD_BENCHMARKS "Build the micro-benchmarks (Google Benchmark)" OFF)
option(PULSELOB_BUILD_TOOLS "Build the native command-line tools (pulselob-replay)" ON)
option(PULSELOB_BUILD_PYTHON "Build the pybind11 extension module pulselob._core" OFF)
option(PULSELOB_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(PULSELOB_NATIVE_ARCH "Tune for the build machine (-march/-mcpu=native); benchmarks only" OFF)
set(PULSELOB_SANITIZERS "" CACHE STRING
    "Semicolon-separated sanitizers: address;undefined or thread (not both groups)")
option(PULSELOB_BENCH_FP_CONTRACT "Benchmarks only: allow FMA contraction to measure the cost of -ffp-contract=off" OFF)
