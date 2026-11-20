set(PULSELOB_TEST_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tests/support")

add_executable(pulselob_bench bench_core.cpp bench_engine.cpp)
target_include_directories(pulselob_bench PRIVATE "${PULSELOB_TEST_SUPPORT_DIR}")
target_link_libraries(pulselob_bench PRIVATE pulselob::pulselob pulselob::warnings benchmark::benchmark_main)

# Latency distributions, component breakdown, split-mode experiment (docs/benchmarks.md).
add_executable(pulselob_latency latency_report.cpp)
target_include_directories(pulselob_latency PRIVATE "${PULSELOB_TEST_SUPPORT_DIR}")
find_package(Threads REQUIRED)
target_link_libraries(pulselob_latency PRIVATE pulselob::pulselob pulselob::warnings Threads::Threads)

# Benchmark-only build with FMA contraction allowed, to measure the cost of -ffp-contract=off
# (docs/architecture.md Section 4.3). The later flag wins over the core target's policy.
if(PULSELOB_BENCH_FP_CONTRACT)
  target_compile_options(pulselob_bench PRIVATE -ffp-contract=fast)
  target_compile_options(pulselob_latency PRIVATE -ffp-contract=fast)
endif()
