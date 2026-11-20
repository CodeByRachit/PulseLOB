add_executable(pulselob_bench bench_core.cpp)
target_link_libraries(pulselob_bench PRIVATE pulselob::pulselob pulselob::warnings benchmark::benchmark_main)
