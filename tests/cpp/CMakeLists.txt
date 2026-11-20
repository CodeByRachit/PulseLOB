find_package(Threads REQUIRED)

set(PULSELOB_TEST_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../support")
set(PULSELOB_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures")

add_executable(pulselob_tests
  test_fixed_point.cpp
  test_fixed_matrix.cpp
  test_model_shape.cpp
  test_market_event.cpp
  test_spsc_ring.cpp
  test_time.cpp
  test_version.cpp
  test_psi.cpp
  test_linalg.cpp
  test_kernel.cpp
  test_estimator.cpp
  test_stability.cpp
  test_forecast.cpp
  test_control.cpp
  test_engine.cpp
  test_stress.cpp
  test_regime.cpp
  test_concurrency.cpp
  test_crosscheck.cpp)
target_include_directories(pulselob_tests PRIVATE "${PULSELOB_TEST_SUPPORT_DIR}")
target_compile_definitions(pulselob_tests PRIVATE PULSELOB_FIXTURE_DIR="${PULSELOB_FIXTURE_DIR}")
target_link_libraries(pulselob_tests PRIVATE
  pulselob::pulselob pulselob::warnings GTest::gtest_main Threads::Threads)

# Zero-allocation proof: replaces the global operator new/delete, so it is its own binary.
add_executable(pulselob_alloc_tests alloc/test_zero_alloc.cpp)
target_include_directories(pulselob_alloc_tests PRIVATE "${PULSELOB_TEST_SUPPORT_DIR}")
target_link_libraries(pulselob_alloc_tests PRIVATE
  pulselob::pulselob pulselob::warnings GTest::gtest_main Threads::Threads)

include(GoogleTest)
gtest_discover_tests(pulselob_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60)
gtest_discover_tests(pulselob_alloc_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60)
