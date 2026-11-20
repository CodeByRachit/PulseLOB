find_package(Threads REQUIRED)

add_executable(pulselob_tests
  test_fixed_point.cpp
  test_fixed_matrix.cpp
  test_model_shape.cpp
  test_market_event.cpp
  test_spsc_ring.cpp
  test_time.cpp
  test_version.cpp)
target_link_libraries(pulselob_tests PRIVATE
  pulselob::pulselob pulselob::warnings GTest::gtest_main Threads::Threads)

include(GoogleTest)
gtest_discover_tests(pulselob_tests DISCOVERY_MODE PRE_TEST)
