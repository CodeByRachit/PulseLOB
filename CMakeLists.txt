cmake_minimum_required(VERSION 3.25)

# VERSION is the single source of truth for the project version (see README "Versioning").
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PULSELOB_VERSION_FULL LIMIT_COUNT 1)
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" PULSELOB_VERSION_CORE "${PULSELOB_VERSION_FULL}")
if(NOT PULSELOB_VERSION_CORE)
  message(FATAL_ERROR "VERSION file does not start with MAJOR.MINOR.PATCH: '${PULSELOB_VERSION_FULL}'")
endif()

project(PulseLOB VERSION ${PULSELOB_VERSION_CORE} LANGUAGES CXX
        DESCRIPTION "Online cross-venue order-flow toxicity with marked Hawkes processes")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PulseLOBOptions)
include(PulseLOBWarnings)
include(PulseLOBSanitizers)
include(PulseLOBDependencies)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The numerical contract (docs/math_spec.md Sections 4 and 10) forbids value-unsafe FP optimizations.
foreach(_flags_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELWITHDEBINFO)
  if("${${_flags_var}}" MATCHES "fast-math|Ofast")
    message(FATAL_ERROR "${_flags_var} contains -ffast-math/-Ofast, which breaks the numerical contract")
  endif()
endforeach()

configure_file(include/pulselob/version.hpp.in
               "${PROJECT_BINARY_DIR}/include/pulselob/version.hpp" @ONLY)

# ---- Core library: header-only, templated on the model shape -------------------------------
add_library(pulselob INTERFACE)
add_library(pulselob::pulselob ALIAS pulselob)
target_compile_features(pulselob INTERFACE cxx_std_20)
target_include_directories(pulselob INTERFACE
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>")
pulselob_apply_fp_policy(pulselob)

if(PULSELOB_NATIVE_ARCH)
  pulselob_apply_native_arch(pulselob)
endif()

# ---- Header contract: every public header is self-contained and exception/RTTI-free ---------
file(GLOB_RECURSE PULSELOB_PUBLIC_HEADERS CONFIGURE_DEPENDS
     "${PROJECT_SOURCE_DIR}/include/pulselob/*.hpp")
set(_hdr_check_dir "${PROJECT_BINARY_DIR}/header_check")
set(_hdr_check_sources "")
foreach(_hdr IN LISTS PULSELOB_PUBLIC_HEADERS)
  file(RELATIVE_PATH _rel "${PROJECT_SOURCE_DIR}/include" "${_hdr}")
  string(MAKE_C_IDENTIFIER "${_rel}" _id)
  set(_src "${_hdr_check_dir}/${_id}.cpp")
  file(CONFIGURE OUTPUT "${_src}" CONTENT "#include \"${_rel}\"\n" @ONLY)
  list(APPEND _hdr_check_sources "${_src}")
endforeach()
add_library(pulselob_header_check OBJECT ${_hdr_check_sources})
target_link_libraries(pulselob_header_check PRIVATE pulselob::pulselob pulselob::warnings)
target_compile_options(pulselob_header_check PRIVATE -fno-exceptions -fno-rtti)

# ---- Sub-projects ----------------------------------------------------------------------------
if(PULSELOB_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

if(PULSELOB_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

if(PULSELOB_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()

message(STATUS "PulseLOB ${PULSELOB_VERSION_FULL}: tests=${PULSELOB_BUILD_TESTS} "
               "benchmarks=${PULSELOB_BUILD_BENCHMARKS} python=${PULSELOB_BUILD_PYTHON} "
               "sanitizers='${PULSELOB_SANITIZERS}' native=${PULSELOB_NATIVE_ARCH}")
