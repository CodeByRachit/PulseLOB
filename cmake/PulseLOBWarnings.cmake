# pulselob::warnings carries the warning set for PulseLOB's own targets only.
# Third-party code is fetched with SYSTEM include directories and never sees these flags.
add_library(pulselob_warnings INTERFACE)
add_library(pulselob::warnings ALIAS pulselob_warnings)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(pulselob_warnings INTERFACE
    -Wall -Wextra -Wpedantic
    -Wconversion -Wsign-conversion -Wdouble-promotion
    -Wshadow -Wold-style-cast -Wcast-align -Wcast-qual
    -Wnon-virtual-dtor -Woverloaded-virtual -Wnull-dereference
    -Wimplicit-fallthrough -Wformat=2 -Wundef)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(pulselob_warnings INTERFACE
      -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wuseless-cast)
  endif()
  if(PULSELOB_WARNINGS_AS_ERRORS)
    target_compile_options(pulselob_warnings INTERFACE -Werror)
  endif()
else()
  message(WARNING "PulseLOB is only tested with GCC and Clang; no warning set for ${CMAKE_CXX_COMPILER_ID}")
endif()

# Floating-point policy (docs/architecture.md Section 4.3): IEEE semantics, no FMA contraction,
# so results are reproducible across compilers and optimization levels on the same platform.
function(pulselob_apply_fp_policy target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target} INTERFACE -ffp-contract=off -fno-fast-math)
  endif()
endfunction()

include(CheckCXXCompilerFlag)
function(pulselob_apply_native_arch target)
  check_cxx_compiler_flag("-march=native" PULSELOB_HAS_MARCH_NATIVE)
  if(PULSELOB_HAS_MARCH_NATIVE)
    target_compile_options(${target} INTERFACE -march=native)
  else()
    check_cxx_compiler_flag("-mcpu=native" PULSELOB_HAS_MCPU_NATIVE)
    if(PULSELOB_HAS_MCPU_NATIVE)
      target_compile_options(${target} INTERFACE -mcpu=native)
    endif()
  endif()
endfunction()
