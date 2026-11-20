# Sanitizers are applied to every target in the build (including fetched test frameworks), so that
# instrumented and uninstrumented code never mix.
if(PULSELOB_SANITIZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "PULSELOB_SANITIZERS requires GCC or Clang")
  endif()
  if("thread" IN_LIST PULSELOB_SANITIZERS AND "address" IN_LIST PULSELOB_SANITIZERS)
    message(FATAL_ERROR "ThreadSanitizer cannot be combined with AddressSanitizer")
  endif()
  list(JOIN PULSELOB_SANITIZERS "," _san)
  add_compile_options(-fsanitize=${_san} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=${_san})
endif()
