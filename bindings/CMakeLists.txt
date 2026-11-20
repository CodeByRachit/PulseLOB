pybind11_add_module(_core MODULE module.cpp)
target_link_libraries(_core PRIVATE pulselob::pulselob pulselob::warnings)

# scikit-build-core places this inside the `pulselob` package of the wheel.
install(TARGETS _core LIBRARY DESTINATION pulselob)
