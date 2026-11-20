# Native command-line tools.
add_executable(pulselob_replay replay.cpp)
set_target_properties(pulselob_replay PROPERTIES OUTPUT_NAME pulselob-replay)
target_link_libraries(pulselob_replay PRIVATE pulselob::pulselob pulselob::warnings)

if(PULSELOB_BUILD_PYTHON)
  # Shipped inside the wheel as pulselob/bin/pulselob-replay (pulselob.native_replay_path()).
  install(TARGETS pulselob_replay RUNTIME DESTINATION pulselob/bin)
endif()
