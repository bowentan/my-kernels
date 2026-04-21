add_subdirectory(cpp)

if(BUILD_PYTHON_TESTS AND BUILD_PYTHON_EXTENSION)
  if(CMAKE_HOST_WIN32)
    set(MY_KERNELS_PYTHONPATH_SEP ";")
  else()
    set(MY_KERNELS_PYTHONPATH_SEP ":")
  endif()

  set(MY_KERNELS_TEST_PYTHONPATH
    "${PROJECT_BINARY_DIR}/python${MY_KERNELS_PYTHONPATH_SEP}${PROJECT_SOURCE_DIR}"
  )

  function(my_kernels_add_pytest name path)
    set(options RUN_SERIAL)
    set(one_value_args TIMEOUT)
    set(multi_value_args LABELS)
    cmake_parse_arguments(PARSE_ARGV 2 ARG "${options}" "${one_value_args}" "${multi_value_args}")

    add_test(
      NAME "${name}"
      COMMAND
        ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${MY_KERNELS_TEST_PYTHONPATH}"
          "MY_KERNELS_DISABLE_JIT=1"
          "MY_KERNELS_BUILD_PYTHON_ROOT=${PROJECT_BINARY_DIR}/python"
          "${Python3_EXECUTABLE}" -m pytest "${PROJECT_SOURCE_DIR}/${path}"
    )

    set(test_properties LABELS "${ARG_LABELS}")
    if(ARG_TIMEOUT)
      list(APPEND test_properties TIMEOUT "${ARG_TIMEOUT}")
    endif()
    if(ARG_RUN_SERIAL)
      list(APPEND test_properties RUN_SERIAL TRUE)
    endif()
    set_tests_properties("${name}" PROPERTIES ${test_properties})
  endfunction()

  my_kernels_add_pytest(
    "python.correctness"
    "tests/python/correctness"
    LABELS "python;correctness"
    TIMEOUT 120
  )

  my_kernels_add_pytest(
    "python.dispatch"
    "tests/python/dispatch"
    LABELS "python;dispatch"
    TIMEOUT 120
  )

  my_kernels_add_pytest(
    "python.perf_smoke"
    "tests/python/perf_smoke"
    LABELS "python;perf"
    TIMEOUT 300
    RUN_SERIAL
  )
endif()
