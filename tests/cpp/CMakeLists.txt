set(MY_KERNELS_TEST_SUPPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

find_package(GTest CONFIG QUIET)

if(NOT GTest_FOUND)
  include(FetchContent)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

function(my_kernels_add_cpp_test target source)
  set(options RUN_SERIAL)
  set(one_value_args DISCOVERY_TIMEOUT TIMEOUT)
  set(multi_value_args LABELS)
  cmake_parse_arguments(PARSE_ARGV 2 ARG "${options}" "${one_value_args}" "${multi_value_args}")

  add_executable(${target}
    ${source}
  )

  target_include_directories(${target} PRIVATE
    ${MY_KERNELS_TEST_SUPPORT_DIR}
  )

  target_link_libraries(${target} PRIVATE
    GTest::gtest_main
    my_kernels::my_kernels
  )

  set(test_properties
    LABELS "${ARG_LABELS}"
    TIMEOUT "${ARG_TIMEOUT}"
  )
  if(ARG_RUN_SERIAL)
    list(APPEND test_properties RUN_SERIAL TRUE)
  endif()

  gtest_discover_tests(${target}
    TEST_PREFIX "${target}."
    DISCOVERY_TIMEOUT ${ARG_DISCOVERY_TIMEOUT}
    PROPERTIES ${test_properties}
  )
endfunction()

my_kernels_add_cpp_test(
  test_add_cpu
  add/test_add_cpu.cpp
  LABELS "cpp;unit;cpu"
  DISCOVERY_TIMEOUT 60
  TIMEOUT 60
)

if(MY_KERNELS_WITH_CUDA)
  my_kernels_add_cpp_test(
    test_add_cuda
    add/test_add_cuda.cpp
    LABELS "cpp;unit;cuda"
    DISCOVERY_TIMEOUT 60
    TIMEOUT 120
    RUN_SERIAL
  )
endif()
