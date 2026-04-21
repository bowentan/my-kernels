set(MY_KERNELS_BENCHMARK_SUPPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(bench_add
  add/bench_add.cpp
)

target_include_directories(bench_add PRIVATE
  ${MY_KERNELS_BENCHMARK_SUPPORT_DIR}
)

target_link_libraries(bench_add PRIVATE
  my_kernels::my_kernels
)
